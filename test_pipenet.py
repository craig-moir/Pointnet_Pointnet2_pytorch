import os
import shutil
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import sys
import importlib
import torch
import argparse
import logging
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

AUTOPIPEFIT_DIR = os.path.join("/", "home", "craig", "autopipefit")
sys.path.insert(0, os.path.join(AUTOPIPEFIT_DIR, "pysrc"))
AUTOPIPEFIT_DIR_2 = os.path.join(ROOT_DIR, "autopipefit")
sys.path.insert(0, os.path.join(AUTOPIPEFIT_DIR_2, "pysrc"))

from pyautopipe.virtual_scan import VirtualScanImage
import pyautopipe.open3d as ao3d

import pyautopipe
pyautopipe.setResourcesPath(os.path.join(AUTOPIPEFIT_DIR_2, 'resources'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[2, 10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def main(args):
    
    # network
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = "pipenet" #os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    
    # model creation
    pipe_data_list = pyautopipe.loadASMEPipes()
    elbow_data_list = pyautopipe.loadASMEElbows()

    noise_sigma = 3
    nominal_diameter = "150"
    realism = 1

    # creating model
    model = pyautopipe.ParameterisedModel()

    start = np.identity(4)
    start[2, 3] = 10000
    pipe = pyautopipe.ParameterisedPipe(pipe_data_list[nominal_diameter], start, 2000)
    model.connect(pipe, 1)

    new_elbow_1 = pyautopipe.ParameterisedElbow(elbow_data_list[nominal_diameter]["long"]["90"], np.identity(4))
    new_pipe_1 = pyautopipe.ParameterisedPipe(pipe_data_list[nominal_diameter], start, 10000)

    pipe.connectAndAlignSquareToConnection(new_elbow_1, 0, 1)

    T = np.identity(4)
    # T[:3, :3] = R.from_euler('x', random.uniform(-180, 180), degrees=True).as_matrix()

    new_elbow_1.currentTransform = new_elbow_1.currentTransform.dot(T)

    new_elbow_1.connectAndAlignSquareToConnection(new_pipe_1, 0, 1)

    new_elbow_2 = pyautopipe.ParameterisedElbow(elbow_data_list[nominal_diameter]["long"]["90"], np.identity(4))
    new_pipe_2 = pyautopipe.ParameterisedPipe(pipe_data_list[nominal_diameter], start, 2000)

    new_pipe_1.connectAndAlignSquareToConnection(new_elbow_2, 0, 1)

    T = np.identity(4)
    # T[:3, :3] = R.from_euler('x', random.uniform(-180, 180), degrees=True).as_matrix()

    new_elbow_2.currentTransform = new_elbow_2.currentTransform.dot(T)

    new_elbow_2.connectAndAlignSquareToConnection(new_pipe_2, 0, 1)

    scan = VirtualScanImage(ignore_floor_and_walls=True, realism=realism)
    scan.add_parameterised_model(model)

    origin = np.array([
        3000, 
        -7000, 
        10000
    ])
    scan.take_scan(origin, noise_sigma=noise_sigma, max_distance=50000)

    # origin = np.array([
    #     -3000, 
    #     -3000, 
    #     11000
    # ])
    # scan.take_scan(origin, noise_sigma=noise_sigma, max_distance=50000)

    # hit_geom = scan.hit_geometry[0]
    # for i in range(1, len(scan.hit_geometry)):
    #     hit_geom = hit_geom.append(scan.hit_geometry[i], 0)

    # print(np.dot(pipe.currentTransform, np.array([1, 0, 0, 0])))
    # pipe_dir = np.dot(new_pipe_1.currentTransform, np.array([1, 0, 0, 0]))[:3]
    # print("pipe direction:", pipe_dir)
    # print(np.dot(new_pipe_2.currentTransform, np.array([1, 0, 0, 0])))
    # exit()

    # print(scan.pointcloud.point["positions"].shape)
    # print(hit_geom == 2)
    # print(scan.pointcloud.point["positions"][hit_geom == 0])

    pcd_master = ao3d.Open3DToPCL(scan.pointcloud).copy()

    # sphere_f = ao3d.CloudToOpen3D(pcd)
    # print(len(pcd))
    # ao3d.visualise(sphere_f)

    # scan.visualise()
    # for i in range(len(scan.geometry_ids)):
    #     ao3d.visualise(scan.pointcloud.select_by_mask(hit_geom == scan.geometry_ids[i]).to_legacy())

    # middle_pipe_geoid = 2
    # eq_points = scan.pointcloud.select_by_mask(hit_geom == scan.geometry_ids[middle_pipe_geoid]).point["positions"]
    # number_of_points = eq_points.shape[0]

    scan_points = scan.pointcloud.point["positions"]
    number_of_points = scan_points.shape[0]
    
    point_sample_list = []

    number_of_samples = 10
    for i in tqdm(range(number_of_samples), total=number_of_samples):
        
        sphere_query_radius = 500 # mm
        
        number_of_points_in_sample = 0
        sample_counter = 0
        min_number_of_points_per_case = 1024
        while number_of_points_in_sample < min_number_of_points_per_case:
        
            random_index = random.randint(0, number_of_points-1)
            random_point = scan_points[random_index]
            #print(number_of_points, random_index, random_point.numpy())
            sphere_pcl = pcd_master.sphereFilter(random_point.numpy(), sphere_query_radius, False)
            sphere_o3d = ao3d.CloudToOpen3dTensor(sphere_pcl)
            number_of_points_in_sample = sphere_o3d.point["positions"].shape[0]
            sample_counter += 1
            if sample_counter > 10:
                print("number of points in sample is too low", number_of_points_in_sample, "Exiting...")
                exit()
        
        point_set = sphere_o3d.point["positions"].numpy().astype(np.float32)
                        
        point_set = np.hstack((point_set, np.zeros(point_set.shape)))
        
        if args.use_uniform_sample:
            point_set = farthest_point_sample(point_set, args.num_point)
        else:
            point_set = point_set[0:args.num_point, :]
        
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not args.use_normals:
            point_set = point_set[:, 0:3]
            
        points = torch.tensor(point_set)
        
        point_sample_list.append(points)
        
    points = torch.stack(point_sample_list, dim=0)
    points = points.to(torch.float32)
        
    # print(points)
    # exit()
        
    if not args.use_cpu:
        points = points.cuda()

    points = points.transpose(2, 1)
    pred, pred_direction, pred_normal, pred_radius, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    
    print(pred_choice, pred, pred_direction, pred_normal, pred_radius)
    exit()

    # print("point:", random_point.numpy())
    # pipe_pos = new_pipe_1.currentTransform[:3, 3]
    # print("pipe position:", pipe_pos)
    # pipe_to_point = random_point.numpy() - pipe_pos
    # print("pipe to point:", pipe_to_point)
    # centerline_to_point = pipe_to_point - pipe_to_point * pipe_dir
    # print("distance point to centerline:", np.linalg.norm(centerline_to_point))
    # point_normal = centerline_to_point / np.linalg.norm(centerline_to_point)
    # print("normal at point:", point_normal)
    
    # for j in range(number_of_rotation_transforms):
        
        # sphere_o3d_copy = sphere_o3d.clone()
        
        # T1 = np.identity(4)
        # T1[:3, :3] = R.from_euler('x', random.uniform(-180, 180), degrees=True).as_matrix()
        
        # T2 = np.identity(4)
        # T2[:3, :3] = R.from_euler('y', random.uniform(-180, 180), degrees=True).as_matrix()
        
        # T = T2.dot(T1)
        
        # sphere_o3d_copy.transform(T)
        
        # transformed_point_normal = np.dot(T[:3, :3], point_normal)
        # transformed_pipe_dir = np.dot(T[:3, :3], pipe_dir)
        
        # assert(np.linalg.norm(transformed_point_normal) - 1 < 1e-6)
        # assert(np.linalg.norm(transformed_pipe_dir) - 1 < 1e-6)
        # # print(transformed_point_normal, transformed_pipe_dir, np.cross(transformed_point_normal, transformed_pipe_dir), np.linalg.norm(np.cross(transformed_point_normal, transformed_pipe_dir)))
        # assert(np.linalg.norm(np.cross(transformed_point_normal, transformed_pipe_dir)) - 1 < 1e-6)
    
        # numpy_positions = sphere_o3d_copy.point["positions"].numpy()
    
        # numpy_positions = np.hstack((numpy_positions, np.zeros(numpy_positions.shape)))
        # # print(sphere_o3d.point["positions"])
        # # ao3d.visualise(sphere_o3d_copy.to_legacy())
        # case_filename = os.path.join(YES_PIPE_DIR, f"pipe_{yes_cnt:05d}_d_{'_'.join(transformed_pipe_dir.astype(str))}_n_{'_'.join(transformed_point_normal.astype(str))}_r_{float(nominal_diameter)/2}.txt")
        # print(f"saving {case_filename}")
        # np.savetxt(case_filename, numpy_positions, delimiter=',')
        # yes_cnt += 1
        
        
    if not args.use_cpu:
        points, target = points.cuda(), target.cuda()

if __name__ == '__main__':
    args = parse_args()
    main(args)
