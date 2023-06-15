import os
import shutil
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
import open3d as o3d

random.seed(42)

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_DIR = os.path.join(ROOT_DIR, "data")

MODELNET40_DIR = os.path.join(DATA_DIR, "modelnet40_normal_resampled")

PIPEDATA_DIR = os.path.join(DATA_DIR, "pipes")
os.makedirs(PIPEDATA_DIR, exist_ok=True)

NO_PIPE_DIR = os.path.join(PIPEDATA_DIR, "no_pipe")
os.makedirs(NO_PIPE_DIR, exist_ok=True)
YES_PIPE_DIR = os.path.join(PIPEDATA_DIR, "pipe")
os.makedirs(YES_PIPE_DIR, exist_ok=True)

if 1:
        
    cnt = 0
    for i, d in enumerate(os.listdir(MODELNET40_DIR)):
        
        CASE_DIR = os.path.join(MODELNET40_DIR, d)
        
        if os.path.isdir(CASE_DIR):
            
            txt_filenames = [f for f in os.listdir(CASE_DIR) if f.endswith(".txt")]
            
            for txt_file in txt_filenames:
                txt_abs_path = os.path.join(CASE_DIR, txt_file)
                
                new_filename = os.path.join(NO_PIPE_DIR, f"no_pipe_{cnt:05d}_d_0_0_0_n_0_0_0_r_0.txt")
                
                print(f"copy {txt_abs_path} to {new_filename}")
                shutil.copy(txt_abs_path, new_filename)
                
                cnt += 1
            
if 1:
    
    import sys
    AUTOPIPEFIT_DIR = os.path.join("/", "home", "craig", "autopipefit")
    sys.path.insert(0, os.path.join(AUTOPIPEFIT_DIR, "pysrc"))
    AUTOPIPEFIT_DIR_2 = os.path.join(ROOT_DIR, "autopipefit")
    sys.path.insert(0, os.path.join(AUTOPIPEFIT_DIR_2, "pysrc"))
    
    from pyautopipe.virtual_scan import VirtualScanImage
    import pyautopipe.open3d as ao3d

    import pyautopipe
    pyautopipe.setResourcesPath(os.path.join(AUTOPIPEFIT_DIR_2, 'resources'))
    
    pipe_data_list = pyautopipe.loadASMEPipes()
    elbow_data_list = pyautopipe.loadASMEElbows()
    
    yes_cnt = 0
    
    for noise_sigma in [2, 3, 4]:
        for nominal_diameter in ["40", "50", "65", "80", "100", "150", "200", "250", "300"]:
            for realism in [1]:
                for number_of_scans in [1, 2]:
        
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
                    
                    if number_of_scans == 2:
                        origin = np.array([
                            -3000, 
                            -3000, 
                            11000
                        ])
                        scan.take_scan(origin, noise_sigma=noise_sigma, max_distance=50000)
                    
                    hit_geom = scan.hit_geometry[0]
                    for i in range(1, len(scan.hit_geometry)):
                        hit_geom = hit_geom.append(scan.hit_geometry[i], 0)
                    
                    # print(np.dot(pipe.currentTransform, np.array([1, 0, 0, 0])))
                    pipe_dir = np.dot(new_pipe_1.currentTransform, np.array([1, 0, 0, 0]))[:3]
                    print("pipe direction:", pipe_dir)
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
                    
                    middle_pipe_geoid = 2
                    eq_points = scan.pointcloud.select_by_mask(hit_geom == scan.geometry_ids[middle_pipe_geoid]).point["positions"]
                    number_of_points = eq_points.shape[0]
                    
                    number_of_samples = 10
                    number_of_rotation_transforms = 10
                    for i in range(number_of_samples):
                        
                        sphere_query_radius = 500 # mm
                        
                        number_of_points_in_sample = 0
                        sample_counter = 0
                        min_number_of_points_per_case = 1024
                        while number_of_points_in_sample < min_number_of_points_per_case:
                        
                            random_index = random.randint(0, number_of_points-1)
                            random_point = eq_points[random_index]
                            #print(number_of_points, random_index, random_point.numpy())
                            sphere_pcl = pcd_master.sphereFilter(random_point.numpy(), sphere_query_radius, False)
                            sphere_o3d = ao3d.CloudToOpen3dTensor(sphere_pcl)
                            number_of_points_in_sample = sphere_o3d.point["positions"].shape[0]
                            sample_counter += 1
                            if sample_counter > 10:
                                print("number of points in sample is too low", number_of_points_in_sample, "Exiting...")
                                exit()
                        
                        print("point:", random_point.numpy())
                        pipe_pos = new_pipe_1.currentTransform[:3, 3]
                        print("pipe position:", pipe_pos)
                        pipe_to_point = random_point.numpy() - pipe_pos
                        print("pipe to point:", pipe_to_point)
                        centerline_to_point = pipe_to_point - pipe_to_point * pipe_dir
                        print("distance point to centerline:", np.linalg.norm(centerline_to_point))
                        point_normal = centerline_to_point / np.linalg.norm(centerline_to_point)
                        print("normal at point:", point_normal)
                        
                        for j in range(number_of_rotation_transforms):
                            
                            sphere_o3d_copy = sphere_o3d.clone()
                            
                            T1 = np.identity(4)
                            T1[:3, :3] = R.from_euler('x', random.uniform(-180, 180), degrees=True).as_matrix()
                            
                            T2 = np.identity(4)
                            T2[:3, :3] = R.from_euler('y', random.uniform(-180, 180), degrees=True).as_matrix()
                            
                            T = T2.dot(T1)
                            
                            sphere_o3d_copy.transform(T)
                            
                            transformed_point_normal = np.dot(T[:3, :3], point_normal)
                            transformed_pipe_dir = np.dot(T[:3, :3], pipe_dir)
                            
                            assert(np.linalg.norm(transformed_point_normal) - 1 < 1e-6)
                            assert(np.linalg.norm(transformed_pipe_dir) - 1 < 1e-6)
                            # print(transformed_point_normal, transformed_pipe_dir, np.cross(transformed_point_normal, transformed_pipe_dir), np.linalg.norm(np.cross(transformed_point_normal, transformed_pipe_dir)))
                            assert(np.linalg.norm(np.cross(transformed_point_normal, transformed_pipe_dir)) - 1 < 1e-6)
                        
                            numpy_positions = sphere_o3d_copy.point["positions"].numpy()
                        
                            numpy_positions = np.hstack((numpy_positions, np.zeros(numpy_positions.shape)))
                            # print(sphere_o3d.point["positions"])
                            # ao3d.visualise(sphere_o3d_copy.to_legacy())
                            case_filename = os.path.join(YES_PIPE_DIR, f"pipe_{yes_cnt:05d}_d_{'_'.join(transformed_pipe_dir.astype(str))}_n_{'_'.join(transformed_point_normal.astype(str))}_r_{float(nominal_diameter)/2}.txt")
                            print(f"saving {case_filename}")
                            np.savetxt(case_filename, numpy_positions, delimiter=',')
                            yes_cnt += 1
                        
if 1:
    no_filenames = os.listdir(NO_PIPE_DIR)[:]
    yes_filenames = os.listdir(YES_PIPE_DIR)
    
    no_filenames = [n.replace(".txt", "") for n in no_filenames]
    yes_filenames = [y.replace(".txt", "") for y in yes_filenames]
    
    train_no_filenames = no_filenames[:int(len(no_filenames) * 0.8)]
    train_yes_filenames = yes_filenames[:int(len(yes_filenames) * 0.8)]
    
    test_no_filenames = no_filenames[int(len(no_filenames) * 0.8):]
    test_yes_filenames = yes_filenames[int(len(yes_filenames) * 0.8):]
    
    print(len(no_filenames), len(train_no_filenames), len(test_no_filenames), len(train_no_filenames) + len(test_no_filenames))
    print(len(yes_filenames), len(train_yes_filenames), len(test_yes_filenames), len(train_yes_filenames) + len(test_yes_filenames))
    
    train_file_filename = os.path.join(PIPEDATA_DIR, "modelnet40_train.txt")
    train_file = open(train_file_filename, "w")
    train_file.writelines("\n".join(train_no_filenames + train_yes_filenames))
    
    test_file_filename = os.path.join(PIPEDATA_DIR, "modelnet40_test.txt")
    test_file = open(test_file_filename, "w")
    test_file.writelines("\n".join(test_no_filenames + test_yes_filenames))