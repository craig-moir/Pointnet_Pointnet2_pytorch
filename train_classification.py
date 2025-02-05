"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import torch.onnx
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import PipeNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[2, 10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_class=40):
    mean_correct = []
    mean_direction_loss = []
    mean_normal_loss = []
    mean_radius_loss = []
    
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target_label, target_direction, target_normal, target_radius) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target_label, target_direction, target_normal, target_radius = points.cuda(), target_label.cuda(), target_direction.cuda(), target_normal.cuda(), target_radius.cuda()

        points = points.transpose(2, 1)
        pred, pred_direction, pred_normal, pred_radius, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target_label.cpu()):
            classacc = pred_choice[target_label == cat].eq(target_label[target_label == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target_label == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target_label.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
        
        # normalising vector output
        pred_normal = torch.nn.functional.normalize(pred_normal, dim=1)
        pred_direction = torch.nn.functional.normalize(pred_direction, dim=1)
        
        direction_loss = (torch.minimum(torch.norm(pred_direction - target_direction, dim=1), torch.norm(pred_direction + target_direction, dim=1))*target_label).sum()
        
        normal_loss = (torch.norm(pred_normal - target_normal, dim=1)*target_label).sum()
        
        target_radius = target_radius / 1000 # convert pred radius from mm to m
        target_radius[target_radius == 0] = 0.0001 # avoid divide by zero
        A5 = 20
        big_radius_loss = (torch.abs(target_radius - torch.flatten(pred_radius))*target_label).sum()
        small_radius_loss = (torch.abs(target_radius - torch.flatten(pred_radius))/(target_radius*A5)*target_label).sum()
        radius_loss = big_radius_loss + small_radius_loss
    
        if target_label.sum().item() != 0:
            direction_loss /= target_label.sum()
            normal_loss /= target_label.sum()
            radius_loss /= target_label.sum()
        
        mean_direction_loss.append(direction_loss.item())
        mean_normal_loss.append(normal_loss.item())
        mean_radius_loss.append(radius_loss.item())

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    direction_loss = np.mean(mean_direction_loss)
    normal_loss = np.mean(mean_normal_loss)
    radius_loss = np.mean(mean_radius_loss)

    return instance_acc, class_acc, direction_loss, normal_loss, radius_loss


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    # data_path = 'data/modelnet40_normal_resampled/'
    data_path = 'data/pipes/'

    train_dataset = PipeNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = PipeNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    
    # print(classifier)
    # torch.onnx.export(classifier, torch.randn(1, 3, 1024), "model.onnx", verbose=True)
    # #traced_script_module = torch.jit.trace(classifier, torch.randn(24, 3, 1024))
    # #traced_script_module.save("traced_resnet_model.pt")
    # exit()
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        mean_loss = []
        mean_direction_loss = []
        mean_normal_loss = []
        mean_radius_loss = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (points, target_label, target_direction, target_normal, target_radius) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            # print(points.size(), target)
            # continue
            
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            
            points = points.transpose(2, 1)
            # print(points.size())

            if not args.use_cpu:
                points, target_label, target_direction, target_normal, target_radius = points.cuda(), target_label.cuda(), target_direction.cuda(), target_normal.cuda(), target_radius.cuda()

            pred_label, pred_direction, pred_normal, pred_radius, trans_feat = classifier(points)
            
            if 0:
                # visualising the net
                from torchviz import make_dot
                y = classifier(points)
                make_dot(y, params=dict(list(classifier.named_parameters()))).render("pipenet", format="png")
                exit()
            
            # print(pred)
            # print("pred size =", pred_label.size())
            # print(target_label.long())
            # print("target size =", target_label.size())
            # continue
            # exit()
            classification_loss, direction_loss, normal_loss, radius_loss = criterion(pred_label, pred_direction, pred_normal, pred_radius, target_label.long(), target_direction, target_normal, target_radius, trans_feat)
            loss = 2.5*classification_loss + 2*direction_loss + 2*normal_loss + 0.5*radius_loss
            pred_choice = pred_label.data.max(1)[1]

            # print(loss)
            # print("loss size:", loss.size())
            #continue
            #exit()

            correct = pred_choice.eq(target_label.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            mean_loss.append(loss.item())
            mean_direction_loss.append(direction_loss.item())
            mean_normal_loss.append(normal_loss.item())
            mean_radius_loss.append(radius_loss.item())
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        log_string('Train Loss: %f' % np.mean(mean_loss))
        log_string('Train Direction Loss: %f' % np.mean(mean_direction_loss))
        log_string('Train Normal Loss: %f' % np.mean(mean_normal_loss))
        log_string('Train Radius Loss: %f' % np.mean(mean_radius_loss))

        with torch.no_grad():
            instance_acc, class_acc, direction_loss, normal_loss, radius_loss = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string(f"Test Direction Loss: {direction_loss}")
            log_string(f"Test Normal Loss: {normal_loss}")
            log_string(f"Test Radius Loss: {radius_loss}")
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                
            if epoch % 10 == 0 or epoch == args.epoch - 1:
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + f'/model_epoch_{epoch}.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
