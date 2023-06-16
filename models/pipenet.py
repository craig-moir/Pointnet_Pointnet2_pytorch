import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

import torch

class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)
        
        self.fc_r = nn.Linear(256, 1)
        self.fc_n = nn.Linear(256, 3)
        self.fc_d = nn.Linear(256, 3)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        cl = l3_points.view(B, 1024)
        cl = self.drop1(F.relu(self.bn1(self.fc1(cl))))
        cl = self.drop2(F.relu(self.bn2(self.fc2(cl))))
        cl = self.fc3(cl)
        cl = F.log_softmax(cl, -1)

        r = l3_points.view(B, 1024)
        r = self.drop1(F.relu(self.bn1(self.fc1(r))))
        r = self.drop2(F.relu(self.bn2(self.fc2(r))))
        r = self.fc_r(r)
        # print("r:", r)
        # print("r size:", r.size())
        
        d = l3_points.view(B, 1024)
        d = self.drop1(F.relu(self.bn1(self.fc1(d))))
        d = self.drop2(F.relu(self.bn2(self.fc2(d))))
        d = self.fc_d(d)
        # print("d:", d)
        # print("d size:", d.size())
        
        n = l3_points.view(B, 1024)
        n = self.drop1(F.relu(self.bn1(self.fc1(n))))
        n = self.drop2(F.relu(self.bn2(self.fc2(n))))
        n = self.fc_n(n)
        # print("n:", n)
        # print("n size:", n.size())
        # print("n norm:", torch.norm(n, dim=1))
        # exit()

        return cl, d, n, r, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred_label, pred_direction, pred_normal, pred_radius, target_label, target_direction, target_normal, target_radius, trans_feat):
        classification_loss = F.nll_loss(pred_label, target_label)
        # print(target_label)
        
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
        
        # print("classification loss", classification_loss)
        # print("direction loss", direction_loss)
        # print("normal loss", normal_loss)
        # print("radius loss", radius_loss, "(", big_radius_loss, "+", small_radius_loss, ")")
        
        # print(target_radius, torch.abs(target_radius - torch.flatten(pred_radius))*target_label, torch.abs(target_radius - torch.flatten(pred_radius)))
        
        return classification_loss, direction_loss, normal_loss, radius_loss


