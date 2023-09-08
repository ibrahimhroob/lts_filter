import torch
import torch.nn as nn
from lts.models.pn2_utils_v2 import PointNetSetAbstraction,PointNetFeaturePropagation

class get_model(nn.Module):
    def __init__(self, act_fun = nn.ReLU):
        super(get_model, self).__init__()

        self.sa1 = PointNetSetAbstraction(2048, 0.2,  8,   3 + 3, [32, 64], False, act_fun)    #self, npoint, radius, nsample, in_channel, mlp, group_all
        self.sa2 = PointNetSetAbstraction(2048, 0.4, 16,  64 + 3, [64, 128], False, act_fun)
        self.sa3 = PointNetSetAbstraction(1024, 0.8, 32, 128 + 3, [128, 256], False, act_fun)
        self.sa4 = PointNetSetAbstraction(1024, 1.0, 32, 256 + 3, [256, 512], False, act_fun)

        self.fp4 = PointNetFeaturePropagation(768, [256, 256], act_fun)
        self.fp3 = PointNetFeaturePropagation(384, [256, 256], act_fun)
        self.fp2 = PointNetFeaturePropagation(320, [256, 128], act_fun)
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128], act_fun)

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)            
        x = self.sigmoid(x) 

        x = x.permute(0, 2, 1)

        return x
    

if __name__ == '__main__':
    import  torch
    model = get_model(1)
    xyz = torch.rand(6, 9, 2048)
    print(model(xyz))