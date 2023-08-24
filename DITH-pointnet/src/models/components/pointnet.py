import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from src.models.components.pointnet_utils import PointNetEncoder


class PointNetSemSeg(nn.Module):
    def __init__(self, hparams: dict):
        super(PointNetSemSeg, self).__init__()
        self.k = hparams["num_class"]
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=6)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        if self.k > 2:
            self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        else:
            self.conv4 = torch.nn.Conv1d(128, 1, 1)
            
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        if self.k > 2:
            x = F.log_softmax(x.view(-1,self.k), dim=-1)
            x = x.view(batchsize, n_pts, self.k)
        else:
            x = torch.sigmoid(x)
            
        return x, trans_feat