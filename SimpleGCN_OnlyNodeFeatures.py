from dgl.nn import GraphConv
import dgl
import torch.nn as nn
import torch.nn.functional as F


class GCN_OnlyNodeFeatures(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN_OnlyNodeFeatures, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')