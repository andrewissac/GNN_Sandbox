from dgl.nn import GraphConv
import dgl
import torch.nn as nn
import dgl.function as fn
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


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature

            # simply copies the node features of all neighbors and puts them into a message
            gcn_msg = fn.copy_u(u='h', out='m') 
            # sums all messages up
            gcn_reduce = fn.sum(msg='m', out='h')

            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)