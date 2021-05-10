import os
import dgl
import glob
import torch
import random
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from os import path
from tqdm import tqdm
from functools import reduce
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
from GraphDatasetInfo import GraphDatasetInfo

# TODO: switch to building graphs entirely in DGL, instead of numpy->networkx->dgl
class ToyDGLDataset(DGLDataset):
    """ Template for customizing graph datasets in DGL.

    Parameters
    ----------
    name : str
        Name of the dataset
    graphdatasetInfo: GraphDatasetInfo
        Stores the information to create the 
        synthetic toy dataset from pseudorandom numbers
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """
    def __init__(self, 
                 name: str, 
                 save_dir: str,
                 info: GraphDatasetInfo=None,
                 shuffleDataset: bool=True, 
                 url=None, 
                 raw_dir=None, 
                 force_reload=False, 
                 verbose=True):
        self.info = info
        self.shuffleDataset = shuffleDataset
        super(ToyDGLDataset, self).__init__(name=name,
                                         url=url, 
                                         raw_dir=raw_dir, 
                                         save_dir=save_dir,
                                         force_reload=force_reload,
                                         verbose=verbose)

    def process(self):
        # process raw data to graphs, labels, splitting masks
        self.graphs = []
        self.labels = []

        # generate graphs from GraphDatasetInfo
        it = 1
        for graphInfo in self.info.SubDatasetInfoList:
            nodeCounts = graphInfo.NodesPerGraph.ToNumpy(graphInfo.GraphCount)

            for i in tqdm(range(graphInfo.GraphCount), 
            desc=f'({it}/{len(self.info.SubDatasetInfoList)}) Generating graphs from SubDataset {graphInfo.Name}'):
                nodeCount = int(nodeCounts[i])
                nodeFeatures = {}

                # generate nodefeature numpy arrays from distribution information
                nodeFeatures['label'] = np.arange(nodeCount)
                for key in graphInfo.NodeFeatures.keys():
                    nodeFeatures[key] = graphInfo.NodeFeatures[key].ToNumpy(nodeCount)
                nodeAttrKeys = nodeFeatures.keys()

                # convert from dict of list to list dicts (needed to add node tuples (label, featureDict) to graph)
                nodeFeatures = self._dictOfListsToListOfDicts(nodeFeatures)
                nodes = []

                # Add nodes with node-label j and nodeFeatures
                for j in range(nodeCount):
                    nodes.append((j, nodeFeatures[j]))
                g = nx.DiGraph()
                g.add_nodes_from(nodes)

                # Add edges with edge features, this example: fully connected.
                edges = []
                for j in range(nodeCount):
                    for k in range(nodeCount):
                        if not j == k: # no self-loops
                            deltaPhi = abs(g.nodes[j]['Phi'] - g.nodes[k]['Phi'])
                            deltaEta = abs(g.nodes[j]['Eta'] - g.nodes[k]['Eta'])
                            rapiditySquared = deltaEta * deltaEta + deltaPhi * deltaPhi
                            # possibly have a threshold here to reduce number of edges
                            edgeFeatures = {
                                'DeltaPhi': deltaPhi, 
                                'DeltaEta': deltaEta, 
                                'RapiditySquared': rapiditySquared
                            }
                            edges.append((j, k, edgeFeatures))

                g.add_edges_from(edges)
                g = dgl.from_networkx(g, 
                                    node_attrs=nodeAttrKeys, 
                                    edge_attrs=graphInfo.EdgeFeatures.keys())
                self.graphs.append(g)
                self.labels.append(graphInfo.Label)
            it += 1

        # shuffle dataset
        if self.shuffleDataset:
            # hacky, better way to shuffle both lists?
            random.seed(0)
            random.shuffle(self.graphs)
            random.seed(0)
            random.shuffle(self.labels)
            
        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

        # save histograms
        print('Calculating and saving histograms...')
        self.saveHistograms()
        self.printProperties()

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    def _dictOfListsToListOfDicts(self, DL):
        """
        DL = {'a': [0, 1], 'b': [2, 3]}
        to
        LD = [{'a': 0, 'b': 2}, {'a': 1, 'b': 3}]
        """
        return [dict(zip(DL, t)) for t in zip(*DL.values())]

    def save(self):
        # save graphs and labels
        graph_path = os.path.join(self.save_dir, f'{self.name}_graphs.bin')
        save_graphs(graph_path, self.graphs, {'labels': self.labels})
        # save other information in python dict
        info_path = os.path.join(self.save_dir, f'{self.name}_properties.pkl')
        save_info(info_path,{
            'num_graph_classes' : self.num_graph_classes,
            'graphClasses' : self.graphClasses,
            'num_graphs' : self.num_graphs,
            'num_all_nodes' : self.num_all_nodes,
            'num_all_edges' : self.num_all_edges,
            'dim_nfeats' : self.dim_nfeats,
            'dim_efeats' : self.dim_efeats,
            'dim_allfeats' : self.dim_allfeats,
            'nodeFeatKeys' : self.nodeFeatKeys,
            'edgeFeatKeys' : self.edgeFeatKeys,
            'nodeAndEdgeFeatKeys' : self.nodeAndEdgeFeatKeys
            })

    def load(self):
        # load processed data from directory `self.save_dir`
        graph_path = os.path.join(self.save_dir, f'{self.name}_graphs.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']

        datasetinfo_path = os.path.join(self.save_dir, f'{self.name}.json')
        self.info = GraphDatasetInfo.LoadFromJsonfile(datasetinfo_path)
        # propertyinfo_path = os.path.join(self.save_dir, f'{self.name}_properties.pkl')
        # loadedInfos = load_info(propertyinfo_path)

    def has_cache(self):
        # check whether there are processed data in `self.save_dir`
        graph_path = os.path.join(self.save_dir, f'{self.name}_graphs.bin')
        propertyinfo_path = os.path.join(self.save_dir, f'{self.name}_properties.pkl')
        datasetinfo_path = os.path.join(self.save_dir, f'{self.name}.json')
        return os.path.exists(graph_path) and os.path.exists(propertyinfo_path) and os.path.exists(datasetinfo_path)
    
    @property
    def num_graph_classes(self):
        return len(self.info.SubDatasetInfoList)

    @property
    def graphClasses(self):
        return self.info.GraphClasses
    
    @property
    def num_graphs(self):
        return self.info.TotalGraphCount
    
    @property
    def num_all_nodes(self):
        return sum([g.number_of_nodes() for g in self.graphs])
    
    @property
    def num_all_edges(self):
        return sum([g.number_of_edges() for g in self.graphs])

    @property
    def nodeFeatKeys(self):
        return list(self.info.SubDatasetInfoList[0].NodeFeatures.keys())

    @property
    def edgeFeatKeys(self):
        return list(self.info.SubDatasetInfoList[0].EdgeFeatures.keys())

    @property
    def nodeAndEdgeFeatKeys(self):
        return list(self.nodeFeatKeys) + list(self.edgeFeatKeys)
    
    @property
    def dim_nfeats(self):
        return len(self.nodeFeatKeys)

    @property
    def dim_efeats(self):
        return len(self.edgeFeatKeys)

    @property
    def dim_allfeats(self):
        return self.dim_nfeats + self.dim_efeats
    
    def get_split_indices(self):
        train_split_idx = int(self.info.TotalGraphCount * self.info.SplitPercentages['train'])
        valid_split_idx = train_split_idx + int(self.info.TotalGraphCount * self.info.SplitPercentages['valid'])
        return {
            'train': torch.arange(train_split_idx),
            'valid': torch.arange(train_split_idx, valid_split_idx),
            'test': torch.arange(valid_split_idx, self.info.TotalGraphCount)
        }
    
    def download(self):
        # download raw data to local disk
        pass
    
    def printProperties(self):
        print(f'Num Graph classes: {self.num_graph_classes}')
        print(f'Graph classes: {self.graphClasses}')
        print(f'Number of graphs: {self.num_graphs}')
        print(f'Number of all nodes in all graphs: {self.num_all_nodes}')
        print(f'Number of all edges in all graphs: {self.num_all_edges}')
        print(f'Dim node features: {self.dim_nfeats}')
        print(f'Node feature keys: {self.nodeFeatKeys}')
        print(f'Dim edge features: {self.dim_efeats}')
        print(f'Edge feature keys: {self.edgeFeatKeys}')

    def _getFeatureByKey(self, g, key):
        if(key in g.ndata.keys()):
            return g.ndata[key]
        elif(key in g.edata.keys()):
            return g.edata[key]
        else:
            raise KeyError(f'Key {key} not found in node or edge data.')

    def _accumulateFeature(self, key, graphLabel):
        """
        Goes through all graphs, concats the specified feature (by key) 
        if the graphLabel matches and returns the tensor
        """
        if self.num_graphs <= 0:
            raise Exception('There are no graphs in the dataset.')
        feat = self._getFeatureByKey(self.graphs[0], key)
        for i in range(1, self.num_graphs):
            if self.labels[i] == graphLabel:
                data = self._getFeatureByKey(self.graphs[i], key)
                feat = torch.cat((feat, data))
        return feat

    def saveHistograms(self, outputPath='', nBins=20):
        plt.figure(figsize=(10,7))
        matplotlib.rcParams.update({'font.size': 16})
        #iterate through all edge/node features
        for key in self.nodeAndEdgeFeatKeys:
            # iterate through all graphClasses
            for gclass in self.graphClasses:
                data = self._accumulateFeature(key, gclass).detach().cpu().numpy()
                p = np.percentile(data, [1, 99])
                bins = np.linspace(p[0], p[1], nBins)
                plt.hist(data, bins, alpha=0.7, label=f'GraphClass {gclass}', histtype="step")

            plt.title(key)
            plt.ylabel("frequency")
            plt.legend(loc='upper right')
            from os import path
            filename = f"Histo_{self.name}_{key}.jpg"
            if outputPath == '':
                outputPath = self.save_dir
            outputFilePath = path.join(outputPath, filename)
            plt.savefig(outputFilePath)
            plt.clf()


def GetNodeFeatureVectors(graph):
    #print(graph.ndata.values())
    feat = []
    for key, val in graph.ndata.items():
        if key != 'label':
            feat.append(val)
    #print(feat)
    return _getFeatureVec(feat)

def GetEdgeFeatureVectors(graph):
    return _getFeatureVec(graph.edata.values())

def _getFeatureVec(data):
    feat = tuple([x.data for x in data])
    feat = torch.dstack(feat).squeeze()
    feat = feat.float()
    return feat

def GetNeighborNodes(graph, sourceNodeLabel: int):
    """
    returns tensor of [srcNodeID, dstNodeId]
    e.g.:
    neighborhood = GetNeighborNodes(graph, sourceNodeLabel=7)
    print(neighborhood)
    tensor([[ 7,  0],
        [ 7,  1],
        [ 7,  2],
        [ 7,  3],
        [ 7,  4],
        [ 7,  5],
        [ 7,  6],
        [ 7,  8],
        [ 7,  9],
        [ 7, 10],
        [ 7, 11],
        [ 7, 12],
        [ 7, 13],
        [ 7, 14]])
    """
    # if sourceNodeLabel > graph.num_nodes() - 1:
    #     raise Exception(f'Specified source node label exceeds the number of available nodes in the graph.')
    # edgeListWholeGraph = GetEdgeList(graph)
    # u, v = edgeListWholeGraph
    # indices = (u == sourceNodeLabel).nonzero(as_tuple=True)[0]
    # neighbors = torch.dstack(edgeListWholeGraph).squeeze()[indices]
    return graph.out_edges(sourceNodeLabel)

def GetEdgeList(graph):
    """
    returns a tuple(tensor(srcNodeID), tensor(dstNodeId)) of the whole graph
    """
    return graph.edges(form='uv', order='srcdst')