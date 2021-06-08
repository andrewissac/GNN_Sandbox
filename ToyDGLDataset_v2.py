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

class ToyDGLDataset_v2(DGLDataset):
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
        super(ToyDGLDataset_v2, self).__init__(name=name,
                                         url=url, 
                                         raw_dir=raw_dir, 
                                         save_dir=save_dir,
                                         force_reload=force_reload,
                                         verbose=verbose)

    def process(self):
        # process raw data to graphs, labels, splitting masks
        self.graphs = []
        self.labels = []
        nFeatMapping = self.info.nodeFeatMapping
        eFeatMapping = self.info.edgeFeatMapping
        gFeatMapping = self.info.graphFeatMapping
        phi = nFeatMapping['Phi']
        eta = nFeatMapping['Eta']
        pt = nFeatMapping['P_t']

        # needed to generate bins for histograms
        minNFeatValues = [9999999999 for x in range(self.dim_nfeats)]
        maxNFeatValues = [-9999999999 for x in range(self.dim_nfeats)]
        minEFeatValues = [9999999999 for x in range(self.dim_efeats)]
        maxEFeatValues = [-9999999999 for x in range(self.dim_efeats)]
        minNodeCount = 9999999999
        maxNodeCount = -9999999999

        def calcAbsDiff(vec):
            """
            To calculate deltaPhi, deltaEta, rapiditysquared efficiently the following is done:
            Take input array with Phi values of all nodes, create matrix by repeating the column vector
            Get second matrix as the transposed first matrix
            Substract both and take the absolute.
            """
            dim = len(vec)
            matB = np.expand_dims(vec, axis=1)
            matB = np.repeat(matB, dim, axis=1)
            matA = np.transpose(matB)
            return np.absolute(matA - matB)

        # generate graphs from GraphDatasetInfo
        it = 1
        for graphInfo in self.info.subDatasetInfoList:
            nodeCounts = graphInfo.nodesPerGraph.ToNumpy(graphInfo.graphCount)

            for i in tqdm(range(graphInfo.graphCount), 
            desc=f'({it}/{len(self.info.subDatasetInfoList)}) Generating graphs from SubDataset {graphInfo.name}'):

                nodeCount = int(nodeCounts[i])
                # get min/max nodeCount for histogram binning
                if nodeCount < minNodeCount:
                    minNodeCount = nodeCount
                if nodeCount > maxNodeCount:
                    maxNodeCount = nodeCount

                nodeFeatures = []
                # generate nodefeature numpy arrays from distribution information
                for feat in graphInfo.nodeFeat:
                    # nodeFeature[nFeatMapping['Phi']] -> contains a list with ALL Phis from this subdataset
                    nodeFeatures.append(feat.ToNumpy(nodeCount))

                # get min/max node feature values for histogram binning
                for i in range(self.dim_nfeats):
                    tempMin = round(np.amin(nodeFeatures[i]))
                    if tempMin < minNFeatValues[i]:
                        minNFeatValues[i] = tempMin
                    tempMax = round(np.amax(nodeFeatures[i]))
                    if tempMax > maxNFeatValues[i]:
                        maxNFeatValues[i] = tempMax
 
                # calculate edge features efficiently through matrices
                deltaPhi = calcAbsDiff(nodeFeatures[phi])
                deltaEta = calcAbsDiff(nodeFeatures[eta])
                rapiditySquared = deltaPhi * deltaPhi + deltaEta * deltaEta

                src_ids = []
                dst_ids = []
                edgeFeatures = []
                for j in range(nodeCount):
                    for k in range(nodeCount):
                        if not j == k: # no self-loops
                            # add src/dst node ids
                            src_ids.append(j)
                            dst_ids.append(k)

                            # add edge features (care, order should be the same as in eFeatMapping!)
                            edgeFeatures.append([deltaEta[j][k], deltaPhi[j][k], rapiditySquared[j][k]])

                # get min/max edge feature values for histogram binning
                if(len(edgeFeatures) > 0):
                    tempMin = np.amin(edgeFeatures, axis=0)
                    tempMin = [round(x) for x in tempMin]
                    tempMax = np.amax(edgeFeatures, axis=0)
                    tempMax = [round(x) for x in tempMax]
                    for i in range(self.dim_efeats):
                        if tempMin[i] < minEFeatValues[i]:
                            minEFeatValues[i] = tempMin[i]
                        if tempMax[i] > maxEFeatValues[i]:
                            maxEFeatValues[i] = tempMax[i]


                # build graph based on src/dst node ids
                g = dgl.graph((src_ids, dst_ids))
                # dstack -> each entry is now a node feature vec containing [P_t, Eta, Phi, Mass, Type] for that node
                if(len(nodeFeatures) == 0):
                    print(nodeFeatures)
                nodeFeatures = np.dstack(nodeFeatures).squeeze()
                g.ndata['feat'] = torch.from_numpy(nodeFeatures)
                g.edata['feat'] = torch.tensor(edgeFeatures)
                # order: 1st NodeCount, 2nd: TotalP_t
                graphFeatures = { 'feat' : torch.from_numpy(np.array([nodeCount, np.sum(nodeFeatures[pt])]))}
                setattr(g, 'gdata', graphFeatures)
                
                self.graphs.append(g)
                self.labels.append(graphInfo.label)
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
        nBins = 20
        bins = {}
        for key in self.nodeFeatKeys:
            bins[key] = np.linspace(minNFeatValues[nFeatMapping[key]], maxNFeatValues[nFeatMapping[key]], nBins)
        for key in self.edgeFeatKeys:
            bins[key] = np.linspace(minEFeatValues[eFeatMapping[key]], maxEFeatValues[eFeatMapping[key]], nBins)
        bins['NodeCount'] = np.arange(minNodeCount, maxNodeCount+1)
        self.saveHistograms(bins)
        self.saveEtaPtScatterPlot()
        self.printProperties()

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

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
        return len(self.info.subDatasetInfoList)

    @property
    def graphClasses(self):
        return self.info.graphClasses
    
    @property
    def num_graphs(self):
        return self.info.totalGraphCount
    
    @property
    def num_all_nodes(self):
        return sum([g.number_of_nodes() for g in self.graphs])
    
    @property
    def num_all_edges(self):
        return sum([g.number_of_edges() for g in self.graphs])

    @property
    def nodeFeatKeys(self):
        return list(self.info.nodeFeatMapping.keys())

    @property
    def edgeFeatKeys(self):
        return list(self.info.edgeFeatMapping.keys())

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
        train_split_idx = int(self.info.totalGraphCount * self.info.splitPercentages['train'])
        valid_split_idx = train_split_idx + int(self.info.totalGraphCount * self.info.splitPercentages['valid'])
        return {
            'train': torch.arange(train_split_idx),
            'valid': torch.arange(train_split_idx, valid_split_idx),
            'test': torch.arange(valid_split_idx, self.info.totalGraphCount)
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
        if(key in self.info.nodeFeatMapping):
            return g.ndata['feat'][:,self.info.nodeFeatMapping[key]].tolist()
        elif(key in self.info.edgeFeatMapping):
            return g.edata['feat'][:,self.info.edgeFeatMapping[key]].tolist()
        else:
            raise KeyError(f'Key {key} not found in node or edge data.')

    def _accumulateFeature(self, key, graphLabel):
        """
        Goes through all graphs, concats the specified feature (by key) 
        if the graphLabel matches and returns the tensor
        would be better to use masks instead of iterating 
        """
        if self.num_graphs <= 0:
            raise Exception('There are no graphs in the dataset.')
        accumulatedFeat = []
        for i in range(self.num_graphs):
            if self.labels[i] == graphLabel:
                feat = self._getFeatureByKey(self.graphs[i], key)
                accumulatedFeat.extend(feat)
        return accumulatedFeat

    def saveHistograms(self, bins, outputPath=''):
        plt.figure(figsize=(10,7))
        matplotlib.rcParams.update({'font.size': 16})
        lineWidth = 2
        # iterate through all edge/node features
        for key in self.nodeAndEdgeFeatKeys:
            # iterate through all graphClasses
            for gclass in self.graphClasses:
                data = self._accumulateFeature(key, gclass)
                plt.hist(data, bins[key], label=f'GraphClass {gclass}', histtype="step", linewidth=lineWidth)

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

        # get node count histogram
        for gclass in self.graphClasses:
            data = []
            # dumb und bruteforce, but who cares..
            for i in range(0, self.num_graphs):
                if self.labels[i] == gclass:
                    data.append(self.graphs[i].num_nodes())
            plt.hist(data, bins['NodeCount'], label=f'GraphClass {gclass}', histtype="step", linewidth=lineWidth)

        plt.title('Node count')
        plt.ylabel("frequency")
        plt.legend(loc='upper right')
        from os import path
        filename = f"Histo_{self.name}_NodeCount.jpg"
        if outputPath == '':
            outputPath = self.save_dir
        outputFilePath = path.join(outputPath, filename)
        plt.savefig(outputFilePath)
        plt.clf()

    def saveEtaPtScatterPlot(self, outputPath=''):
        data = { 'P_t' : [], 'Eta' : []}

        # iterate through all graphClasses
        for gclass in self.graphClasses:
            pt = self._accumulateFeature('P_t', gclass)
            eta = self._accumulateFeature('Eta', gclass)
            data['P_t'].append((gclass, pt))
            data['Eta'].append((gclass, eta))

        fig, ax = plt.subplots(figsize=(10,7))
        ax.scatter(data['Eta'][0][1], data['P_t'][0][1], label=f'GraphClass0', alpha=0.3)
        ax.scatter(data['Eta'][1][1], data['P_t'][1][1], label=f'GraphClass1', alpha=0.3)
        ax.set_xlabel('Eta')
        ax.set_ylabel('P_t')
        ax.legend()
        if outputPath == '':
            outputPath = self.save_dir
        outputFilePath = path.join(outputPath, self.name + '_Pt_Eta_ScatterPlot.jpg')
        fig.savefig(outputFilePath)
        plt.clf()
        plt.cla()
        plt.close()

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