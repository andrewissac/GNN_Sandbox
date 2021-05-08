from Enumbase import FloatEnumBase, EnumVal
from JsonSerializable import JsonSerializable
from PPrintable import PPrintable
from scipy.stats import truncnorm
from tqdm import tqdm
import numpy as np
import networkx as nx

class DistributionType(FloatEnumBase):
    Uniform = EnumVal(0.0, 'Uniform')
    Normal = EnumVal(1.0, 'Normal')
    TruncNorm = EnumVal(2.0, 'TruncNorm')
    SingleValue = EnumVal(3.0, "SingleValue") # no distribution

class Split(FloatEnumBase):
    Train = EnumVal(0.0, 'Train')
    Validation = EnumVal(1.0, 'Validation')
    Test = EnumVal(2.0, 'Test')

class Distribution(JsonSerializable, PPrintable):
    def __init__(
        self, mininum: float, maximum: float, distributionType: DistributionType, 
        mean: float=None, standardDeviation: float=None, roundToNearestInt: bool=False):
        self.Min = mininum
        self.Max = maximum
        self.DistributionType = distributionType
        self.RoundToNearestInt = roundToNearestInt
        if self.DistributionType == DistributionType.Normal or self.DistributionType == DistributionType.TruncNorm:
            if mean == None or standardDeviation == None:
                raise Exception('mean or standardDeviation are None, must be set to a float value!')
            self.Mean = mean
            self.StandardDeviation = standardDeviation
        if self.DistributionType == DistributionType.SingleValue:
            if self.Min != self.Max:
                raise Exception(f'Min and Max value must be identical for DistributionType {DistributionType.SingleValue}!')

    def ToNumpy(self, size: int):
        dist = None
        if(self.DistributionType == DistributionType.Normal):
            dist = np.random.normal(self.Mean, self.StandardDeviation, size)
        elif(self.DistributionType == DistributionType.Uniform):
            dist = np.random.uniform(self.Min, self.Max, size)
        elif(self.DistributionType == DistributionType.TruncNorm):
            dist = truncnorm(
                (self.Min - self.Mean) / self.StandardDeviation, 
                (self.Max - self.Mean) / self.StandardDeviation, 
                loc=self.Mean, 
                scale=self.StandardDeviation).rvs(size)
        elif(self.DistributionType == DistributionType.SingleValue):
            dist = np.empty(size)
            dist.fill(self.Min)
        return dist if self.RoundToNearestInt == False else np.rint(dist)

    def __str__(self):
        return f'min: {self.Min}, max: {self.Max}, type: {self.DistributionType.displayname}, mean: {self.Mean}, std: {self.StandardDeviation}'


class GraphSubdatasetInfo(JsonSerializable, PPrintable):
    def __init__(self, name: str, description: str, label: int,
    graphCount: int, nodesPerGraph: Distribution, nodeFeatures: dict, edgeFeatures: dict):
        self.Name = name
        self.Description = description
        self.Label = label
        self.GraphCount = graphCount
        self.NodesPerGraph = nodesPerGraph
        self.NodeFeatures = nodeFeatures
        self.EdgeFeatures = edgeFeatures
    
    
class GraphDatasetInfo(JsonSerializable, PPrintable):
    def __init__(self, name: str, description: str, splitPercentages: dict, graphSubDatasetInfos: list):
        self.Name = name
        self.Description = description
        splitSum = splitPercentages['train'] + splitPercentages['valid'] + splitPercentages['test']
        if abs(splitSum - 1)  > 0.00001:
            raise Exception(f'Split percentages must add up to 1.0!, currently it adds up to {splitSum}')
        self.SplitPercentages = splitPercentages
        self.TotalGraphCount = 0
        self.SubDatasetInfoList = graphSubDatasetInfos
        self.GraphClasses = []

        gclasses = set()
        for subdatasetinfo in self.SubDatasetInfoList:
            self.TotalGraphCount += subdatasetinfo.GraphCount
            gclasses.add(subdatasetinfo.Label)
        self.GraphClasses = list(gclasses)

    def ToNetworkxGraphList(self):
        nxgraphs = []
        rng = np.random.default_rng(seed=0)

        for graphInfo in self.SubDatasetInfoList:
            nodeCounts = graphInfo.NodesPerGraph.ToNumpy(graphInfo.GraphCount)

            for i in tqdm(range(graphInfo.GraphCount), desc='Graphs created'):
                nodeCount = int(nodeCounts[i])
                nodeFeatures = {}
                
                # generate nodefeature numpy arrays from distribution information
                for key in graphInfo.NodeFeatures.keys():
                    nodeFeatures[key] = graphInfo.NodeFeatures[key].ToNumpy(nodeCount)
                    
                # convert from dict of list to list dicts (needed to add node tuples (label, featureDict) to graph)
                nodeFeatures = self._dictOfListsToListOfDicts(nodeFeatures)
                nodes = []
                
                # Add nodes with label j and nodeFeatures
                for j in range(nodeCount):
                    nodes.append((j, nodeFeatures[j]))
                g = nx.DiGraph()
                g.graph['Label'] = graphInfo.Label
                g.add_nodes_from(nodes)
                
                # Add edges with edge features, this example: fully connected.
                edgeTupleList = []
                for j in range(nodeCount):
                    for k in range(nodeCount):
                        if not j == k: # no self-loops
                            deltaPhi = abs(g.nodes[j]['Phi'] - g.nodes[k]['Phi'])
                            deltaEta = abs(g.nodes[j]['Eta'] - g.nodes[k]['Eta'])
                            rapiditySquared = deltaEta * deltaEta + deltaPhi * deltaPhi
                            edgeFeatures = {
                                'DeltaPhi': deltaPhi, 
                                'DeltaEta': deltaEta, 
                                'RapiditySquared': rapiditySquared
                            }
                            edgeTupleList.append((j, k, edgeFeatures))

                g.add_edges_from(edgeTupleList)
                
                nxgraphs.append(g)
        return nxgraphs

    def _dictOfListsToListOfDicts(self, DL):
        return [dict(zip(DL, t)) for t in zip(*DL.values())]


