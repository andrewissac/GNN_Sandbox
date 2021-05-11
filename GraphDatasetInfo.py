from Enumbase import FloatEnumBase, EnumVal
from JsonSerializable import JsonSerializable
from PPrintable import PPrintable
from scipy.stats import truncnorm
from tqdm import tqdm
import numpy as np
import networkx as nx

class DistributionType(FloatEnumBase):
    uniform = EnumVal(0.0, 'uniform')
    normal = EnumVal(1.0, 'normal')
    truncnorm = EnumVal(2.0, 'truncnorm')
    singlevalue = EnumVal(3.0, "singlevalue") # no distribution

class Distribution(JsonSerializable, PPrintable):
    def __init__(
        self, minimum: float, maximum: float, distributionType: DistributionType, 
        mean: float=None, standardDeviation: float=None, roundToNearestInt: bool=False):
        self.min = minimum
        self.max = maximum
        self.distributionType = distributionType
        self.RoundToNearestInt = roundToNearestInt
        if self.distributionType == DistributionType.normal or self.distributionType == DistributionType.truncnorm:
            if mean == None or standardDeviation == None:
                raise Exception('mean or standardDeviation are None, must be set to a float value!')
            self.mean = mean
            self.standardDeviation = standardDeviation
        if self.distributionType == DistributionType.singlevalue:
            if self.min != self.max:
                raise Exception(f'min and max value must be identical for DistributionType {DistributionType.singlevalue}!')

    def ToNumpy(self, size: int):
        dist = None
        if(self.distributionType == DistributionType.normal):
            dist = np.random.normal(self.mean, self.standardDeviation, size)
        elif(self.distributionType == DistributionType.uniform):
            dist = np.random.uniform(self.min, self.max, size)
        elif(self.distributionType == DistributionType.truncnorm):
            dist = truncnorm(
                (self.min - self.mean) / self.standardDeviation, 
                (self.max - self.mean) / self.standardDeviation, 
                loc=self.mean, 
                scale=self.standardDeviation).rvs(size)
        elif(self.distributionType == DistributionType.singlevalue):
            dist = np.empty(size)
            dist.fill(self.min)
        return dist if self.RoundToNearestInt == False else np.rint(dist)

    def __str__(self):
        return f'min: {self.min}, max: {self.max}, type: {self.distributionType.displayname}, mean: {self.mean}, std: {self.standardDeviation}'


class GraphSubdatasetInfo(JsonSerializable, PPrintable):
    def __init__(self, name: str, description: str, label: int,
    graphCount: int, nodesPerGraph: Distribution, 
    nodeFeatMapping: dict, nodeFeat: list, edgeFeatMapping: dict, edgeFeat: list):
        self.name = name
        self.description = description
        self.label = label
        self.graphCount = graphCount
        self.nodesPerGraph = nodesPerGraph
        self.nodeFeatMapping = nodeFeatMapping
        self.nodeFeat = nodeFeat
        self.edgeFeatMapping = edgeFeatMapping
        self.edgeFeat = edgeFeat
    
    
class GraphDatasetInfo(JsonSerializable, PPrintable):
    def __init__(self, name: str, description: str, splitPercentages: dict, graphSubDatasetInfos: list):
        self.name = name
        self.description = description
        splitSum = splitPercentages['train'] + splitPercentages['valid'] + splitPercentages['test']
        if abs(splitSum - 1)  > 0.00001:
            raise Exception(f'Split percentages must add up to 1.0!, currently it adds up to {splitSum}')
        self.splitPercentages = splitPercentages
        self.totalGraphCount = 0
        self.subDatasetInfoList = graphSubDatasetInfos
        self.graphClasses = []
        # assuming the node/edge feature mapping of all subdatasets are equal!
        self.nodeFeatMapping = self.subDatasetInfoList[0].nodeFeatMapping
        self.edgeFeatMapping = self.subDatasetInfoList[0].edgeFeatMapping

        gclasses = set()
        for subdatasetinfo in self.subDatasetInfoList:
            self.totalGraphCount += subdatasetinfo.graphCount
            gclasses.add(subdatasetinfo.label)
        self.graphClasses = list(gclasses)


