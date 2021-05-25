
from PPrintable import PPrintable
from Distribution import Distribution
from JsonSerializable import JsonSerializable


class GraphSubdatasetInfo(JsonSerializable, PPrintable):
    def __init__(self, name: str, label: int,
    graphCount: int, nodesPerGraph: Distribution, 
    nodeFeatMapping: dict, nodeFeat: list, edgeFeatMapping: dict, edgeFeat: list):
        self.name = name
        self.label = label
        self.graphCount = graphCount
        self.nodesPerGraph = nodesPerGraph
        self.nodeFeatMapping = nodeFeatMapping
        self.nodeFeat = nodeFeat
        self.edgeFeatMapping = edgeFeatMapping
        self.edgeFeat = edgeFeat
    
    
class GraphDatasetInfo(JsonSerializable, PPrintable):
    def __init__(self, name: str, splitPercentages: dict, graphSubDatasetInfos: list):
        self.name = name
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


