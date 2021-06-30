import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from os import path
from ToyDGLDataset_v2 import ToyDGLDataset_v2
from tqdm import tqdm
from statistics import mean


def getAllDatasetNames(datasetRootDir):
    files = glob.glob(datasetRootDir + '/*/*/*.json', recursive=True)
    files.sort()
    datasetDirectories = [path.dirname(file) for file in files]
    datasetnames = [path.normpath(dir).split(path.sep)[-1] for dir in datasetDirectories]
    return datasetDirectories, datasetnames


matplotlib.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,7))
matplotlib.rcParams.update({'font.size': 16})
lineWidth = 2

datasetRootDir = '/home/andrew/GNN_Sandbox/GraphToyDatasets_v3'
datasetDirs, datasetNames = getAllDatasetNames(datasetRootDir)

for idx in tqdm(range(len(datasetDirs))):
    dataset = ToyDGLDataset_v2(datasetNames[idx], datasetDirs[idx])
    featKey = 'P_t'

    accumulatedFeat_Pt = {}
    sumPt = {}
    meanPt = {}

    sumMin = 999999999999
    sumMax = -999999999999
    meanMin = 999999999999
    meanMax = -999999999999

    for gclass in dataset.graphClasses:
        accumulatedFeat_Pt[gclass] = []
        sumPt[gclass] = []
        meanPt[gclass] = []
        for i in range(dataset.num_graphs):
            if dataset.labels[i] == gclass:
                feat = dataset._getFeatureByKey(dataset.graphs[i], featKey)
                accumulatedFeat_Pt[gclass].append(feat)

        sumPt[gclass] = [sum(x) for x in accumulatedFeat_Pt[gclass]]
        meanPt[gclass] = [mean(x) for x in accumulatedFeat_Pt[gclass]]

        tempSumMin = min(sumPt[gclass])
        tempSumMax = max(sumPt[gclass])
        tempMeanMin = min(meanPt[gclass])
        tempMeanMax = max(meanPt[gclass])

        if sumMin > tempSumMin:
            sumMin = tempSumMin
        if sumMax < tempSumMax:
            sumMax = tempSumMax
        if meanMin > tempMeanMin:
            meanMin = tempMeanMin
        if meanMax < tempMeanMax:
            meanMax = tempMeanMax

    nBins = 20
    binsSumPt = np.linspace(sumMin, sumMax, nBins)
    binsMeanPt = np.linspace(meanMin, meanMax, nBins)

    for gclass in dataset.graphClasses:
        plt.hist(sumPt[gclass], binsSumPt,label=f'GraphClass {gclass}', histtype="step", linewidth=lineWidth)

    plt.title("sum P_t")
    plt.xlabel("sum P_t")
    plt.ylabel("frequency")
    plt.legend(loc='upper right')
    from os import path
    filename = f"Histo_sum_P_t.jpg"
    outputPath = dataset.save_dir
    outputFilePath = path.join(outputPath, filename)
    plt.savefig(outputFilePath)
    plt.clf()

    for gclass in dataset.graphClasses:
        plt.hist(meanPt[gclass], binsMeanPt,label=f'GraphClass {gclass}', histtype="step", linewidth=lineWidth)

    plt.title("mean P_t")
    plt.xlabel("mean P_t")
    plt.ylabel("frequency")
    plt.legend(loc='upper right')
    from os import path
    filename = f"Histo_mean_P_t.jpg"
    outputPath = dataset.save_dir
    outputFilePath = path.join(outputPath, filename)
    plt.savefig(outputFilePath)
    plt.clf()



