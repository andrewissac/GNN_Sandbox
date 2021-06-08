import glob
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib
from os import path
from ToyDGLDataset import ToyDGLDataset
from tqdm import tqdm


def getAllDatasetNames(datasetRootDir):
    files = glob.glob(datasetRootDir + '/*/*/*.json', recursive=True)
    files.sort()
    datasetDirectories = [path.dirname(file) for file in files]
    datasetnames = [path.normpath(dir).split(path.sep)[-1] for dir in datasetDirectories]
    return datasetDirectories, datasetnames


matplotlib.rcParams.update({'font.size': 20})
datasetRootDir = '/home/andrew/GNN_Sandbox/GraphToyDatasets'
datasetDirs, datasetNames = getAllDatasetNames(datasetRootDir)

idx = 0
for idx in tqdm(range(len(datasetDirs))):
    dataset = ToyDGLDataset(datasetNames[idx], datasetDirs[idx])
    data = { 'P_t' : [], 'Eta' : []}

    # iterate through all graphClasses
    for gclass in dataset.graphClasses:
        pt = dataset._accumulateFeature('P_t', gclass)
        eta = dataset._accumulateFeature('Eta', gclass)
        data['P_t'].append((gclass, pt))
        data['Eta'].append((gclass, eta))

    fig, ax = plt.subplots(figsize=(10,7))
    ax.scatter(data['Eta'][0][1], data['P_t'][0][1], label=f'GraphClass0', alpha=0.3)
    ax.scatter(data['Eta'][1][1], data['P_t'][1][1], label=f'GraphClass1', alpha=0.3)
    ax.set_xlabel('Eta')
    ax.set_ylabel('P_t')
    ax.legend()
    outputFilePath = path.join(datasetDirs[idx], datasetNames[idx] + '_Pt_Eta_ScatterPlot.jpg')
    fig.savefig(outputFilePath)
    plt.clf()
    plt.cla()
    plt.close()

    # fig, ax = plt.subplots(figsize=(10,7))
    # res = sn.kdeplot(x=data['Eta'][0][1], y=data['P_t'][0][1], levels = 6, label=f'GraphClass0')
    # res = sn.kdeplot(x=data['Eta'][1][1], y=data['P_t'][1][1], levels = 6, label=f'GraphClass1')
    # outputFilePath = path.join(datasetDirs[idx], datasetNames[idx] + 'Pt_Eta_KDEPlot.jpg')
    # setPlotLabelsAndLegend(ax)
    # fig.savefig(outputFilePath)

