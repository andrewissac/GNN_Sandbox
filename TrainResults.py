import pickle
import matplotlib
import matplotlib.pyplot as plt
from os import path
from PPrintable import PPrintable
from sklearn.metrics import roc_curve, auc
from JsonSerializable import JsonSerializable

matplotlib.rcParams.update({'font.size': 16})

class TrainResults(PPrintable):
    def __init__(self):
        self.epochIdx = 0
        self.epoch = []
        self.loss = []

        self.train_result = []
        self.train_acc = []

        self.test_result = []
        self.test_acc = []

        self.val_result = []
        self.val_acc = []

        self.best_val_acc = 0
        self.best_test_acc = 0
        self.best_val_acc_epoch = -1
        self.best_result = {}

    def addResult(self, loss, train_result, val_result, test_result):
        self.epoch.append(self.epochIdx)
        self.loss.append(loss)

        self.train_result.append(train_result)
        trainAcc = train_result['acc']
        self.train_acc.append(trainAcc)

        self.val_result.append(val_result)
        valAcc = val_result['acc']
        self.val_acc.append(valAcc)

        self.test_result.append(test_result)
        testAcc = test_result['acc']
        self.test_acc.append(testAcc)

        # Save the best validation accuracy and the corresponding test accuracy.
        if self.best_val_acc < valAcc:
            self.best_val_acc = valAcc
            self.best_val_acc_epoch = self.epochIdx
            self.best_test_acc = testAcc
            self.best_result = {
                'train': self.train_result[self.epochIdx], 
                'val': self.val_result[self.epochIdx], 
                'test': self.test_result[self.epochIdx]
                }

        self.epochIdx += 1

    def printLastResult(self):
        self._printResult(-1)
    
    def printBestResult(self):
        self._printResult(self.best_val_acc_epoch)

    def _printResult(self, idx):
        print(f'Epoch: {self.epoch[idx]:02d}, '
            f'Loss: {self.loss[idx]:.4f}, '
            f'Train: {100 * self.train_acc[idx]:.2f}%, '
            f'Valid: {100 * self.val_acc[idx]:.2f}% '
            f'Test: {100 * self.test_acc[idx]:.2f}%')

    def plotLoss(self):
        plt.figure(figsize=(10,7))
        plt.plot(self.epoch, self.loss)
        plt.title('loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()

    def plotAcc(self):
        plt.figure(figsize=(10,7))
        plt.plot(self.epoch, self.train_acc, label='train acc')
        plt.plot(self.epoch, self.val_acc, label='val acc')
        plt.plot(self.epoch, self.test_acc, label='test acc')
        plt.title('accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()

    def plotROC(self):
        y_true = self.best_result['test']['y_true']
        y_pred = self.best_result['test']['y_pred']

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr) # AUC = Area Under Curve, ROC = Receiver operating characteristic
        
        plt.figure(figsize=(10,7))
        plt.plot(fpr, tpr, label='ROC (area = %0.2f)'%(roc_auc))
        plt.plot([0, 1], [0, 1], '--', color='red', label='Luck')
        plt.xlabel('False Positive Rate') 
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (ROC) curve')
        plt.legend()
        plt.grid()
        plt.show()

    def pickle(self, outputPath, filename):
        with open(path.join(outputPath, filename), 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def loadPickle(filePath):
        with open(filePath, 'rb') as f:
            return pickle.load(f)