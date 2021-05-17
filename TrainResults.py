import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from PPrintable import PPrintable
from JsonSerializable import JsonSerializable


class TrainResults(JsonSerializable, PPrintable):
    def __init__(self):
        self.epochIdx = 0
        self.epoch = []
        self.loss = []
        self.train_acc = []
        self.test_acc = []
        self.val_acc = []
        self.best_val_acc = 0
        self.best_test_acc = 0
        self.best_val_acc_epoch = -1
        self.best_model = None

    def addResult(self, model, loss, trainAcc, valAcc, testAcc):
        self.epoch.append(self.epochIdx)
        self.loss.append(loss)
        self.train_acc.append(trainAcc)
        self.val_acc.append(valAcc)
        self.test_acc.append(testAcc)

        # Save the best validation accuracy and the corresponding test accuracy.
        if self.best_val_acc < valAcc:
            self.best_val_acc = valAcc
            self.best_val_acc_epoch = self.epochIdx
            self.best_test_acc = testAcc
            self.best_model = deepcopy(model)

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
        plt.figure(figsize=(8, 6))
        plt.plot(self.epoch, self.loss)
        plt.title('loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()

    def plotAcc(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.epoch, self.train_acc, label='train acc')
        plt.plot(self.epoch, self.val_acc, label='val acc')
        plt.plot(self.epoch, self.test_acc, label='test acc')
        plt.title('accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()