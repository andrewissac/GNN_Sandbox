import numpy as np
from PPrintable import PPrintable
from scipy.stats import truncnorm
from JsonSerializable import JsonSerializable


class Distribution(JsonSerializable, PPrintable):
    def __init__(
        self, minimum: float, maximum: float, distributionType: str, 
        mean: float=None, standardDeviation: float=None, roundToNearestInt: bool=False):
        self.min = minimum
        self.max = maximum
        self.distributionType = distributionType
        self.RoundToNearestInt = roundToNearestInt
        if self.distributionType == 'normal' or self.distributionType == 'truncnorm':
            if mean == None or standardDeviation == None:
                raise Exception('mean or standardDeviation are None, must be set to a float value!')
            self.mean = mean
            self.standardDeviation = standardDeviation
        if self.distributionType == 'singlevalue':
            if self.min != self.max:
                raise Exception(f'min and max value must be identical for DistributionType {self.distributionType}!')

    def ToNumpy(self, size: int):
        dist = None
        if(self.distributionType == 'normal'):
            dist = np.random.normal(self.mean, self.standardDeviation, size)
        elif(self.distributionType == 'uniform'):
            dist = np.random.uniform(self.min, self.max, size)
        elif(self.distributionType == 'truncnorm'):
            dist = truncnorm(
                (self.min - self.mean) / self.standardDeviation, 
                (self.max - self.mean) / self.standardDeviation, 
                loc=self.mean, 
                scale=self.standardDeviation).rvs(size)
        elif(self.distributionType == 'singlevalue'):
            dist = np.empty(size)
            dist.fill(self.min)
        return dist if self.RoundToNearestInt == False else np.rint(dist)

    def __str__(self):
        return f'min: {self.min}, max: {self.max}, type: {self.distributionType}, mean: {self.mean}, std: {self.standardDeviation}'