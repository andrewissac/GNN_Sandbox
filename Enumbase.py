from enum import Enum, IntEnum, EnumMeta
from numpy import zeros

# Enables using the first value of an enum class to be the default value by imitating a default constructor: MyEnum()
class DefaultEnumMeta(EnumMeta):
    default = object()
    def __call__(cls, value=default, *args, **kwargs):
        if value is DefaultEnumMeta.default:
            # Assume the first enum is default
            return next(iter(cls))
        return super().__call__(value, *args, **kwargs)

class EnumVal(object):
    def __init__(self, value, displayname):
        self.value = value
        self.displayname = displayname

class EnumBase(metaclass=DefaultEnumMeta):
    def __len__(self):
        return len(self.getAllMembers())

    @classmethod
    def getAllMembers(cls):
        return cls.__members__.items()
    
    @classmethod
    def getAllValues(cls):
        return [m.value for m in cls]

    @classmethod
    def getAllNames(cls):
        return [m.name for m in cls]

    @classmethod
    def oneHotVectorToEnum(cls, oneHotVector):
        """
        Usage example for class Category(EnumBase):
        import numpy as np
        oneHotVector = np.array([0., 1.])
        print(Category.oneHotVectorToEnum(oneHotVector))
        """
        allEnumMember = cls.getAllMembers()
        if(len(oneHotVector) != len(allEnumMember)):
            raise Exception("Dimension of the one-hot-encoded vector does not match with the specified enum.")

        from numpy import identity, array_equal
        oneHotMatrix = identity(len(allEnumMember))
        matchedEnumValue = None
        for i in range(oneHotMatrix.shape[0]):
            if(array_equal(oneHotMatrix[i], oneHotVector)):
                matchedEnumValue = float(i)
                break
        if matchedEnumValue is not None:
            return cls(matchedEnumValue)
        else:
            return None

    def toOneHotVector(self):
        """
        Usage example for class Category(EnumBase):
        myCategory = Category.FakeTau
        myOneHotVector= myCategory.toOneHotVector()
        """
        oneHotVector = zeros(len(self))
        oneHotVector[int(self)] = 1.0
        return oneHotVector

class FloatEnumBase(EnumBase, Enum):
    def __int__(self):
        return int(self.value)

    def __new__(cls, value):
        obj = object.__new__(cls)
        obj._value_ = value.value
        obj._displayname_ = value.displayname
        return obj

    @property
    def displayname(self):
        return self._displayname_

    @classmethod
    def getAllDisplaynames(cls):
        return [m.displayname for m in cls]

class IntEnumBase(EnumBase, IntEnum):
    def __new__(cls, value):
        obj = int.__new__(cls, value)
        obj._value_ = value
        return obj