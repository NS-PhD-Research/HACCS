from AlexNet import AlexNet
from DenseNet import DenseNet
from GoogleNet import GoogleNet
from LeNet import LeNet
from ResNet import ResNet
from VGG import VGG
from WideResNet import WideResNet

class ModelFactory:
    factories = {}
    
    def addFactory(id, mftory):
        ModelFactory.factories.put[id] = mftory
    addFactory = staticmethod(addFactory)
    
    def getModel(id):
        if id not in ModelFactory.factories:
            ModelFactory.factories[id] = eval(id + '.Factory()')

        return ModelFactory.factories[id].get()
    getDataset = staticmethod(getModel)