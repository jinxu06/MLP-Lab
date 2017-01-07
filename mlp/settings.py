from mlp.initialisers import *

class Setting(object):

    def __init__(self, model, error, learning_rule, train_set, valid_set, schedulers=[],
                 weights_init=GlorotUniformInit(), kernel_init=UniformInit(-0.01, 0.01),
                 biases_init=ConstantInit(0.)):
        self.model = model
        self.error = error
        self.learning_rule = learning_rule
        self.train_set = train_set
        self.valid_set = valid_set
        self.schedulers = schedulers

        self.weights_init = weights_init
        self.kernel_init = kernel_init
        self.biases_init = biases_init

    def __repr__(self):
        s = ""
        s += "\n-------------Model----------------\n"
        s +=  str(self.model)+'\n'
        s += "\n-------------Error----------------\n"
        s +=  str(self.error)+'\n'
        s +=  "\n----------Learning Rule------------\n"
        s +=  str(self.learning_rule)+'\n'
        s +=  "\n-----------Data Provider------------\n"
        s +=  str(self.train_set)+'\n'
        s +=  "\n------------Schedulers--------------\n"
        s +=  str(self.schedulers)+'\n'
        s +=  "\n-----------Weights Init--------------\n"
        s +=  str(self.weights_init)+'\n'
        s +=  "\n------------Kernel Init--------------\n"
        s +=  str(self.kernel_init)+'\n'
        s +=  "\n------------Biases Init--------------\n"
        s +=  str(self.biases_init)+'\n'
        return s



