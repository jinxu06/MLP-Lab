import numpy as np
import matplotlib.pyplot as plt
import logging
from mlp.layers import *
from mlp.models import *
from mlp.data_providers import *
from mlp.errors import *
from mlp.initialisers import *
from mlp.learning_rules import *
from mlp.optimisers import *
from mlp.schedulers import *
from mlp.penalties import *
from mlp import DEFAULT_SEED

class Environment(object):

    def __init__(self, data_set, batch_size, error=CrossEntropySoftmaxError(), weights_init=None, biases_init=None, logging_level='INFO', random_seed=DEFAULT_SEED):

        self.random_seed = random_seed
        self.rng = np.random.RandomState(self.random_seed)

        self.logger = logging.getLogger()
        level = "logging."+logging_level
        self.logger.setLevel(eval(level))
        self.logger.handlers = [logging.StreamHandler()]

        self.data_set = data_set
        self.batch_size = batch_size
        self.train_set = eval("{0}DataProvider('train', batch_size=self.batch_size, rng=self.rng)".format(self.data_set))
        self.valid_set = eval("{0}DataProvider('valid', batch_size=self.batch_size, rng=self.rng)".format(self.data_set))

        if weights_init is None:
            weights_init = GlorotUniformInit(rng=self.rng)
        self.weights_init = weights_init
        if biases_init is None:
            biases_init = ConstantInit(0.)
        self.biases_init = biases_init

        self.error = error

    def hint(self):
        print "    Environment Setting Hints    "
        print "---------------------------------"
        print "data_set:{0}".format(self.data_set)
        print "choice:\n"
        print "batch_size:{0}\n".format(self.batch_size)
        print "logging_level:{0}".format(self.logger_level)
        print "choice:DEBUG,INFO,WARNING,ERROR\n"
        print "random_seed:{0}\n".format(self.random_seed)
        print "error(default):CrossEntropySoftmaxError"


