import datetime
import os
import cPickle as pickle
import numpy as np
from optimisers import Optimiser, SOptimiser
from layers import LayerWithParameters
from visualisers import plot_training_evolvement, plot_setting_compare



class Experiment(object):

    def __init__(self, settings, num_epoch,
                 data_monitors=None, stats_interval=1, reset=True, remark="", legends=[]):
        if reset:
            for setting in settings:
                layers = setting.model.layers
                for layer in layers:
                    if isinstance(layer, LayerWithParameters):
                        layer.reset_params()

        self.results_dir = "../results/"
        self.settings = settings
        self.num_epoch = num_epoch
        self.data_monitors = data_monitors
        self.stats_interval = stats_interval
        if data_monitors is None:
            self.data_monitors = {'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}
        self.remark = remark

        self.legends = legends


    def do_experiment(self):


        DIR = self.results_dir+datetime.datetime.now().strftime("%d-%H_%M_%S")
        os.mkdir(DIR)

        if len(self.settings)==1:
            setting = self.settings[0]
            os.mkdir(DIR+"/"+str(1))
            with open(DIR+"/"+str(1)+"/setting.txt", 'w') as f:
                f.write(self.remark+"\n")
                f.write(str(setting))

            optimiser = SOptimiser(setting.model, setting.error, setting.learning_rule,
                                  setting.train_set, setting.valid_set, self.data_monitors, setting.schedulers)
            stats, keys, run_time = optimiser.train(self.num_epoch, self.stats_interval)

            with open(DIR+"/"+str(1)+"/activations.pkl", 'wb') as f:
                pickle.dump(optimiser.activations_record, f, -1)

            with open(DIR+"/"+str(1)+"/stats.pkl", 'wb') as f:
                pickle.dump((stats, keys, run_time), f, -1)
            fig = plot_training_evolvement(stats, keys)
            fig.savefig(DIR+"/evolvement.pdf")
            return

        assert len(self.legends)>0, "not have legends"

        stats_arr = []
        keys1 = None

        for i, setting in enumerate(self.settings):
            os.mkdir(DIR+"/"+str(i+1))
            with open(DIR+"/"+str(i+1)+"/setting.txt", 'w') as f:
                f.write(self.remark+"\n")
                f.write(str(setting))

            optimiser = Optimiser(setting.model, setting.error, setting.learning_rule,
                                  setting.train_set, setting.valid_set, self.data_monitors, setting.schedulers)
            stats, keys, run_time = optimiser.train(self.num_epoch, self.stats_interval)

            keys1 = keys
            stats_arr.append(stats)

            with open(DIR+"/"+str(i+1)+"/stats.pkl", 'wb') as f:
                pickle.dump((stats, keys, run_time), f, -1)

            with open(DIR+"/"+str(i+1)+"/activations.pkl", 'wb') as f:
                pickle.dump(optimiser.activations_record, f, -1)

        fig1, fig2 = plot_setting_compare(stats_arr, keys1, legends=self.legends)
        fig1.savefig(DIR+"/compare-acc.pdf")
        fig2.savefig(DIR+"/compare-err.pdf")

    def load_results(self, time, index):

        DIR = self.results_dir+time+"/"+str(index)+"/"
        with open(DIR+"setting.txt", 'r') as f:
            setting = f.read()
        with open(DIR+"stats.pkl", 'rb') as f:
            stats, keys, run_time = pickle.load(f)
        with open(DIR+"activations.pkl", 'rb') as f:
            activations = pickle.load(f)
        return stats, keys, run_time, setting, activations





