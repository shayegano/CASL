import sys, time, os
from multiprocessing import Process
from Config import Config

class ProcessTensorboard(Process):
    def run(self):
        print Config.LOGDIR
        os.system('tensorboard --logdir=' + Config.LOGDIR)
