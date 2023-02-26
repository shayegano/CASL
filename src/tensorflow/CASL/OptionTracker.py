import time
import numpy as np
import matplotlib.pyplot as plt
from Config import Config

class OptionTracker():
    """
        Class used for visualizing options
        Useful for analyzing learned options
    """
    def __init__(self):
        self.img_traj_tracker        = np.zeros((Config.ENV_COL, Config.ENV_ROW, Config.NUM_OPTIONS))
        self.img_option_term_tracker = np.zeros((Config.ENV_COL, Config.ENV_ROW, Config.NUM_OPTIONS))

        self._init_axes()

    def _init_axes(self):
        self.fig = plt.figure("Option Tracker")
        self.fig.show()

        self.axes = {}
        for i_option in range(Config.NUM_OPTIONS):
            key = str(i_option) + '_0'
            ax = plt.subplot2grid((2,Config.NUM_OPTIONS), (0,i_option))
            ax.set_title("Option " + str(i_option) + " traj")
            self.axes[key] = ax

            key = str(i_option) + '_1'
            ax = plt.subplot2grid((2,Config.NUM_OPTIONS), (1,i_option))
            ax.set_title("Option " + str(i_option) + " term")
            self.axes[key] = ax

        self.artists = {}
        for i_option in range(Config.NUM_OPTIONS):
            key = str(i_option) + '_0'
            img_artist = self.axes[key].imshow(np.zeros((5,5)), cmap = 'copper')
            self.artists[key] = img_artist

            key = str(i_option) + '_1'
            img_artist = self.axes[key].imshow(np.zeros((5,5)), cmap = 'copper')
            self.artists[key] = img_artist

    def _update_tracker(self, agt_loc, i_option, option_term):
        self.img_traj_tracker[agt_loc[0], agt_loc[1], i_option] += 1
        self.img_option_term_tracker[agt_loc[0], agt_loc[1], i_option] += option_term 

    def _plot_tracker(self):
        for i_option in range(Config.NUM_OPTIONS):
            key = str(i_option) + '_0'
            self.artists[key].set_data(self.img_traj_tracker[:, :, i_option])
            self.artists[key].autoscale()

            key = str(i_option) + '_1'
            self.artists[key].set_data(self.img_option_term_tracker[:, :, i_option])
            self.artists[key].autoscale()

        plt.pause(Config.TIMER_DURATION)
