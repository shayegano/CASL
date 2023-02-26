# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import time, models, importlib
import numpy as np
from multiprocessing import Queue
from Config import Config
from ProcessAgent import ProcessAgent
from ProcessStats import ProcessStats
from ProcessTensorboard import ProcessTensorboard 
from NoThreadDynamicAdjustment import ThreadDynamicAdjustment
from ThreadPredictor import ThreadPredictor
from ThreadTrainer import ThreadTrainer

class Server:
    def __init__(self):
        self.stats        = ProcessStats()
        self.tensorboard  = ProcessTensorboard()
        self.training_q   = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.prediction_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        if Config.GAME_CHOICE == Config.game_doorpuzzle:
            from Doorpuzzle import Actions
        elif Config.GAME_CHOICE == Config.game_minecraft:
            from Minecraft import Actions
        gridworld_actions = Actions()
        self.num_actions  = gridworld_actions.num_actions

        self.model = self.make_model()
        if Config.LOAD_CHECKPOINT: 
            self.stats.episode_count.value = self.model.load()

        self.training_step      = 0
        self.frame_counter      = 0
        self.agents             = []
        self.predictors         = []
        self.trainers           = []
        self.dynamic_adjustment = ThreadDynamicAdjustment(self)# NOTE Server is passed in here.

    def make_model(self):
        net_model = getattr(importlib.import_module('models.' + Config.NET_ARCH), Config.NET_ARCH)
        return net_model(Config.DEVICE, self.num_actions) 

    def add_trainer(self):
        self.trainers.append(ThreadTrainer(self, len(self.trainers)))
        self.trainers[-1].start()

    def remove_trainer(self):
        self.trainers[-1].exit_flag = True
        self.trainers[-1].join()
        self.trainers.pop()

    def add_predictor(self):
        self.predictors.append(ThreadPredictor(self, len(self.predictors)))
        self.predictors[-1].start()

    def remove_predictor(self):
        self.predictors[-1].exit_flag = True
        self.predictors[-1].join()
        self.predictors.pop()

    def add_agent(self):
        if len(self.agents) == 0:
            self.agents.append(ProcessAgent(self.model, len(self.agents), self.prediction_q, self.training_q, self.stats.episode_log_q, self.num_actions, self.stats))
        else:
            self.agents.append(ProcessAgent(self.model, len(self.agents), self.prediction_q, self.training_q, self.stats.episode_log_q, self.num_actions, None))
        self.agents[-1].start()

    def remove_agent(self):
        self.agents[-1].exit_flag.value = True
        self.agents[-1].join()
        self.agents.pop()

    def train_model(self, x_, audio_, r_, a_, o_, rnn_state_, seq_lengths_):
        self.model.train(x_, audio_, r_, a_, o_, rnn_state_, seq_lengths_)
        self.training_step += 1
        self.frame_counter += np.shape(x_)[0]
        self.stats.training_count.value += 1

        # Tensorboard logging
        if Config.TENSORBOARD and self.stats.training_count.value % Config.TENSORBOARD_UPDATE_FREQUENCY == 0:
            reward, roll_reward = self.stats.return_reward_log()
            self.model.log(self.stats.episode_count.value, x_, audio_, r_, a_, o_, rnn_state_, seq_lengths_, reward, roll_reward)

    def save_model(self):
        self.model.save(self.stats.episode_count.value)

    def main(self):
        # Start Thread objects by calling start() methods
        if Config.TENSORBOARD:
            self.tensorboard.start()
        self.stats.start()
        self.dynamic_adjustment.run()# NOTE self.dynamic_adjustment is NOT thread anymore

        # If Config.PLAY_MODE == True, disable trainers
        if Config.PLAY_MODE:
            for trainer in self.trainers:
                trainer.enabled = False

        # Algorithm parameters
        learning_rate_multiplier = (Config.LEARNING_RATE_END - Config.LEARNING_RATE_START)/Config.ANNEALING_EPISODE_COUNT
        beta_multiplier = (Config.BETA_END - Config.BETA_START)/Config.ANNEALING_EPISODE_COUNT
        if Config.USE_OPTIONS:
            option_epsilon_multiplier = (Config.OPTION_EPSILON_END- Config.OPTION_EPSILON_START)/Config.ANNEALING_EPISODE_COUNT
            option_cost_delib_multiplier = (Config.COST_DELIB_END- Config.COST_DELIB_START)/Config.ANNEALING_EPISODE_COUNT

        while self.stats.episode_count.value < Config.EPISODES:
            # Linearly anneals the learning rate up to Config.ANNEALING_EPISODE_COUNT, after which it maintains at Config.LEARNING_RATE_END
            step = min(self.stats.episode_count.value, Config.ANNEALING_EPISODE_COUNT - 1)
            self.model.learning_rate = Config.LEARNING_RATE_START + learning_rate_multiplier * step
            self.model.beta = Config.BETA_START + beta_multiplier * step
            if Config.USE_OPTIONS:
                self.model.option_epsilon = Config.OPTION_EPSILON_START + option_epsilon_multiplier * step 
                self.model.option_cost_delib = Config.COST_DELIB_START + option_cost_delib_multiplier * step 

            # Saving is async - even if we start saving at a given episode, we may save the model at a later episode
            if Config.SAVE_MODELS and self.stats.should_save_model.value > 0:
                self.save_model()
                self.stats.should_save_model.value = 0

        # Terminate all with exit_flag == True
        self.dynamic_adjustment.exit_flag = True
        while self.agents:
            self.remove_agent()
        while self.predictors:
            self.remove_predictor()
        while self.trainers:
            self.remove_trainer()
