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

import sys, time, os, errno, threading, shutil
import numpy as np
from Config import Config
from shutil import copyfile, copy, copytree
if sys.version_info >= (3,0): from queue import Queue as queueQueue
else: from Queue import Queue as queueQueue
from datetime import datetime
from multiprocessing import Process, Queue, Value

class ProcessStats(Process):
    def __init__(self):
        super(ProcessStats, self).__init__()
        self.episode_log_q     = Queue(maxsize=100)
        self.episode_count     = Value('i', 0)
        self.training_count    = Value('i', 0)
        self.should_save_model = Value('i', 0)
        self.trainer_count     = Value('i', 0)
        self.predictor_count   = Value('i', 0)
        self.agent_count       = Value('i', 0)
        self.total_frame_count = 0
        self.reward_log        = Value('d', 0.0) # For tensorboard log
        self.roll_reward_log   = Value('d', 0.0)# For tensorboard log
        self._log_config_file()

    def copy_files_in_dir(self, src, dest):
        try:
            shutil.copytree(src, dest)
        except OSError as e:
            # If the error was caused because the source wasn't a directory
            if e.errno == errno.ENOTDIR:
                shutil.copy(src, dest)
            else:
                print('Directory not copied. Error: %s' % e)

    def _log_config_file(self):
        if Config.TRANSFER_MODE:
            if Config.LOAD_CHECKPOINT == False:
                raise RuntimeError("[ ERROR ] Please set LOAD_CHECKPOINT to True")

            new_logdir = Config.LOGDIR + '_transfer' 
            self.copy_files_in_dir(Config.LOGDIR, new_logdir)
            Config.LOGDIR = new_logdir

        else:
            if not os.path.exists(Config.LOGDIR):
                os.makedirs(Config.LOGDIR)

            # Only backup the Config file if it does not exist (avoids cyclical updates of Config.py if loading checkpoints)
            if not os.path.isfile(os.path.join(Config.LOGDIR,'Config.py')):
                copyfile('Config.py', os.path.join(Config.LOGDIR,'Config.py')) 

    def return_reward_log(self):
        return self.reward_log.value, self.roll_reward_log.value

    def FPS(self):
        # average FPS from the beginning of the training (not current FPS)
        return np.ceil(self.total_frame_count / (time.time() - self.start_time))

    def TPS(self):
        # average TPS from the beginning of the training (not current TPS)
        return np.ceil(self.training_count.value / (time.time() - self.start_time))

    def run(self):
        #  try:
        with open(os.path.join(Config.LOGDIR, Config.RESULTS_FILENAME), 'a') as results_logger:
            # Init parameters
            rolling_frame_count = 0
            rolling_reward      = 0
            results_q           = queueQueue(maxsize=Config.STAT_ROLLING_MEAN_WINDOW)
            self.start_time     = time.time()
            first_time          = datetime.now()

            while True:
                episode_time, reward, length = self.episode_log_q.get()

                self.total_frame_count += length
                self.episode_count.value += 1

                rolling_frame_count += length
                rolling_reward += reward

                # Append episode_time, reward, length to results_q
                if results_q.full():
                    old_episode_time, old_reward, old_length = results_q.get()
                    rolling_frame_count -= old_length
                    rolling_reward -= old_reward
                    first_time = old_episode_time
                results_q.put((episode_time, reward, length))

                if self.episode_count.value % Config.SAVE_FREQUENCY == 0:
                    self.should_save_model.value = 1

                # Print result to table
                if self.episode_count.value % Config.PRINT_STATS_FREQUENCY == 0:
                    print('[Time: %8d] '
                          '[Episode: %8d Score: %10.4f] '
                          '[RScore: %10.4f RPPS: %5d] '
                          '[PPS: %5d TPS: %5d] '
                          '[NT: %2d NP: %2d NA: %2d]'
                        % (int(time.time()-self.start_time),
                           self.episode_count.value,
                           reward,
                           rolling_reward / results_q.qsize(),
                           rolling_frame_count / (datetime.now() - first_time).total_seconds(),
                           self.FPS(),
                           self.TPS(),
                           self.trainer_count.value,
                           self.predictor_count.value,
                           self.agent_count.value))
                    self.reward_log.value = reward
                    self.roll_reward_log.value = rolling_reward/results_q.qsize()

                    sys.stdout.flush()

                # Results_logger (results.txt)
                # Log date, rolling reward, length
                results_logger.write('%s, %10.4f, %d\n' % (episode_time.strftime("%Y-%m-%d %H:%M:%S"), rolling_reward / results_q.qsize(), length))
                results_logger.flush()
