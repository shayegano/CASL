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
import sys, time, cv2, os
if sys.version_info >= (3,0): from queue import Queue
else: from Queue import Queue
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Config import Config

class Environment:
    def __init__(self):
        self._set_env()
        self.nb_frames    = Config.STACKED_FRAMES
        self.frame_q      = Queue(maxsize=self.nb_frames)
        self.audio_q      = Queue(maxsize=self.nb_frames)
        self.total_reward = 0

        if Config.USE_AUDIO:
            self.previous_state = self.current_state = [None, None]
        else:
            self.previous_state = self.current_state = None

        self.reset()

    def _set_env(self):
        if Config.GAME_CHOICE == Config.game_doorpuzzle:
            from Doorpuzzle import Doorpuzzle
            self.game = Doorpuzzle()
        elif Config.GAME_CHOICE == Config.game_minecraft:
            from Minecraft import Minecraft
            self.game = Minecraft()
        else: 
            raise ValueError("[ ERROR ] Invalid choice of game. Check Config.py for choices")

    def _get_current_state(self):
        if Config.USE_AUDIO:
            if not self.frame_q.full() or not self.audio_q.full():
                return [None, None]

            audio_ = np.array(self.audio_q.queue)
            audio_ = np.transpose(audio_, [1, 2, 0])

        if not self.frame_q.full():
            return None

        image_ = np.array(self.frame_q.queue)
        image_ = np.transpose(image_, [1, 2, 0]) # e.g., changes image from (1,84,84) to (84,84,1) 

        if Config.USE_AUDIO:
            return [image_, audio_]
        else:
            return image_

    def _update_frame_q(self, frame):
        if self.frame_q.full():
            self.frame_q.get()# Pop oldest frame
        self.frame_q.put(frame)

    def _update_audio_q(self, audio):
        if self.audio_q.full():
            self.audio_q.get()# Pop oldest frame
        self.audio_q.put(audio)

    def reset(self):
        self.total_reward = 0
        self.frame_q.queue.clear()

        if Config.USE_AUDIO:
            self.audio_q.queue.clear() 
            
            image, audio = self.game.reset()
            self._update_frame_q(image)
            self._update_audio_q(audio)

            self.previous_state = self.current_state = [None, None]
        else:
            self._update_frame_q(self.game.reset())
            self.previous_state = self.current_state = None

        if self.frame_q.full():
            self.current_state = self._get_current_state()

    def step(self, action, pid=None, count=None):
        observation, reward, done = self.game.step(action, pid, count)
        self.total_reward += reward

        if Config.USE_AUDIO:
            image = observation[0]
            audio = observation[1]

            self._update_frame_q(image)
            self._update_audio_q(audio)
        else:
            image = observation
            self._update_frame_q(image)

        self.previous_state = self.current_state
        self.current_state = self._get_current_state()

        return reward, done

    def visualize_env(self, attention_i, attention_a):
        image, audio = self.game._get_obs(show_gt = True)
        
        # Display
        fig = plt.figure(0)
        ax = plt.subplot2grid((2,2), (0,0))
        ax.set_anchor('W')
        plt.title("Image")
        plt.imshow(image)
        ax = plt.subplot2grid((2,2), (0,1))
        ax.set_anchor('E')
        plt.title("Audio")
        plt.imshow(audio, cmap='gray')

        if attention_i is not None:
            with sns.axes_style("whitegrid"):
                plt.subplot2grid((2,2), (1,0), colspan=2)
                plt.title("Attention")
                plt.plot(np.asarray(attention_a), np.arange(len(attention_a)), label='Attention-audio')
                plt.legend(loc='upper center', ncol=2, fancybox=True, shadow=True)
                plt.xlabel("Probability")
                plt.ylabel("Step #")
                ax = plt.gca()
                ax.set_xlim(-0.2, 1.2)

        plt.pause(Config.TIMER_DURATION)
