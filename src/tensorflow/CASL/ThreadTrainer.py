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

import time
import numpy as np
from threading import Thread
from Config import Config

class ThreadTrainer(Thread):
    def __init__(self, server, id):
        super(ThreadTrainer, self).__init__()
        self.setDaemon(True)
        self.id        = id
        self.server    = server
        self.exit_flag = False

    @staticmethod
    def _dynamic_pad(image_, audio_, r_, a_, o_):
        t = image_.shape[0]
        if t != Config.TIME_MAX:
            imaget = np.zeros((Config.TIME_MAX, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.STACKED_FRAMES), dtype=np.float32)
            audiot = np.zeros((Config.TIME_MAX, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.STACKED_FRAMES), dtype=np.float32)
            rt = np.zeros((Config.TIME_MAX), dtype=np.float32)
            at = np.zeros((Config.TIME_MAX, a_.shape[1]), dtype=np.float32)
            ot = np.zeros((Config.TIME_MAX), dtype=np.float32)

            imaget[:t] = image_; audiot[:t] = audio_; rt[:t] = r_; at[:t] = a_; ot[:t] = o_# Fill from beginning to t with true image
            image_ = imaget; audio_ = audiot; r_ = rt; a_ = at; o_ = ot;# Zero pad the suffix

        return image_, audio_, r_, a_, o_, t

    def run(self):
        while not self.exit_flag:
            batch_size = 0
            seq_lengths__ = []
            state_image__ = []
            r__           = []
            a__           = []
            o__           = []
            state_audio__ = []
            rnn_state__   = [] if Config.USE_RNN else None

            while batch_size <= Config.TRAINING_MIN_BATCH_SIZE:
                state_image_, state_audio_, r_, a_, o_, rnn_state_ = self.server.training_q.get() # state_audio_ and rnn_state_ are None if not used

                if Config.USE_RNN:
                    state_image_, state_audio_, r_, a_, o_, t = ThreadTrainer._dynamic_pad(state_image_, state_audio_, r_, a_, o_)
                    seq_lengths__.append(t)
                    rnn_state__.append(rnn_state_)

                state_image__.extend(state_image_)
                r__.extend(r_)
                a__.extend(a_)
                o__.extend(o_)

                if Config.USE_AUDIO: 
                    assert state_audio_ is not None
                    state_audio__.extend(state_audio_)

                batch_size += state_image_.shape[0]

            if Config.TRAIN_MODELS:
                self.server.train_model(state_image__, state_audio__, r__, a__, o__, rnn_state__, seq_lengths__)
