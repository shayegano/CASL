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
from Config import Config
from threading import Thread

class ThreadPredictor(Thread):
    def __init__(self, server, id):
        super(ThreadPredictor, self).__init__()
        self.setDaemon(True)
        self.id        = id
        self.server    = server
        self.exit_flag = False

    def run(self):
        ids          = np.zeros(Config.PREDICTION_BATCH_SIZE, dtype=np.uint16)
        states_image = np.zeros((Config.PREDICTION_BATCH_SIZE, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.STACKED_FRAMES), dtype=np.float32)
        states_audio = np.zeros((Config.PREDICTION_BATCH_SIZE, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.STACKED_FRAMES), dtype=np.float32)
        i_option     = [None] * Config.PREDICTION_BATCH_SIZE 
        rnn_state    = [None] * Config.PREDICTION_BATCH_SIZE 

        while not self.exit_flag:
            batch_size = 0
            q_empty = False # If prediction_q is empty, .get() will wait until prediction_q has at least one element
            while batch_size < Config.PREDICTION_BATCH_SIZE and not q_empty:
                ids[batch_size], states_image[batch_size], states_audio[batch_size], rnn_state[batch_size], i_option[batch_size]=self.server.prediction_q.get() 
                batch_size += 1
                q_empty = self.server.prediction_q.empty()

            # Prediction for multiple agents
            predict_dict_batched = self.server.model.predict_p_and_v(states_image[:batch_size], states_audio[:batch_size], rnn_state[:batch_size], i_option[:batch_size])

            # Put p and v into wait_q 
            for i_batch in range(batch_size):
                if Config.USE_OPTIONS:
                    predict_dict_agt = {'cur_intra_option_probs': predict_dict_batched['cur_intra_option_probs'][i_batch], 'option_v_model': predict_dict_batched['option_v_model'][i_batch]}
                    predict_dict_agt['option_term_probs'] = predict_dict_batched['option_term_probs'][i_batch]
                    predict_dict_agt['option_q_model'] = predict_dict_batched['option_q_model'][i_batch]
                else:
                    predict_dict_agt = {'p_actions': predict_dict_batched['p_actions'][i_batch], 'v': predict_dict_batched['v'][i_batch]}

                if Config.USE_RNN:
                    predict_dict_agt['rnn_state_out'] =  predict_dict_batched['rnn_state_out'][i_batch]
                    if Config.USE_ATTENTION and Config.ATTN_TYPE == Config.attn_multimodal:
                        predict_dict_agt['attn'] =  predict_dict_batched['attn'][i_batch]

                self.server.agents[ids[i_batch]].wait_q.put(predict_dict_agt)
