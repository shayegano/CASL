import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from NetworkVPCore import NetworkVPCore
from models import CustomLayers
from Config import Config

class Net_3C_A_K_XL_F(NetworkVPCore):
    def __init__(self, device, num_actions):
        super(self.__class__, self).__init__(device, num_actions)

    def _create_graph(self):
        self._assert_net_type(is_rnn_model = True, is_attention_model = True)

        # Use shared parent class to construct graph inputs
        self._create_graph_inputs()

        # -------- Put custom architecture here --------
        # Video CNN
        fc1_i = CustomLayers.multilayer_cnn(
                input         = self.x,
                n_conv_layers = 3,
                layer_tracker = self.layer_tracker,
                filters       = 32,
                kernel_size   = [3,3],
                strides       = [2,2],
                use_bias      = True,
                padding       = "SAME",
                activation    = tf.nn.relu,
                base_name     = 'conv_v_'
            )

        # Audio CNN
        if Config.USE_AUDIO:
            fc1_a = CustomLayers.multilayer_cnn(
                    input         = self.input_audio,
                    n_conv_layers = 3,
                    layer_tracker = self.layer_tracker,
                    filters       = 32,
                    kernel_size   = [3,3],
                    strides       = [2,2],
                    use_bias      = True,
                    padding       = "SAME",
                    activation    = tf.nn.relu, 
                    base_name = 'conv_a_'
                )
            input = [fc1_i, fc1_a]
        else:
            input = fc1_i

        # LSTM
        rnn_out_dict, self.n_lstm_layers_total = CustomLayers.multilayer_lstm(input=input, 
                                                                              n_lstm_layers_total=self.n_lstm_layers_total, 
                                                                              global_rnn_state_in=self.rnn_state_in, 
                                                                              global_rnn_state_out=self.rnn_state_out, 
                                                                              base_name='', 
                                                                              seq_lengths=self.seq_lengths)

        # Save attention softmax probs for regularization
        if Config.USE_ATTENTION:
            if Config.ATTN_TYPE == Config.attn_multimodal:
                self.attn_softmaxes = rnn_out_dict['attn_softmaxes']

        # Output to NetworkVP
        self.final_flat = rnn_out_dict['lstm_outputs'] # Final layer must always be be called final_flat
        # -------- End custom architecture here --------
        
        # Use shared parent class to construct graph outputs/objectives
        self._create_graph_outputs()
