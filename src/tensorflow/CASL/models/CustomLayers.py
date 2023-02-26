import time, collections
import tensorflow as tf
import numpy as np
from Config import Config
from tensorflow.contrib import rnn
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.cudnn_rnn import CudnnLSTM

def IntraOptionPolicy(netcore, name):
    feat_size = netcore.final_flat.get_shape().as_list()[1]
    limits = (6./np.sqrt(feat_size + netcore.num_actions))/Config.NUM_OPTIONS
    W = tf.get_variable('W_intra_option',
                        shape=[Config.NUM_OPTIONS, feat_size, netcore.num_actions],
                        dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(-limits,limits))
    b = tf.get_variable('b_intra_option',
                        shape=[Config.NUM_OPTIONS, netcore.num_actions],
                        dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(-limits,limits))

    # NOTE tf.gather outputs zeros by design for invalid indices. But, it is not an issue.
    W_option = tf.gather(W, netcore.option_index, axis=0) 
    b_option = tf.gather(b, netcore.option_index, axis=0)

    out = tf.matmul(tf.expand_dims(netcore.final_flat, axis=1), W_option)
    out = tf.squeeze(out, axis=1) + b_option

    return tf.nn.softmax(out, name = name)

def multilayer_cnn(input, n_conv_layers, layer_tracker, filters, kernel_size, strides, use_bias, padding, activation, base_name):
    for i_conv in xrange(0,n_conv_layers):
        layer_tracker.append(tf.layers.conv2d(inputs=input, filters=filters, kernel_size=kernel_size, strides=strides,
                                              use_bias=use_bias, padding=padding, activation=activation, name='%s%s' % (base_name,str(i_conv))))
        input = layer_tracker[-1]

    return tf.contrib.layers.flatten(layer_tracker[-1])

def lstm_layer(input, out_dim, global_rnn_state_in, seq_lengths, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        batch_size = tf.shape(seq_lengths)[0]
        input_reshaped, initial_state_input = RNNInputStateHandler.process_input_state(batch_size, input, global_rnn_state_in)
        cell_lstm = rnn.LSTMCell(num_units = out_dim, state_is_tuple=True)  
        if Config.USE_ATTENTION:
            cell = AttentionCellWrapper(cell_lstm, project_output=False) 
        else:
            cell = cell_lstm
        outputs, state = tf.nn.dynamic_rnn(cell,
                                           input_reshaped,
                                           initial_state=initial_state_input,
                                           sequence_length=seq_lengths,
                                           time_major=False)

        return outputs, state

class RNNInputStateHandler():
    @classmethod
    def process_input_state(cls,batch_size,input,global_rnn_state_in):
        # NOTE Not implemented yet for lstm_layer>1
        state_tuple = cls.get_state_tuple(global_rnn_state_in, is_global_state=True)

        if Config.USE_ATTENTION:
            if Config.ATTN_TYPE == Config.attn_multimodal:
                input_dim = input[0].get_shape()[1].value # [1] is for feature size
                input_i_reshaped = tf.reshape(input[0], [batch_size, -1, input_dim]) # (B,T,I)
                input_a_reshaped = tf.reshape(input[1], [batch_size, -1, input_dim]) # (B,T,I)
                input_reshaped = tf.stack([input_i_reshaped, input_a_reshaped], axis = 2) # (B,T,2,I) so we can keep track of audio/video separately in attentioncellwrapper
            elif Config.ATTN_TYPE == Config.attn_temporal:
                input = cls.concat_a_v(input)
                input_dim = input.get_shape()[1].value # even when only an image. [1] is for feature size
                input_reshaped = tf.reshape(input, [batch_size, -1, input_dim]) # (B,T,I)
        else:
            input = cls.concat_a_v(input)
            input_dim = input.get_shape()[1].value # even when only an image. [1] is for feature size
            input_reshaped = tf.reshape(input, [batch_size, -1, input_dim]) # (B,T,I)

        return input_reshaped, state_tuple 

    @staticmethod
    def get_state_tuple(rnn_state_in, is_global_state):
        if Config.USE_ATTENTION:
            if Config.ATTN_TYPE == Config.attn_multimodal:
                if is_global_state:
                    state_tuple = AttnMultimodalState(rnn.LSTMStateTuple(rnn_state_in['c'][-1], rnn_state_in['h'][-1]), 
                                                      rnn_state_in['attn_state'][-1]) 
                else:
                    state_tuple = AttnMultimodalState(rnn_state_in['lstm_state'], 
                                                      rnn_state_in['attn_state']) 
            elif Config.ATTN_TYPE == Config.attn_temporal:
                if is_global_state:
                    state_tuple = AttnTemporalState(rnn.LSTMStateTuple(rnn_state_in['c'][-1], rnn_state_in['h'][-1]), 
                                                    rnn_state_in['attn_state'][-1], 
                                                    rnn_state_in['attn_state_hist'][-1])  
                else:
                    state_tuple = AttnTemporalState(rnn_state_in['lstm_state'], 
                                                    rnn_state_in['attn_state'], 
                                                    rnn_state_in['attn_state_hist'])
        else:
            if is_global_state:
                state_tuple = rnn.LSTMStateTuple(rnn_state_in['c'][-1], rnn_state_in['h'][-1]) 
            else:
                raise RuntimeError('Should not be here! LSTM without attention is handled by LSTMCell, not AttentionCellWrapper, so this should never be called.')                

        return state_tuple

    @staticmethod
    def concat_a_v(input):
        if Config.USE_AUDIO:
            return tf.concat([input[0], input[1]], axis=1) # axis = 0 would concat batches instead 
        else:
            return input

    @staticmethod
    def get_rnn_dict(init_with_zeros=False, n_lstm_layers_total=None):
        if init_with_zeros:
            assert n_lstm_layers_total is not None
            if Config.USE_ATTENTION:
                if Config.ATTN_TYPE == Config.attn_multimodal:
                    rnn_dict = [{'c': np.zeros(Config.NCELLS, dtype=np.float32),
                                 'h': np.zeros(Config.NCELLS, dtype=np.float32),
                                 'attn_state': np.zeros(Config.NMODES, dtype=np.float32)} ] * n_lstm_layers_total 
                if Config.ATTN_TYPE == Config.attn_temporal:
                    rnn_dict = [{'c': np.zeros(Config.NCELLS, dtype=np.float32),
                                 'h': np.zeros(Config.NCELLS, dtype=np.float32),
                                 'attn_state': np.zeros(Config.ATTN_STATE_NCELLS, dtype=np.float32),
                                 'attn_state_hist': np.zeros(Config.ATTN_TEMPORAL_WINDOW*Config.ATTN_STATE_NCELLS, dtype=np.float32) }] * n_lstm_layers_total 
            else:
                rnn_dict = [{'c': np.zeros(Config.NCELLS, dtype=np.float32),
                             'h': np.zeros(Config.NCELLS, dtype=np.float32)}] * n_lstm_layers_total
        else:
            rnn_dict = {'c':[], 'h':[]}
            if Config.USE_ATTENTION:
                rnn_dict['attn_state'] = []
                if Config.ATTN_TYPE == Config.attn_temporal:
                    rnn_dict['attn_state_hist'] = []
        return rnn_dict

    @staticmethod
    def update_global_state_dict(rnn_dict, rnn_state_tuple):
        if Config.USE_ATTENTION:
            rnn_dict['c'].extend([rnn_state_tuple[0].c]) 
            rnn_dict['h'].extend([rnn_state_tuple[0].h])
            rnn_dict['attn_state'].extend([rnn_state_tuple[1]])
            if Config.ATTN_TYPE == Config.attn_temporal:
                rnn_dict['attn_state_hist'].extend([rnn_state_tuple[2]])
        else: 
            rnn_dict['c'].extend([rnn_state_tuple.c]) # Brackets matter!
            rnn_dict['h'].extend([rnn_state_tuple.h]) # Brackets matter!

    @staticmethod
    def get_output_dict_from_output_tuple(outputs):
        if Config.USE_ATTENTION:
            if Config.ATTN_TYPE == Config.attn_multimodal:
                lstm_outputs, attn_softmaxes = outputs
                attn_softmaxes = tf.reshape(attn_softmaxes, [-1, Config.NMODES]) #(B,TMAX,I)-->(B*TMAX,I)
                lstm_outputs = tf.reshape(lstm_outputs, [-1, Config.NCELLS])
                return {'lstm_outputs' : lstm_outputs, 'attn_softmaxes' : attn_softmaxes}
            elif Config.ATTN_TYPE == Config.attn_temporal:
                lstm_outputs = tf.reshape(outputs, [-1, Config.NCELLS])
                return {'lstm_outputs':lstm_outputs}
        else:
            lstm_outputs = tf.reshape(outputs, [-1, Config.NCELLS])
            return {'lstm_outputs':lstm_outputs}

    @staticmethod
    def reshaped_batched_outputs(batched_outputs, feat_shape):
        return 

    @staticmethod
    def append_rnn_placeholders(rnn_dict, base_name, lstm_count):
        c0 = tf.placeholder(tf.float32, [None, Config.NCELLS], '%s%s%s%s' % ('rnn_', base_name, 'c0_layer', str(lstm_count))) 
        h0 = tf.placeholder(tf.float32, [None, Config.NCELLS], '%s%s%s%s' % ('rnn_', base_name, 'h0_layer', str(lstm_count)))
        rnn_dict['c'].extend([c0]) 
        rnn_dict['h'].extend([h0])

        if Config.USE_ATTENTION:
            if Config.ATTN_TYPE == Config.attn_multimodal:
                attn_state0 = tf.placeholder(tf.float32, [None, Config.NMODES], '%s%s%s%s' % ('rnn_', base_name, 'attn_state0_layer', str(lstm_count)))
                rnn_dict['attn_state'].extend([attn_state0])
            elif Config.ATTN_TYPE == Config.attn_temporal:
                attn_state0 = tf.placeholder(tf.float32, [None, Config.ATTN_STATE_NCELLS], '%s%s%s%s' % ('rnn_', base_name, 'attn_state0_layer', str(lstm_count)))
                rnn_dict['attn_state'].extend([attn_state0])
                attn_state_hist0 = tf.placeholder(tf.float32, [None, Config.ATTN_TEMPORAL_WINDOW*Config.ATTN_STATE_NCELLS], '%s%s%s%s' % ('rnn_', base_name, 'attn_state_hist0_layer', str(lstm_count)))
                rnn_dict['attn_state_hist'].extend([attn_state_hist0])

class AttnMultimodalState(collections.namedtuple("AttnMultimodalState", ("lstm_state", "attn_state"))):
    def clone(self, **kwargs):
        return super(AttnMultimodalState, self)._replace(**kwargs)

class AttnTemporalState(collections.namedtuple("AttnTemporalState", ("lstm_state", "attn_state", "attn_state_hist"))):
    def clone(self, **kwargs):
        return super(AttnTemporalState, self)._replace(**kwargs)

class AttentionCellWrapper(rnn_cell_impl.RNNCell):
    # Attention mode enums
    FUSION_SUM, FUSION_CONC = range(2)

    def __init__(self, cell, attn_vec_size=None,project_output=False, state_is_tuple=True, reuse=None):
        """Create a cell with attention.
        Args:
          cell: an RNNCell, an attention is added to it.
          attn_vec_size: integer, the number of convolutional features calculated
              on attention state and a size of the hidden layer built from
              base cell state. Equal Config.ATTN_STATE_NCELLS to by default.
          project_output: bool, whether or not a projection layer should be used at the output
          state_is_tuple: If True, accepted and returned states are n-tuples, where
            `n = len(cells)`.  By default (False), the states are all
            concatenated along the column axis.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
        Raises:
          TypeError: if cell is not an RNNCell.
          ValueError: if cell returns a state tuple but the flag
              `state_is_tuple` is `False` or if attn_length is zero or less.
        """
        super(AttentionCellWrapper, self).__init__(_reuse=reuse)
        self._check_inputs(cell, state_is_tuple)
        self._state_is_tuple = state_is_tuple
        self._cell = cell
        if Config.ATTN_TYPE == Config.attn_temporal:
            if attn_vec_size is None:
                attn_vec_size = Config.ATTN_STATE_NCELLS
            self._attn_vec_size = attn_vec_size
        self._project_output = project_output
        self._reuse = reuse

    def _check_inputs(self, cell, state_is_tuple):
        if not rnn_cell_impl._like_rnncell(cell):  
            raise TypeError("The parameter cell is not RNNCell.")
        if Config.ATTN_TYPE == Config.attn_temporal and Config.ATTN_TEMPORAL_WINDOW <= 0:
            raise ValueError("Config.ATTN_TEMPORAL_WINDOW should be greater than zero, got %s" % str(Config.ATTN_TEMPORAL_WINDOW))
        if not state_is_tuple:
            raise ValueError("Using a concatenated state is slower and will soon be deprecated. Use state_is_tuple=True.")

    @property
    def state_size(self):
        if Config.ATTN_TYPE == Config.attn_multimodal:
            return (self._cell.state_size, Config.NMODES)
        elif Config.ATTN_TYPE == Config.attn_temporal:
            return (self._cell.state_size, Config.ATTN_STATE_NCELLS, Config.ATTN_STATE_NCELLS * Config.ATTN_TEMPORAL_WINDOW)

    @property
    def output_size(self):
        if Config.ATTN_TYPE == Config.attn_multimodal:
            return (Config.NCELLS,Config.NMODES)
        if Config.ATTN_TYPE == Config.attn_temporal:
            return Config.NCELLS

    def call(self, inputs, state):
        if Config.ATTN_TYPE == Config.attn_multimodal:
            # Attend using prev lstm state
            lstm_state, prev_attn_softmax = state
            input_i = inputs[:,0,:]
            input_a = inputs[:,1,:]
            attn_dim = Config.NCELLS
            inputs_attended, cur_attn_softmax = self._attention_multimodal(input_i, input_a, lstm_state.h, attn_dim, self.FUSION_CONC)

            # LSTM
            lstm_output, lstm_state = self._cell(inputs_attended, lstm_state)
            lstm_output = (lstm_output, cur_attn_softmax)

            # Postprocess
            new_state = RNNInputStateHandler.get_state_tuple({'lstm_state' : lstm_state, 'attn_state' : cur_attn_softmax}, is_global_state=False)

        elif Config.ATTN_TYPE == Config.attn_temporal:
            # Attend using prev attn_state
            state, attn_state, attn_state_hist = state
            input_size = inputs.get_shape().as_list()[1]# [0] is batch size, [1] is feature size 
            inputs_attended = rnn_cell_impl._linear(args=[inputs, attn_state], output_size=input_size, bias=True)

            # LSTM
            lstm_output, lstm_state = self._cell(inputs_attended, state)

            # Attention for next timestep
            new_state_cat = tf.concat(nest.flatten(lstm_state), 1) # NOTE this is [c,h] being used for _attention_temporal (not just h)
            attn_state_hist = tf.reshape(attn_state_hist, [-1, Config.ATTN_TEMPORAL_WINDOW, Config.ATTN_STATE_NCELLS])
            new_attn_state, new_attn_state_hist = self._attention_temporal(new_state_cat, attn_state_hist)

            # Projection layer
            if self._project_output: 
                with tf.variable_scope("attn_output_projection"):
                    output = rnn_cell_impl._linear(args=[lstm_output, new_attn_state], output_size=Config.ATTN_STATE_NCELLS, bias=True)
            else:
                output = new_attn_state

            # Postprocess
            new_attn_state_hist = tf.concat( [new_attn_state_hist, tf.expand_dims(output, 1)], 1) # Concats latest output to new_attn_state_hist
            new_attn_state_hist = tf.reshape( new_attn_state_hist, [-1, Config.ATTN_TEMPORAL_WINDOW * Config.ATTN_STATE_NCELLS])
            new_state = RNNInputStateHandler.get_state_tuple({'lstm_state' : lstm_state, 'attn_state' : new_attn_state, 'attn_state_hist' : new_attn_state_hist}, is_global_state=False)

        else:
            raise ValueError('Invalid Config.ATTN_TYPE selected. Check Config.py!')

        return lstm_output, new_state

    def _attention_multimodal(self, input_i, input_a, input_h, attn_dim, fusion_mode):
        linear_i = tf.layers.dense(inputs=input_i, units=attn_dim, activation=None, name='linear_i')
        linear_a = tf.layers.dense(inputs=input_a, units=attn_dim, activation=None, name='linear_a')
        linear_h = tf.layers.dense(inputs=input_h, units=attn_dim, activation=None, name='linear_h')

        tanh_layer = tf.add_n([linear_i, linear_a, linear_h]) 
        tanh_layer = tf.layers.dense(inputs=tanh_layer, units=attn_dim, activation=tf.tanh, name='tanh_layer')
        
        softmax_attention = tf.layers.dense(inputs=tanh_layer, units=Config.NMODES, activation=tf.nn.softmax, name='softmax_attention')
        feat_i_attention = tf.multiply(input_i, tf.reshape(softmax_attention[:,0], [-1,1]), name='feat_i_attention')
        feat_a_attention = tf.multiply(input_a, tf.reshape(softmax_attention[:,1], [-1,1]), name='feat_a_attention')

        if fusion_mode == self.FUSION_SUM:
            fused_layer = tf.add(feat_i_attention, feat_a_attention, name='fused_layer')
        elif fusion_mode == self.FUSION_CONC:
            fused_layer = tf.concat([feat_i_attention, feat_a_attention], axis=1, name='fused_layer')

        return fused_layer, softmax_attention 

    def _attention_temporal(self, query, attn_state_hist):
        with tf.variable_scope("attention"): 
            k = tf.get_variable("attn_w", [1, 1, Config.ATTN_STATE_NCELLS, self._attn_vec_size])
            v = tf.get_variable("attn_v", [self._attn_vec_size])

            hidden = tf.reshape(attn_state_hist, [-1, Config.ATTN_TEMPORAL_WINDOW, 1, Config.ATTN_STATE_NCELLS])
            hidden_features = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            y = rnn_cell_impl._linear(args=query, output_size=self._attn_vec_size, bias=True)
            y = tf.reshape(y, [-1, 1, 1, self._attn_vec_size])
            s = tf.reduce_sum(v * tf.tanh(hidden_features + y), [2, 3]) # [0] is batch, [1] is temporal window, [2] and [3] not sure

            a = tf.nn.softmax(s)
            d = tf.reduce_sum( tf.reshape(a, [-1, Config.ATTN_TEMPORAL_WINDOW, 1, 1])*hidden, [1, 2])
            new_attn_state = tf.reshape(d, [-1, Config.ATTN_STATE_NCELLS])
            new_attn_state_hist = tf.slice(attn_state_hist, [0, 1, 0], [-1, -1, -1]) # Removes oldest element from new_attn_state_hist

            return new_attn_state, new_attn_state_hist

def multilayer_lstm(input, n_lstm_layers_total, global_rnn_state_in, global_rnn_state_out, base_name, seq_lengths):
    for lstm_count in xrange(Config.NUM_LAYERS_PER_LSTM):
        RNNInputStateHandler.append_rnn_placeholders(global_rnn_state_in, base_name, lstm_count)
        rnn_out, rnn_final_tuple = lstm_layer(input, Config.NCELLS, global_rnn_state_in, seq_lengths, '%s%s%s' % ('rnn_', base_name, str(lstm_count)))
        RNNInputStateHandler.update_global_state_dict(global_rnn_state_out, rnn_final_tuple)
        input = rnn_out # NOTE this does not work for lstm_layer>1
    n_lstm_layers_total += Config.NUM_LAYERS_PER_LSTM

    return RNNInputStateHandler.get_output_dict_from_output_tuple(rnn_out), n_lstm_layers_total
