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
import os, time
import numpy as np
import tensorflow as tf
from models import CustomLayers 
from Config import Config

class NetworkVPCore(object):
    def __init__(self, device, num_actions):
        self.device         = device
        self.num_actions    = num_actions
        self.img_width      = Config.IMAGE_WIDTH
        self.img_height     = Config.IMAGE_HEIGHT
        self.img_channels   = Config.STACKED_FRAMES
        self.learning_rate  = Config.LEARNING_RATE_START
        self.beta           = Config.BETA_START
        if Config.USE_OPTIONS:
            self.option_epsilon    = Config.OPTION_EPSILON_START
            self.option_cost_delib = Config.COST_DELIB_START
        self.log_epsilon    = Config.LOG_EPSILON
        self.graph          = tf.Graph()

        with self.graph.as_default() as g:
            with tf.device(self.device):
                self._create_graph()

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())

                if Config.TENSORBOARD: 
                    self._create_tensorboard()
                if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
                    vars = tf.global_variables()
                    self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=5, keep_checkpoint_every_n_hours=1.0)

    def _assert_net_type(self, is_rnn_model, is_attention_model):
        if (not is_rnn_model and Config.USE_RNN) or (is_rnn_model and not Config.USE_RNN): 
            raise ValueError('User specific Config.USE_RNN: ' + str(Config.USE_RNN) + ', but selected Config.NET_ARCH: ' + str(Config.NET_ARCH))

        if (not is_attention_model and Config.USE_ATTENTION): # Second case not needed, since can turn attention on or off as long as model supports it 
            raise ValueError('User specific Config.USE_ATTENTION: ' + str(Config.USE_ATTENTION) + ', but selected Config.NET_ARCH: ' + str(Config.NET_ARCH))

    def _create_graph_inputs(self):
        self.episode       = tf.Variable(0, dtype=tf.int32, name='episode')
        self.x             = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='X')
        self.action_index  = tf.placeholder(tf.float32, [None, self.num_actions]) # NOTE this is a one-hot vector indicating action index. Doesn't match i_option format, should unify eventually.
        self.layer_tracker = []
        self.y_r           = tf.placeholder(tf.float32, [None], name='Yr')
        self.var_beta      = tf.placeholder(tf.float32, name='beta', shape=[])

        if Config.USE_AUDIO:
            self.input_audio = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='input_audio')

        if Config.USE_RNN:
            self.seq_lengths = tf.placeholder(tf.int32, [None], name='seq_lengths')
            self.loss_mask = tf.placeholder(tf.float32, [None], name='loss_mask')

            # All LSTM inputs/outputs. All are stored/restored properly as a unified dict, so can have any crazy LSTM architectures desired.
            self.rnn_state_in  = CustomLayers.RNNInputStateHandler.get_rnn_dict()
            self.rnn_state_out = CustomLayers.RNNInputStateHandler.get_rnn_dict()
            self.n_lstm_layers_total = 0
        else:
            self.loss_mask = 1.

        if Config.USE_OPTIONS:
            self.option_index = tf.placeholder(tf.int32, [None], name='option_index')
            self.var_option_epsilon = tf.placeholder(tf.float32, name='option_epsilon', shape=[])
            self.var_option_cost_delib = tf.placeholder(tf.float32, name='option_cost_delib', shape=[])

    def _option_costs(self):
        # Create termination_model, option_q_model, and cur_intra_option_probs
        self.option_q_model = tf.layers.dense(inputs=self.final_flat, units=Config.NUM_OPTIONS, activation=None, use_bias=True, name='option_q_model') # Shape: (batch, # of option)
        self.option_v_model = (1. - self.var_option_epsilon)*tf.reduce_sum(tf.nn.softmax(self.option_q_model)*self.option_q_model, axis=1) + self.var_option_epsilon*tf.reduce_mean(self.option_q_model, axis=1)
        disc_option_v = tf.stop_gradient(self.option_v_model, name='disc_option_v')
        self.cur_intra_option_probs = CustomLayers.IntraOptionPolicy(netcore=self, name='cur_intra_option_probs') # Shape: (batch, # of action)
        self.termination_model = tf.layers.dense(inputs=self.final_flat, units=Config.NUM_OPTIONS, activation=tf.nn.sigmoid, use_bias=True, name='termination_model')

        # Cost: Critic (v) -- first gradient in paper Alg. 1
        batch_size = tf.shape(self.final_flat)[0]
        option_index_stacked = tf.stack((tf.range(start=0,limit=batch_size, dtype=tf.int32), self.option_index), axis=1) # Appends column of batch indices, needed for gather_nd
        cur_option_q = tf.gather_nd(self.option_q_model, option_index_stacked ) # Critic in option case is cur_option_q
        disc_cur_option_q = tf.stop_gradient(cur_option_q, name='disc_cur_option_q') 
        self.cost_critic = 0.5*tf.reduce_sum(tf.square(self.y_r - cur_option_q)*self.loss_mask, axis=0, name='cost_critic')/tf.reduce_sum(self.loss_mask) 

        # Cost: Actor (intra_option policy and entropy)
        # Actor policy cost -- second gradient in paper Alg. 1 
        cur_option_intra_q = tf.reduce_sum(self.cur_intra_option_probs*self.action_index, axis=1, name='cur_option_intra_q')
        self.cost_intra_option_policy = -1.*tf.reduce_sum(tf.log(cur_option_intra_q + self.log_epsilon)*(self.y_r - disc_cur_option_q)*self.loss_mask, axis=0, name='cost_intra_option_policy')/tf.reduce_sum(self.loss_mask) # Negative since want to maximize reward J function

        # Actor entropy cost
        cost_intra_option_entropy = -1.*self.var_beta*tf.reduce_sum(tf.log(self.cur_intra_option_probs+ self.log_epsilon)*self.cur_intra_option_probs, axis=1)
        self.cost_intra_option_entropy_agg = -1.* tf.reduce_sum(cost_intra_option_entropy*self.loss_mask, axis=0, name='intra_option_entropy')/tf.reduce_sum(self.loss_mask) #Negative since want to maximize entropy

        # Termination cost -- third gradient in paper Alg. 1 
        self.cur_term_probs   = tf.gather_nd(self.termination_model, option_index_stacked)
        self.cost_termination = tf.reduce_sum(self.cur_term_probs*((disc_cur_option_q - disc_option_v + self.var_option_cost_delib + Config.COST_MARGIN))*self.loss_mask, axis=0, name='cost_termination')/tf.reduce_sum(self.loss_mask) 

        self.cost_all = self.cost_intra_option_policy + self.cost_intra_option_entropy_agg + self.cost_critic + self.cost_termination

        # Cost: attention entropy
        if Config.USE_ATTENTION:
            if Config.ATTN_TYPE == Config.attn_multimodal:
                self.cost_attn_entrop = -1.*Config.BETA_ATTENTION*tf.reduce_sum(tf.log(tf.maximum(self.attn_softmaxes, self.log_epsilon))*self.attn_softmaxes, axis=1)
                self.cost_attn_entrop_agg = -1.*tf.reduce_sum(self.cost_attn_entrop*self.loss_mask, axis=0, name='cost_attn_entrop_agg')/tf.reduce_sum(self.loss_mask) # Negative since want to maximize entropy 
            elif Config.ATTN_TYPE == Config.attn_temporal:
                self.cost_attn_entrop_agg = 0
            self.cost_all += self.cost_attn_entrop_agg

    def _non_option_costs(self):
        # Cost: Critic (v) 
        self.logits_v = tf.squeeze(tf.layers.dense(inputs=self.final_flat, units=1, use_bias=True, activation=None, name='logits_v'), axis=[1])
        self.cost_v = 0.5*tf.reduce_sum(tf.square(self.y_r - self.logits_v)*self.loss_mask, axis=0)/tf.reduce_sum(self.loss_mask)

        # Cost: Actor (p advantage and entropy) 
        # Actor advantage cost
        self.logits_p = tf.layers.dense(inputs=self.final_flat, units=self.num_actions, name='logits_p', activation=None)
        self.softmax_p = tf.nn.softmax(self.logits_p) 
        self.selected_action_prob = tf.reduce_sum(self.softmax_p * self.action_index, axis=1, name='selection_action_prob')
        self.cost_p_advant = tf.log(tf.maximum(self.selected_action_prob, self.log_epsilon))*(self.y_r - tf.stop_gradient(self.logits_v)) # Stop_gradient ensures the value gradient feedback doesn't contribute to policy learning
        self.cost_p_advant_agg = -1.*tf.reduce_sum(self.cost_p_advant*self.loss_mask, axis=0, name='cost_p_advant_agg')/tf.reduce_sum(self.loss_mask) # Negative since want to maximize reward J function

        # Actor entropy cost
        self.cost_p_entrop = -1.*self.var_beta*tf.reduce_sum(tf.log(tf.maximum(self.softmax_p, self.log_epsilon))*self.softmax_p, axis=1)
        self.cost_p_entrop_agg = -1.*tf.reduce_sum(self.cost_p_entrop*self.loss_mask, axis=0, name='cost_p_entrop_agg')/tf.reduce_sum(self.loss_mask) #Negative since want to maximixe entropy

        self.cost_all = self.cost_p_advant_agg + self.cost_p_entrop_agg + self.cost_v

        # Cost: attention entropy
        if Config.USE_ATTENTION:
            if Config.ATTN_TYPE == Config.attn_multimodal:
                self.cost_attn_entrop = -1.*Config.BETA_ATTENTION*tf.reduce_sum(tf.log(tf.maximum(self.attn_softmaxes, self.log_epsilon))*self.attn_softmaxes, axis=1)
                self.cost_attn_entrop_agg = -1.*tf.reduce_sum(self.cost_attn_entrop*self.loss_mask, axis=0, name='cost_attn_entrop_agg')/tf.reduce_sum(self.loss_mask) # Negative since want to maximize entropy 
            elif Config.ATTN_TYPE == Config.attn_temporal:
                self.cost_attn_entrop_agg = 0
            self.cost_all += self.cost_attn_entrop_agg

    def ClipIfNotNone(self, grad):
        if grad is None:
            return grad
        return tf.clip_by_average_norm(grad, Config.GRAD_CLIP_NORM)

    def _create_graph_outputs(self):
        # Cost
        if Config.USE_OPTIONS:
            self._option_costs()
        else:
            self._non_option_costs()

        # Optimizer
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])
        if Config.OPTIMIZER == Config.OPT_RMSPROP:
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.var_learning_rate,
                                                 decay=Config.RMSPROP_DECAY,
                                                 momentum=Config.RMSPROP_MOMENTUM,
                                                 epsilon=Config.RMSPROP_EPSILON)
        elif Config.OPTIMIZER == Config.OPT_ADAM:
            self.opt = tf.train.AdamOptimizer(learning_rate=self.var_learning_rate)
        else:
            raise ValueError('Invalid optimizer chosen! Check Config.py!')

        # Grad clipping
        self.training_step = tf.Variable(0, trainable=False, name='step')
        if Config.USE_GRAD_CLIP:
            self.opt_grad = self.opt.compute_gradients(self.cost_all)
            self.opt_grad_clipped  = [(self.ClipIfNotNone(grad), var) for grad, var in self.opt_grad]
            self.train_op = self.opt.apply_gradients(self.opt_grad_clipped, global_step = self.training_step)
        else:
            self.train_op = self.opt.minimize(self.cost_all, global_step=self.training_step)

    def _create_tensorboard(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

        summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
        summaries.append(tf.summary.scalar("Beta - policy entropy coef", self.var_beta))
        summaries.append(tf.summary.histogram("activation_final_flat", self.final_flat))
        summaries.append(tf.summary.scalar("Cost_all", self.cost_all))

        if Config.USE_OPTIONS:
            summaries.append(tf.summary.scalar("Cost__intra_option_policy", self.cost_intra_option_policy)) 
            summaries.append(tf.summary.scalar("Cost__intra_option_entropy", self.cost_intra_option_entropy_agg)) 
            summaries.append(tf.summary.scalar("Cost__critic", self.cost_critic)) 
            summaries.append(tf.summary.scalar("Cost__option_termination", self.cost_termination))
            summaries.append(tf.summary.scalar("option_val_0", self.option_q_model[0,0]))
            summaries.append(tf.summary.scalar("option_val_1", self.option_q_model[0,1]))
            summaries.append(tf.summary.scalar("option_val_diff", self.option_q_model[0,0]-self.option_q_model[0,1]))
            summaries.append(tf.summary.scalar("cur_term_probs", self.cur_term_probs[0]))
            summaries.append(tf.summary.scalar("Option_epsilon", self.var_option_epsilon))
            summaries.append(tf.summary.scalar("Option_cost_delib", self.var_option_cost_delib))
        else:
            summaries.append(tf.summary.scalar("Cost__P_advantage", self.cost_p_advant_agg))
            summaries.append(tf.summary.scalar("Cost__P_entropy", self.cost_p_entrop_agg))
            summaries.append(tf.summary.scalar("Cost__v", self.cost_v))
            summaries.append(tf.summary.histogram("activation_v", self.logits_v))
            summaries.append(tf.summary.histogram("activation_p", self.softmax_p))
        if Config.USE_ATTENTION:
           summaries.append(tf.summary.scalar("Cost_attn_entropy", self.cost_attn_entrop_agg))

        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        for layer in self.layer_tracker:
            summaries.append(tf.summary.histogram(layer.name, layer))

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter(os.path.join(Config.LOGDIR), self.sess.graph)

    def _get_base_feed_dict(self):
        if Config.USE_OPTIONS:
            return {self.var_beta: self.beta, self.var_option_epsilon: self.option_epsilon, self.var_option_cost_delib: self.option_cost_delib, self.var_learning_rate: self.learning_rate}
        else:
            return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate}

    def update_feed_dict(self, feed_dict, x, audio, rnn_state, is_training, y_r = None, 
                         a = None, i_option = None, seq_lengths = None, loss_mask = None):
        if is_training:
            feed_dict.update({self.y_r: y_r, self.action_index: a})

        feed_dict.update({self.x: x})

        if Config.USE_AUDIO:
            feed_dict.update({self.input_audio: audio})

        if Config.USE_RNN:
            batch_size = np.shape(rnn_state)[0]
            if is_training:
                seq_lengths = np.array(seq_lengths)
            else:
                # Prediction done step-by-step, so seq_length is always 1
                seq_lengths = np.ones((batch_size,), dtype=np.int32) 
                loss_mask = self.create_loss_mask(seq_lengths)

            feed_dict.update({self.seq_lengths: seq_lengths, self.loss_mask: loss_mask})
            for i_lstm_layer in xrange(self.n_lstm_layers_total):
                cb = np.zeros((batch_size, Config.NCELLS))
                hb = np.zeros((batch_size, Config.NCELLS))
                for i_batch in xrange(batch_size):
                    cb[i_batch,:] = rnn_state[i_batch][i_lstm_layer]['c']
                    hb[i_batch,:] = rnn_state[i_batch][i_lstm_layer]['h']
                feed_dict.update({self.rnn_state_in['c'][i_lstm_layer]: cb})
                feed_dict.update({self.rnn_state_in['h'][i_lstm_layer]: hb})

                if Config.USE_ATTENTION:
                    if Config.ATTN_TYPE == Config.attn_multimodal:
                        attn_stateb = np.zeros((batch_size, Config.NMODES))
                        feed_dict.update({self.rnn_state_in['attn_state'][i_lstm_layer]: attn_stateb})
                    if Config.ATTN_TYPE == Config.attn_temporal:
                        attn_stateb = np.zeros((batch_size, Config.ATTN_STATE_NCELLS))
                        feed_dict.update({self.rnn_state_in['attn_state'][i_lstm_layer]: attn_stateb})
                        attn_state_histb = np.zeros((batch_size, Config.ATTN_TEMPORAL_WINDOW*Config.ATTN_STATE_NCELLS))
                        feed_dict.update({self.rnn_state_in['attn_state_hist'][i_lstm_layer]: attn_state_histb})

        if Config.USE_OPTIONS:
            feed_dict.update({self.option_index: i_option}) 

    def get_global_step(self):
        return self.sess.run(self.training_step)

    def predict_p_and_v(self, x, audio, rnn_state, i_option):
        batch_size = x.shape[0]
        feed_dict = self._get_base_feed_dict()
        self.update_feed_dict(feed_dict=feed_dict, x=x, audio=audio, rnn_state=rnn_state, i_option=i_option, is_training=False)

        if Config.USE_RNN:
            # NOTE: Due to many choices in Config, there are many if else statements. We will clean up this in the near future.
            if Config.USE_ATTENTION:
                if Config.ATTN_TYPE == Config.attn_multimodal:
                    if Config.USE_OPTIONS:
                        cur_intra_option_probs, option_v_model, option_q_model, termination, rnn_state_out, attn = \
                        self.sess.run([self.cur_intra_option_probs, self.option_v_model, self.option_q_model, self.termination_model, self.rnn_state_out, self.attn_softmaxes], feed_dict=feed_dict)
                    else:
                        p, v, rnn_state_out, attn = self.sess.run([self.softmax_p, self.logits_v, self.rnn_state_out, self.attn_softmaxes], feed_dict=feed_dict)
                elif Config.ATTN_TYPE == Config.attn_temporal:
                    p, v, rnn_state_out = self.sess.run([self.softmax_p,  self.logits_v, self.rnn_state_out], feed_dict=feed_dict)
            else:
                if Config.USE_OPTIONS:
                    cur_intra_option_probs, option_v_model, option_q_model, termination, rnn_state_out = \
                    self.sess.run([self.cur_intra_option_probs, self.option_v_model, self.option_q_model, self.termination_model, self.rnn_state_out], feed_dict=feed_dict)
                else:
                    p, v, rnn_state_out = self.sess.run([self.softmax_p, self.logits_v, self.rnn_state_out], feed_dict=feed_dict)

            # Update RNN states for next round. Also put them in batch-major order (for threadpredictor)
            rnn_state_out_batched = [None]*batch_size 
            for i_batch in xrange(batch_size):
                mdict = [{'c': None, 'h': None, 'attn_state': None, 'attn_state_hist': None} for i_layer in xrange(self.n_lstm_layers_total)] 
                rnn_state_out_batched[i_batch] = mdict
                for i_layer in xrange(self.n_lstm_layers_total):
                    rnn_state_out_batched[i_batch][i_layer]['c'] = rnn_state_out['c'][i_layer][i_batch] 
                    rnn_state_out_batched[i_batch][i_layer]['h'] = rnn_state_out['h'][i_layer][i_batch]

                    if Config.USE_ATTENTION:
                        rnn_state_out_batched[i_batch][i_layer]['attn_state'] = rnn_state_out['attn_state'][i_layer][i_batch] 
                        if Config.ATTN_TYPE == Config.attn_temporal:
                            rnn_state_out_batched[i_batch][i_layer]['attn_state_hist'] = rnn_state_out['attn_state_hist'][i_layer][i_batch] 
        else:
            if Config.USE_OPTIONS:
                cur_intra_option_probs, option_v_model, option_q_model, termination= \
                self.sess.run([self.cur_intra_option_probs, self.option_v_model, self.option_q_model, self.termination_model], feed_dict=feed_dict)
            else:
                p, v = self.sess.run([self.softmax_p, self.logits_v], feed_dict=feed_dict)

        if Config.USE_OPTIONS:
            predict_dict_batched = {'cur_intra_option_probs': cur_intra_option_probs, 'option_v_model': option_v_model, 'option_term_probs': termination, 'option_q_model': option_q_model}
        else:
            predict_dict_batched = {'p_actions': p, 'v': v}
        if Config.USE_RNN:
            predict_dict_batched['rnn_state_out'] = rnn_state_out_batched
            if Config.USE_ATTENTION and Config.ATTN_TYPE == Config.attn_multimodal:
                predict_dict_batched['attn'] = attn
        
        return predict_dict_batched

    def create_loss_mask(self, seq_lengths):
        if Config.USE_RNN:
            seq_lengths_size = len(seq_lengths)
            loss_mask= np.zeros((seq_lengths_size, Config.TIME_MAX), np.float32)
            for i_row in xrange(0, seq_lengths_size):
                loss_mask[i_row,:seq_lengths[i_row]] = 1.
            return loss_mask.flatten()
        else:
            return None

    def train(self, x, audio, y_r, a, i_option, rnn_state, seq_lengths):
        feed_dict = self._get_base_feed_dict()
        self.update_feed_dict(feed_dict, x, audio, rnn_state, is_training = True, y_r=y_r, 
                              a=a, i_option=i_option, seq_lengths=seq_lengths, 
                              loss_mask=self.create_loss_mask(seq_lengths))
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def log(self, ep_count, x, audio, y_r, a, i_option, rnn_state, seq_lengths, reward, roll_reward):
        feed_dict = self._get_base_feed_dict()
        self.update_feed_dict(feed_dict, x, audio, rnn_state, is_training=True, y_r=y_r, 
                              a=a, i_option=i_option, seq_lengths=seq_lengths, loss_mask=self.create_loss_mask(seq_lengths))

        summary = self.sess.run(self.summary_op, feed_dict=feed_dict)
        self.log_writer.add_summary(summary, ep_count)

        summary = tf.Summary(value=[tf.Summary.Value(tag="Reward", simple_value=reward)])
        self.log_writer.add_summary(summary, ep_count)

        summary = tf.Summary(value=[tf.Summary.Value(tag="Roll_Reward", simple_value=roll_reward)])
        self.log_writer.add_summary(summary, ep_count)

    def _checkpoint_filename(self):
        return os.path.join(Config.LOGDIR, 'checkpoints', 'network')

    def save(self, episode):
        episode_assign_op = self.episode.assign(episode)
        self.sess.run(episode_assign_op) # Save episode number in the checkpoint
        self.saver.save(self.sess, self._checkpoint_filename())
        print '[ INFO ] Saved current model to: {}'.format(self._checkpoint_filename())

    def load(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename()))

        if Config.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(Config.LOAD_EPISODE)
        try:
            self.saver.restore(self.sess, filename)
        except:
            raise ValueError('Error importing checkpoint! Are you sure checkpoint %s exists?' %self._checkpoint_filename())

        return self.sess.run(self.episode)

    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))
