# Copyright (c) 2016, NVIDIA CORPORATION. All rights r, option, option, option, option, optioneserved.
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
from datetime import datetime
from multiprocessing import Process, Queue, Value
from Environment import Environment
from Experience import Experience
from OptionTracker import OptionTracker
from models import CustomLayers

class ProcessAgent(Process):
    def __init__(self, model, id, prediction_q, training_q, episode_log_q, num_actions, stats):
        super(ProcessAgent, self).__init__()
        self.model                  = model
        self.id                     = id
        self.prediction_q           = prediction_q
        self.training_q             = training_q
        self.episode_log_q          = episode_log_q
        self.num_actions            = num_actions
        self.actions                = np.arange(self.num_actions)
        self.discount_factor        = Config.DISCOUNT
        self.wait_q                 = Queue(maxsize=1)
        self.exit_flag              = Value('i', 0)
        self.stats                  = stats
        self.last_vis_episode_num   = 0
        self.is_vis_training        = False # Initialize to False
        self.is_option_tracker_on   = False

        # NOTE: Disable for now
        # if Config.PLAY_MODE and Config.LOAD_CHECKPOINT and Config.USE_OPTIONS:
        #     self.is_option_tracker_on = True
        #     self.option_tracker = OptionTracker()

    @staticmethod
    def _accumulate_rewards(experiences, discount_factor, terminal_reward, game_done):
        reward_sum = terminal_reward # terminal_reward is called R in a3c paper

        returned_exp = experiences[:-1] # Returns all but final experience in most cases. Final exp saved for next batch. 
        leftover_term_exp = None # For special case where game finishes but with 1 experience longer than TMAX
        n_exps = len(experiences)-1 # Does n_exps-step backward updates on all experiences

        # Exception case for experiences length of 0
        if len(experiences) == 1:
            return experiences, leftover_term_exp 
        else:
            if game_done and len(experiences) == Config.TIME_MAX+1:
                leftover_term_exp = [experiences[-1]]
            if game_done and len(experiences) != Config.TIME_MAX+1:
                n_exps = len(experiences)
                returned_exp = experiences

            for t in reversed(xrange(0, n_exps)):
                # experiences[t].reward is single-step reward here
                reward_sum = discount_factor * reward_sum + experiences[t].reward
                # experiences[t]. reward now becomes y_r (target reward, with discounting), and is used as y_r in training thereafter. I.e., variable name is overloaded.
                experiences[t].reward = reward_sum 

            # Final experience is removed 
            return returned_exp, leftover_term_exp

    def convert_to_nparray(self, experiences):
        x_ = np.array([exp.state_image for exp in experiences])
        r_ = np.array([exp.reward for exp in experiences])
        a_ = np.eye(self.num_actions)[np.array([exp.action for exp in experiences], dtype=np.int32)].astype(np.float32)
        o_ = np.array([exp.option for exp in experiences])

        if Config.USE_AUDIO:
            audio_ = np.array([exp.state_audio for exp in experiences])
            return x_, audio_, r_, a_, o_
        else:
            return x_, None, r_, a_, o_

    def predict(self, current_state, rnn_state, i_option):
        if Config.USE_AUDIO:
            state_image = current_state[0]
            state_audio = current_state[1]
            assert state_image is not None
            assert state_audio is not None
        else:
            state_image = current_state
            state_audio = None
            assert state_image is not None

        # Put the state in the prediction q
        self.prediction_q.put((self.id, state_image, state_audio, rnn_state, i_option))

        # Wait for the prediction to come back
        return self.wait_q.get()

    def select_action(self, prediction_dict):
        if Config.USE_OPTIONS:
            if Config.PLAY_MODE:
                return np.argmax(prediction_dict['cur_intra_option_probs'])
            else:
                return np.random.choice(self.actions, p=prediction_dict['cur_intra_option_probs'])
        else:
            if Config.PLAY_MODE:
                return np.argmax(prediction_dict['p_actions'])
            else:
                return np.random.choice(self.actions, p=prediction_dict['p_actions'])

    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def select_option(self, prediction_dict):
        if Config.PLAY_MODE:
            return np.random.choice(Config.NUM_OPTIONS, p=self.softmax(prediction_dict['option_q_model']))
        else:
            if np.random.rand() > self.model.option_epsilon:
                return np.random.choice(Config.NUM_OPTIONS, p=self.softmax(prediction_dict['option_q_model']))
            else:
                return np.random.randint(Config.NUM_OPTIONS)

    def run_episode(self):
        self.env.reset()
        game_done         = False
        experiences       = []
        time_count        = 0
        frame_count       = 0
        reward_sum_logger = 0.0

        if Config.USE_OPTIONS:
            self.option_terminated = True

        if Config.USE_RNN:
            # input states for prediction
            rnn_state = CustomLayers.RNNInputStateHandler.get_rnn_dict(init_with_zeros=True, 
                                                                       n_lstm_layers_total=self.model.n_lstm_layers_total)
            # input states for training
            init_rnn_state = CustomLayers.RNNInputStateHandler.get_rnn_dict(init_with_zeros=True, 
                                                                            n_lstm_layers_total=self.model.n_lstm_layers_total)
        else:
            rnn_state      = None
            init_rnn_state = None

        while not game_done:
            # Initial step (used to ensure frame_q is full before trying to grab a current_state for prediction)
            if Config.USE_AUDIO and (self.env.current_state[0] is None and self.env.current_state[1] is None):
                self.env.step(0) # Action 0 corresponds to null action 
                continue
            elif self.env.current_state is None:
                self.env.step(0) # Action 0 corresponds to null action
                continue

            # Option prediction
            if Config.USE_OPTIONS:
                if self.option_terminated:
                    i_option = 0 # NOTE Fake option input
                    prediction_dict = self.predict(self.env.current_state, rnn_state, i_option)
                    i_option = self.select_option(prediction_dict)# NOTE Select option correctly in here
            else:
                i_option = None
                 
            # Primitive action prediction (for option and non-option cases)
            prediction_dict = self.predict(self.env.current_state, rnn_state, i_option)

            # Update rnn_state
            if Config.USE_RNN:
                rnn_state = prediction_dict['rnn_state_out']

            # Visualize train process or test process
            if (self.id == 0 and self.is_vis_training) or Config.PLAY_MODE:
                if Config.USE_ATTENTION: 
                    self.vis_attention_i.append(prediction_dict['attn'][0])
                    self.vis_attention_a.append(prediction_dict['attn'][1])
                else:
                    self.vis_attention_i = None
                    self.vis_attention_a = None
            
                self.env.visualize_env(self.vis_attention_i, self.vis_attention_a)

            # Select action
            i_action = self.select_action(prediction_dict)

            # Take action --> Receive reward, game_done (and also store self.env.previous_state for access below)
            reward, game_done = self.env.step(i_action)
            reward = np.clip(reward, Config.REWARD_MIN, Config.REWARD_MAX)

            if Config.USE_OPTIONS:
                reward -= float(self.option_terminated)*self.model.option_cost_delib*float(frame_count > 1)
                self.option_terminated = prediction_dict['option_term_probs'][i_option] > np.random.rand()
            reward_sum_logger += reward # Used for logging only
            
            # Add to experience
            if Config.USE_AUDIO:
                exp = Experience(self.env.previous_state[0], self.env.previous_state[1],
                                 i_action, i_option, reward, game_done)
            else:
                exp = Experience(self.env.previous_state, None,
                                 i_action, i_option, reward, game_done)
            experiences.append(exp)
            
            # Plot option trajectories
            if self.is_option_tracker_on:
                self.option_tracker._update_tracker(agt_loc, i_option, self.option_terminated)
                self.option_tracker._plot_tracker()

            # Config.TIME_MAX controls how often data is yielded/sent back to the for loop in the run(). 
            # It is used to ensure, for games w long episodes, that data is sent back to the trainers sufficiently often
            # The shorter Config.TIME_MAX is, the more often the data queue is updated 
            if game_done or time_count == Config.TIME_MAX:# or self.option_terminated:
                if Config.USE_OPTIONS:
                    if self.option_terminated:
                        value = prediction_dict['option_v_model'] - self.model.option_cost_delib*float(frame_count > 1)
                    else:
                        value = prediction_dict['option_q_model'][i_option]
                    terminal_reward = 0 if game_done else value 
                else:
                    terminal_reward = 0 if game_done else prediction_dict['v'] # See A3C paper, Algorithm S2 (n-step q-learning) 

                updated_exps, updated_leftover_exp = ProcessAgent._accumulate_rewards(experiences, self.discount_factor, terminal_reward, game_done)
                x_, audio_, r_, a_, o_ = self.convert_to_nparray(updated_exps) 
                yield x_, audio_, r_, a_, o_, init_rnn_state, reward_sum_logger # Send back data and start here next time fcn is called

                reward_sum_logger = 0.0 # NOTE total_reward_logger in self.run() accumulates reward_sum_logger, so reset here 

                if updated_leftover_exp is not None:
                    x_, audio_, r_, a_, o_ = self.convert_to_nparray(updated_leftover_exp) 
                    yield x_, audio_, r_, a_, o_, init_rnn_state, reward_sum_logger 

                # Reset the tmax count
                time_count = 0

                # Keep the last experience for the next batch
                experiences = [experiences[-1]]

                if Config.USE_RNN:
                    init_rnn_state = rnn_state 

            time_count += 1
            frame_count += 1

    def run(self):
        # Randomly sleep up to 1 second. Helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 5000 + self.id * 10))

        self.env = Environment() 

        while self.exit_flag.value == 0:
            total_reward_logger = 0
            total_length        = 0

            # For visualizing train process
            if self.id == 0:
                self.current_episode_num = self.stats.episode_count.value
                if ((self.current_episode_num - self.last_vis_episode_num > Config.VIS_FREQUENCY)) or Config.PLAY_MODE:
                    self.is_vis_training = True
                    if Config.USE_ATTENTION:
                        self.vis_attention_i = []
                        self.vis_attention_a = []

            for x_, audio_, r_, a_, o_, rnn_state_, reward_sum_logger in self.run_episode():
                if len(x_.shape) <= 1:
                    raise RuntimeError("x_ has invalid shape")
                total_reward_logger += reward_sum_logger
                total_length += len(r_) + 1  # +1 for last frame that we drop
                if Config.TRAIN_MODELS: 
                    self.training_q.put((x_, audio_, r_, a_, o_, rnn_state_)) # NOTE audio_ and rnn_state_ might be None depending on Config.USE_AUDIO/USE_RNN

            self.episode_log_q.put((datetime.now(), total_reward_logger, total_length))

            # Close visualizing train process
            if (self.id == 0 and self.is_vis_training) or Config.PLAY_MODE:
                self.is_vis_training = False
                self.last_vis_episode_num = self.current_episode_num
