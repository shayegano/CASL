import cv2, os, sys, random
import numpy as np
import scipy.io.wavfile as wav
from scipy.misc import imresize
from Config import Config
from python_speech_features import mfcc

class Actions():
    def __init__(self):
        self.UP        = 0
        self.DOWN      = 1
        self.RIGHT     = 2
        self.LEFT      = 3

        self.num_actions = sum(1 for attribute in dir(self) if not attribute.startswith('__'))

class Doorpuzzle():
    def __init__(self):
        self.env_row         = Config.ENV_ROW
        self.env_col         = Config.ENV_COL
        self.max_iter        = Config.MAX_ITER
        self.stacked_frames  = Config.STACKED_FRAMES
        self.cell_px_size    = Config.PIXEL_SIZE
        self.img_width       = Config.IMAGE_WIDTH
        self.img_height      = Config.IMAGE_HEIGHT
        self.simple_render   = Config.SIMPLE_RENDER
        self.hard_mode       = Config.HARD_MODE
        self.actions         = Actions()
        self.count_iter      = 0
        self.has_key         = False
        self._check_params()

        self._init_rewards()
        self._init_key_types()
        self._init_agent_targets_key_loc()
        self._init_obstacle_matrix()
        self._init_render()
        if Config.USE_AUDIO:
            self._init_mfcc()

        self.reset(reset_time = True)

    def dist_manhat(self, loc_a, loc_b):
        return abs(loc_a[0] - loc_b[0]) + abs(loc_a[1] - loc_b[1])

    def dist_euclid(self, loc_a, loc_b):
       return np.linalg.norm(loc_a - loc_b) 

    #########################################################################
    # ENV-RELATED
    def _check_params(self):
        if self.env_row != 5 or self.env_col != 5:
            raise RuntimeError("[ TODO ] Not implemented yet")

        if self.simple_render == False:
            raise RuntimeError("[ TODO ] Not implemented yet")

    def _init_rewards(self):
        self.reward_step = 0.0
        self.reward_good = 1.0

    def _check_overlap(self, row, col):
        if row == self.agent_loc[0] and col == self.agent_loc[1]:
            return True
        elif row == self.target1_loc[0] and col == self.target1_loc[1]:
            return True
        elif row == self.target1_loc[0] and col == self.target1_loc[1]:
            return True
        else:
            return False
    
    def _init_agent_targets_key_loc(self):
        # Agent location
        self.agent_loc = np.array([0, 0], dtype = np.uint16)

        # Target location
        self.target1_loc = np.array([0, self.env_col - 1], dtype = np.uint16)
        self.target2_loc = np.array([self.env_row - 1, 0], dtype = np.uint16)

        # Key loc
        if self.hard_mode:
            row = np.random.randint(low = 0, high = self.env_row)
            col = np.random.randint(low = 0, high = self.env_col)
            while self._check_overlap(row, col):
                row = np.random.randint(low = 0, high = self.env_row)
                col = np.random.randint(low = 0, high = self.env_col)
            self.key_loc = np.array([row, col], dtype = np.uint16)
        else:
            center_row = int(np.floor(self.env_row / 2.))
            center_col = int(np.floor(self.env_col / 2.))
            self.key_loc = np.array([center_row, center_col], dtype = np.uint16)

    def _init_obstacle_matrix(self):
        self.obstacle_matrix = np.zeros((self.env_row, self.env_col))

    def _init_key_types(self):
        self.key_type = random.choice([1, 2])

    #########################################################################
    # RL-RELATED
    def reset(self, reset_time = True):
        if reset_time:
            self.count_iter = 0
        self._init_key_types()
        self._init_agent_targets_key_loc()
        self.has_key = False

        return self._get_obs()

    def reset_to_stage2(self):
        self.has_key    = True

    def _get_obs(self, show_gt = False, return_agt_loc = False):
        if Config.USE_AUDIO:
            image, audio = self._get_image_and_audio(show_gt = show_gt)
            next_observation = [image, audio]
        else:
            next_observation = self._preprocess_img(show_gt = show_gt)

        if return_agt_loc:
            if Config.USE_AUDIO:
                next_observation.append(self.agent_loc)
            else:
                # NOTE send back [image, None for audio, location]
                next_observation = [next_observation, None, self.agent_loc]

        return next_observation

    def _get_image_and_audio(self, show_gt = False):
        image = self._preprocess_img(show_gt = show_gt)

        if self.has_key == True:
            audio = self.mfcc_no_listen
        else:
            dist_to_key = self.dist_euclid(self.agent_loc, self.key_loc)
            if dist_to_key <= Config.LISTEN_RANGE:
                if self.key_type == 1:
                    audio = self.mfcc_target1
                else:
                    audio = self.mfcc_target2
            else:
                audio = self.mfcc_no_listen

        return image, audio

    def _action_noise(self, action):
        #  if hasattr(self.actions,'NULL') and action == self.actions.NULL:
            #  return action
        if hasattr(self.actions,'UP') and action == self.actions.UP:
            actions_possib = [self.actions.LEFT, self.actions.UP, self.actions.RIGHT]
        elif hasattr(self.actions,'RIGHT') and action == self.actions.RIGHT:
            actions_possib = [self.actions.UP, self.actions.RIGHT, self.actions.DOWN]
        elif hasattr(self.actions,'DOWN') and action == self.actions.DOWN:
            actions_possib = [self.actions.RIGHT, self.actions.DOWN, self.actions.LEFT]
        elif hasattr(self.actions,'LEFT') and action == self.actions.LEFT:
            actions_possib = [self.actions.DOWN, self.actions.LEFT, self.actions.UP]
        else:
            print action
            raise ValueError("[ ERROR ] Wrong action!")

        i_action_sampled = np.random.choice(3, p=[Config.NOISE_TRANS/2.0, 1.0-Config.NOISE_TRANS, Config.NOISE_TRANS/2.0])
        return actions_possib[i_action_sampled]

    def _is_obstacle_free(self, row, col):
        if self.obstacle_matrix[row, col] == 0:
            return True
        else:
            return False

    def _take_action(self, action):
        # Apply action noise (NOTE equivalent to state transition noise)
        action = self._action_noise(action)

        if hasattr(self.actions,'UP') and action == self.actions.UP:
            new_loc = self.agent_loc[0] - 1
            if new_loc < 0: new_loc = 0 # Out of Boundary

            if self._is_obstacle_free(new_loc, self.agent_loc[1]):
                self.agent_loc = np.array([new_loc, self.agent_loc[1]])
    
        elif hasattr(self.actions,'DOWN') and action == self.actions.DOWN:
            new_loc = self.agent_loc[0] + 1
            if new_loc >= self.env_row : new_loc = self.env_row-1 # Out of Boundary

            if self._is_obstacle_free(new_loc, self.agent_loc[1]):
                self.agent_loc = np.array([new_loc, self.agent_loc[1]])
    
        elif hasattr(self.actions,'RIGHT') and action == self.actions.RIGHT:
            new_loc = self.agent_loc[1] + 1
            if new_loc >= self.env_col : new_loc = self.env_col-1 # Out of Boundary

            if self._is_obstacle_free(self.agent_loc[0], new_loc):
                self.agent_loc = np.array([self.agent_loc[0], new_loc])
    
        elif hasattr(self.actions,'LEFT') and action == self.actions.LEFT:
            new_loc = self.agent_loc[1] - 1
            if new_loc < 0: new_loc = 0 # Out of Boundary

            if self._is_obstacle_free(self.agent_loc[0], new_loc):
                self.agent_loc = np.array([self.agent_loc[0], new_loc])
    
        else:
            raise ValueError("[ ERROR ] Wrong action!")

    def step(self, action, pid = None, count = None):
        # Action
        self._take_action(action)
    
        # Reward
        if np.array_equal(self.agent_loc, self.key_loc) and self.has_key == False:
            reward = self.reward_step
            self.reset_to_stage2()
        elif np.array_equal(self.agent_loc, self.target1_loc):
            if self.has_key == False:
                reward = self.reward_step
            else:
                if self.key_type == 1:
                    reward = self.reward_good
                else:
                    reward = self.reward_step
        elif np.array_equal(self.agent_loc, self.target2_loc):
            if self.has_key == False:
                reward = self.reward_step
            else:
                if self.key_type == 2:
                    reward = self.reward_good
                else:
                    reward = self.reward_step
        else:
            reward = self.reward_step 
    
        # Observation
        next_observation = self._get_obs()      
    
        # Count iteration
        self.count_iter += 1
    
        # Game over
        game_over = False

        is_maxiter_reached = self.count_iter >= (self.max_iter + self.stacked_frames -1)
        if is_maxiter_reached:
            game_over = True
            self.reset(reset_time = True)

        if np.array_equal(self.agent_loc, self.target1_loc):
            game_over = True
            self.reset(reset_time = True)

        if np.array_equal(self.agent_loc, self.target2_loc):
            game_over = True
            self.reset(reset_time = True)

        return next_observation, reward, game_over

    #########################################################################
    # AUDIO-RELATED
    def _init_mfcc(self):
        self.mfcc_target1   = self._get_mfcc(os.path.join(sys.path[0], '../../environment/Doorpuzzle/assets/target_good.wav'))
        self.mfcc_target2   = self._get_mfcc(os.path.join(sys.path[0], '../../environment/Doorpuzzle/assets/target_bad.wav'))
        self.mfcc_no_listen = self._get_mfcc(os.path.join(sys.path[0], '../../environment/Doorpuzzle/assets/noise.wav'))

    def _get_mfcc(self, filename):
        (samplerate, audio) = wav.read(filename)
        return self._wav_to_mfcc(samplerate, audio)
    
    # TODO move this to a utils dir so all code can use the same one
    def _wav_to_mfcc(self, samplerate, wav):
        mfcc_data = mfcc(signal=wav, samplerate=samplerate)
        mfcc_data = np.swapaxes(mfcc_data, 0 ,1) # To make time axis to be in x-axis 
        mfcc_data = imresize(mfcc_data, (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT), interp='cubic') 
    
        # TODO I think even this should be normalized more carefully (i.e., with an absolute system rather than relative)
        min_data = np.min(mfcc_data.flatten())
        max_data = np.max(mfcc_data.flatten())
        mfcc_data = 1.*(mfcc_data -min_data)/(max_data-min_data)
        mfcc_data = mfcc_data *2.-1.
    
        return mfcc_data

    #########################################################################
    # IMAGE-RELATED
    def _init_render(self):
        self.lg_boundary_px_size = 2 # In pixel

        if self.simple_render:
            self.img_background = np.zeros((self.cell_px_size, self.cell_px_size, 3)) + 128
            self.img_key1       = np.zeros((self.cell_px_size, self.cell_px_size, 3)) + 0
            self.img_target1    = np.zeros((self.cell_px_size, self.cell_px_size, 3)) + 50
            self.img_key2       = np.zeros((self.cell_px_size, self.cell_px_size, 3)) + 235
            self.img_target2    = np.zeros((self.cell_px_size, self.cell_px_size, 3)) + 185
            self.img_agent      = np.zeros((self.cell_px_size, self.cell_px_size, 3)) + 255
            self.img_obstacle   = np.zeros((self.cell_px_size, self.cell_px_size, 3)) + 100

        else:
            self.img_background = self._read_img('../../environment/Doorpuzzle/TexturePacker/All/Tiles/stone.png')
            self.img_target1    = self._read_img('../../environment/Doorpuzzle/TexturePacker/All/Tiles/stone_gold.png')
            self.img_target2    = self._read_img('../../environment/Doorpuzzle/TexturePacker/All/Tiles/stone_iron.png')

            self.img_key1       = self._read_img('../../environment/Doorpuzzle/TexturePacker/All/Items/pick_gold.png', 
                                                 interp = cv2.INTER_NEAREST)
            self.img_key1       = self._overlay_imgs(self.img_key1, self.img_background)

            self.img_key2       = self._read_img('../../environment/Doorpuzzle/TexturePacker/All/Items/shovel_bronze.png', 
                                                 interp = cv2.INTER_NEAREST)
            self.img_key2       = self._overlay_imgs(self.img_key1, self.img_background)

            self.img_obstacle   = self._read_img('../../environment/Doorpuzzle/TexturePacker/All/Tiles/fence_wood.png', 
                                                 interp = cv2.INTER_NEAREST)
            self.img_obstacle   = self._overlay_imgs(self.img_obstacle, self.img_background)

            self.img_agent      = self._read_img('../../environment/Doorpuzzle/TexturePacker/All/Characters/Player_Male/male.png', 
                                                 interp = cv2.INTER_NEAREST)
            self.img_agent      = self._overlay_imgs(self.img_agent, self.img_background)

    def _read_img(self, img_path, interp=cv2.INTER_CUBIC):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = cv2.resize(img, (self.cell_px_size, self.cell_px_size), interpolation=interp)

        return img

    def _overlay_imgs(self, img_foreground, img_background):
        img = np.zeros((img_foreground.shape[0], img_foreground.shape[1], 3))

        for row in range(img_foreground.shape[0]):
            for col in range(img_background.shape[1]):
                if np.array_equal(img_foreground[row, col, :], np.array([0, 0, 0])) or \
                   np.array_equal(img_foreground[row, col, :], np.array([255, 255, 255])):
                    img[row, col, :] = img_background[row, col, :]
                else:
                    img[row, col, :] = img_foreground[row, col, :]

        return img

    def _render_boundary(self, grid, boundary_px_size, value):
        grid[0:0+boundary_px_size, :, :] = value
        grid[:, 0:0+boundary_px_size, :] = value
        grid[grid.shape[0]-boundary_px_size:grid.shape[0], :, :] = value
        grid[:, grid.shape[0]-boundary_px_size:grid.shape[0], :] = value

        return grid
        
    def _render_cell(self, row, col, show_gt):
        if row == self.agent_loc[0] and col == self.agent_loc[1]:
            return self.img_agent

        elif row == self.target1_loc[0] and col == self.target1_loc[1]:
            return self.img_target1

        elif row == self.target2_loc[0] and col == self.target2_loc[1]:
            return self.img_target2

        elif row == self.key_loc[0] and col == self.key_loc[1]:
            if self.has_key:
                return self.img_background
            else:
                if self.key_type == 1:
                    return self.img_key1
                else:
                    return self.img_key2
        else:
            if self.obstacle_matrix[row, col] == 1:
                return self.img_obstacle
            else:
                return self.img_background

    def _preprocess_img(self, show_gt = False):
        img = self._render_grid(show_gt = show_gt)
        if show_gt: return img

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if self.simple_render:
            img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        else:
            img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_CUBIC)
        img = np.asarray(img, dtype=np.float32)

        return img*2./255.-1

    def _render_grid(self, show_gt):
        gridworld_render = np.zeros((self.env_row*self.cell_px_size + 2*self.lg_boundary_px_size,
                                     self.env_col*self.cell_px_size + 2*self.lg_boundary_px_size, 3),
                                     dtype=np.uint8) + 125

        # Render boundary
        gridworld_render = self._render_boundary(gridworld_render, self.lg_boundary_px_size, 0)

        for row in range(self.env_row):
            for col in range(self.env_col):
                row_from = row*self.cell_px_size + self.lg_boundary_px_size
                col_from = col*self.cell_px_size + self.lg_boundary_px_size
                row_to   = (row+1)*self.cell_px_size + self.lg_boundary_px_size
                col_to   = (col+1)*self.cell_px_size + self.lg_boundary_px_size

                gridworld_render[row_from:row_to, col_from:col_to, :] = self._render_cell(row, col, show_gt)

        return gridworld_render
