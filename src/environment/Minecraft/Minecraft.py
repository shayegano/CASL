import cv2, os, sys, random
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.misc import imresize
from Config import Config
from python_speech_features import mfcc

class Actions():
    def __init__(self):
        self.NULL      = 0
        self.UP        = 1
        self.DOWN      = 2
        self.RIGHT     = 3
        self.LEFT      = 4

        self.num_actions = sum(1 for attribute in dir(self) if not attribute.startswith('__'))

class Minecraft():
    def __init__(self):
        self.env_row         = Config.ENV_ROW
        self.env_col         = Config.ENV_COL
        self.max_iter        = Config.MAX_ITER
        self.stacked_frames  = Config.STACKED_FRAMES
        self.cell_px_size    = Config.PIXEL_SIZE
        self.img_width       = Config.IMAGE_WIDTH
        self.img_height      = Config.IMAGE_HEIGHT
        self.simple_render   = Config.SIMPLE_RENDER
        self.actions         = Actions()
        self.count_iter      = 0
        self.agent_loc       = None
        self.target_good_loc = None
        self.target_bad_loc  = None
        self.gem_loc         = None
        self.gem_type_list   = ['gold', 'iron']

        self._init_rewards()
        self._init_gem_type()
        self._init_agent_target_gem_loc()
        self._init_render()
        if Config.USE_AUDIO:
            self._init_mfcc()
        self.reset()

    #########################################################################
    # ENV-RELATED
    def _init_rewards(self):
        self.reward_step =  -1
        self.reward_good =  10
        self.reward_bad  = -10
    
    def _init_gem_type(self):
        self.gem_type = random.choice(self.gem_type_list)

    def _init_agent_target_gem_loc(self):
        # Agent
        row = np.random.randint(low=0, high=self.env_row)
        col = np.random.randint(low=0, high=self.env_col)
        self.agent_loc = np.array([row, col], dtype=np.uint16)

        # Init targets, ensuring it does not initially overlap w agent
        self.target_good_loc = self._gen_xy_no_overlap()
        self.target_bad_loc  = self._gen_xy_no_overlap() 
        self.gem_loc         = self._gen_xy_no_overlap()

    def _gen_xy_no_overlap(self):
        is_not_overlapping_agt = False
        while is_not_overlapping_agt == False:
            row = np.random.randint(low=0, high=self.env_row)
            col = np.random.randint(low=0, high=self.env_col)
            is_not_overlapping_agt = self._check_overlap(row, col)
        return np.array([row, col], dtype=np.uint16)

    def _check_overlap(self, row, col):
        is_check = False
        # Location check for target_good
        if self.target_good_loc is None:
            if row != self.agent_loc[0] or col != self.agent_loc[1]:
                is_check = True

        else:
            # Location check for target_bad
            if self.target_bad_loc is None:
                if row != self.agent_loc[0] or col != self.agent_loc[1]:
                    if row != self.target_good_loc[0] or col != self.target_good_loc[1]:
                        is_check = True
            # Location check for gem_loc
            else:
                if row != self.agent_loc[0] or col != self.agent_loc[1]:
                    if row != self.target_good_loc[0] or col != self.target_good_loc[1]:
                        if row != self.target_bad_loc[0] or col != self.target_bad_loc[1]:
                            is_check = True
        return is_check

    #########################################################################
    # RL-RELATED
    def reset(self, reset_time = True):
        if reset_time:
            self.count_iter = 0
        self._init_gem_type()
        self._init_agent_target_gem_loc()
    
        # Send back initial obs
        return self._get_obs()

    def _get_obs(self, show_gt = False, return_agt_loc = False):
        if Config.USE_AUDIO:
            image, audio = self._get_image_and_audio(show_gt = show_gt)
            next_observation = [image, audio]
        else:
            next_observation = self._preprocess_img(show_gt = show_gt)

        if return_agt_loc:
            next_observation.append(self.agent_loc)

        return next_observation

    def _get_image_and_audio(self, show_gt = False):
        image = self._preprocess_img(show_gt = show_gt)

        # Audio output based on distance to gem and gem type
        dist_to_gem = self.dist_euclid(self.agent_loc, self.gem_loc)

        if dist_to_gem <= Config.LISTEN_RANGE:
            if self.gem_type == 'gold':
                audio = self.mfcc_target_good
            else:
                audio = self.mfcc_target_bad
        else:
            audio = self.mfcc_no_listen

        return image, audio

    def _take_action(self, action):
        if hasattr(self.actions,'NULL') and action == self.actions.NULL:
            dummy = 1
    
        elif hasattr(self.actions,'UP') and  action == self.actions.UP:
            new_loc = self.agent_loc[0]-1
            if new_loc < 0: new_loc = 0# Out of Boundary
            self.agent_loc = np.array([new_loc, self.agent_loc[1]])
    
        elif hasattr(self.actions,'DOWN') and action == self.actions.DOWN:
            new_loc = self.agent_loc[0]+1
            if new_loc >= self.env_row : new_loc = self.env_row-1# Out of Boundary
            self.agent_loc = np.array([new_loc, self.agent_loc[1]])
    
        elif hasattr(self.actions,'RIGHT') and action == self.actions.RIGHT:
            new_loc = self.agent_loc[1]+1
            if new_loc >= self.env_col : new_loc = self.env_col-1# Out of Boundary
            self.agent_loc = np.array([self.agent_loc[0], new_loc])
    
        elif hasattr(self.actions,'LEFT') and action == self.actions.LEFT:
            new_loc = self.agent_loc[1]-1
            if new_loc < 0: new_loc = 0# Out of Boundary
            self.agent_loc = np.array([self.agent_loc[0], new_loc])
    
        else:
            raise ValueError("[ ERROR ] Wrong action!")

    def step(self, action, pid, count):
        # Action
        self._take_action(action)
    
        # Reward
        if (np.array_equal(self.agent_loc, self.target_good_loc) == True and self.gem_type == 'gold') or \
           (np.array_equal(self.agent_loc, self.target_bad_loc) == True and self.gem_type == 'iron'):
            reward = self.reward_good 
        elif (np.array_equal(self.agent_loc, self.target_good_loc) == True and self.gem_type == 'iron') or \
             (np.array_equal(self.agent_loc, self.target_bad_loc) == True and self.gem_type == 'gold'):
            reward = self.reward_bad 
        else:
            reward = self.reward_step 
    
        # Observation
        next_observation = self._get_obs()      
    
        # Count iteration
        self.count_iter += 1
    
        # Game over
        game_over = False
        is_maxiter_reached = self.count_iter >= (self.max_iter + self.stacked_frames -1)
        if np.array_equal(self.agent_loc, self.target_good_loc) or \
           np.array_equal(self.agent_loc, self.target_bad_loc) or \
           is_maxiter_reached:
            game_over = True
            self.reset(reset_time = True)
    
        return next_observation, reward, game_over

    #########################################################################
    # AUDIO-RELATED
    def _init_mfcc(self):
        self.mfcc_target_good = self._get_mfcc(os.path.join(sys.path[0], '../../environment/Minecraft/assets/target_good.wav'))
        self.mfcc_target_bad  = self._get_mfcc(os.path.join(sys.path[0], '../../environment/Minecraft/assets/target_bad.wav'))
        self.mfcc_no_listen   = self._get_mfcc(os.path.join(sys.path[0], '../../environment/Minecraft/assets/noise.wav'))

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
        self.lg_boundary_px_size = 2  # In pixel

        if self.simple_render:
            self.img_stone          = np.zeros((self.cell_px_size, self.cell_px_size, 3))
            self.img_agent_in_stone = np.zeros((self.cell_px_size, self.cell_px_size, 3)) + 150
            self.img_gold_in_stone  = np.zeros((self.cell_px_size, self.cell_px_size, 3)) + 250
            self.img_iron_in_stone  = np.zeros((self.cell_px_size, self.cell_px_size, 3)) + 50
            self.img_stone_gold     = np.zeros((self.cell_px_size, self.cell_px_size, 3)) + 100
            self.img_stone_iron     = np.zeros((self.cell_px_size, self.cell_px_size, 3)) + 200

        else:
            self.img_agent      = self._read_img('../../environment/Minecraft/TexturePacker/All/Characters/Player_Male/male.png', interp = cv2.INTER_NEAREST)
            self.img_stone      = self._read_img('../../environment/Minecraft/TexturePacker/All/Tiles/stone.png')
            self.img_pick_gold  = self._read_img('../../environment/Minecraft/TexturePacker/All/Items/pick_gold.png', interp = cv2.INTER_NEAREST)
            self.img_pick_iron  = self._read_img('../../environment/Minecraft/TexturePacker/All/Items/shovel_bronze.png', interp = cv2.INTER_NEAREST)
            self.img_stone_gold = self._read_img('../../environment/Minecraft/TexturePacker/All/Tiles/stone_gold.png')
            self.img_stone_iron = self._read_img('../../environment/Minecraft/TexturePacker/All/Tiles/stone_iron.png')

            self.img_agent_in_stone = self._overlay_imgs(self.img_agent, self.img_stone)
            self.img_gold_in_stone  = self._overlay_imgs(self.img_pick_gold, self.img_stone)
            self.img_iron_in_stone  = self._overlay_imgs(self.img_pick_iron, self.img_stone)

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
            return self.img_agent_in_stone

        elif row == self.target_good_loc[0] and col == self.target_good_loc[1]:
            return self.img_gold_in_stone

        elif row == self.target_bad_loc[0] and col == self.target_bad_loc[1]:
            return self.img_iron_in_stone

        elif row == self.gem_loc[0] and col == self.gem_loc[1]:
            if show_gt:
                if self.gem_type == 'gold':
                    return self.img_stone_gold
                else:
                    return self.img_stone_iron
            else:
                return self.img_stone_gold
        else:
            return self.img_stone

        return cell

    def _preprocess_img(self, show_gt = False):
        img = self._render_grid(show_gt = show_gt)
        if show_gt:
            return img
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
                                     dtype=np.uint8)

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

    #########################################################################
    # ETC
    def dist_euclid(self, loc_a, loc_b):
        # If type is uint then it will throw a huge number! :o
        loc_a = np.asarray(loc_a, dtype=np.int32)
        loc_b = np.asarray(loc_b, dtype=np.int32)

        dist = np.linalg.norm(loc_a - loc_b) 
        if dist > 1000:
            raise RuntimeError("Dist is too large! Something is wrong.")
        
        return dist
