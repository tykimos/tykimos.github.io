import curses
import time
import sys
import numpy as np
from random import randint
import random
from enum import Enum, IntEnum

import pylab
import random
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from IPython import get_ipython
from IPython.display import clear_output
import os

MAP_WIDTH = 16
MAP_HEIGHT = 16
ROVER_MAX_ENERGY = 40
MAX_EPISODE = 500000
RENDERING_TIME_SLEEP = 1.

class MapBlock(IntEnum):
    AIR = 0
    VISIT = 1
    CRATERS = 2
    HELIUM3 = 3
    ROVER = 4

MAP_BLOCKS_ICONS = {
    0: '. ', 
    1: '  ',     
    2: '▓▓',
    3: '**',
    4: '▞▚',
}

class StandardRenderer:

    def __init__(self):
        self.screen = None

    def set_screen(self, screen):
        pass

    def get_key(self):
        pass

    def render(self, map, rover_current_energy, moving_distance, mineral_sampling):        

        if get_ipython() is None:
            os.system('clear')
        else:
            clear_output(wait=True)

        str_draw = ''
        for row in range(MAP_HEIGHT):
            for col in range(MAP_WIDTH):
                str_draw += MAP_BLOCKS_ICONS[map[row][col]]
            str_draw += '\n'

        print(str_draw, flush=True)

    def logging(self, str_log):
        print(str_log)
        
class CursesRenderer:

    def __init__(self):
        self.screen = None
    
    def set_screen(self, screen):
        self.screen = screen
        self.screen.timeout(0)

    def get_key(self):
        return self.screen.getch()

    def render(self, map, rover_current_energy, moving_distance, mineral_sampling):

        for row in range(MAP_HEIGHT):
            row_string = ''
            for col in range(MAP_WIDTH):
                row_string += MAP_BLOCKS_ICONS[map[row][col]]
            self.screen.addstr(row, 0, row_string)

        self.screen.addstr(0, 0, '%4d♥' % rover_current_energy)
        self.screen.addstr(0, 5, '%4dm' % moving_distance)
        self.screen.addstr(0, 10, '%4d◆' % mineral_sampling)
        self.screen.addstr(16, 0, '')
        self.screen.refresh()

    def logging(self, str_log):
        self.screen.addstr(16, 0, str_log)

class Observation:

    def __init__(self, state_size):
        self.state_size = state_size
        self.rover_current_energy = 0
        self.rover_current_y = 0
        self.visual_range = 0
        self.rover_view_x = 0
        self.map = None

    def get_state(self):
    
        state = np.zeros(self.state_size)
        state[0] = self.rover_current_energy
        state[1] = MAP_HEIGHT - 1 - self.rover_current_y
        
        for col in range(self.visual_range):
            
            item_row_index = 15

            for row in range(MAP_HEIGHT):
                
                if self.map[row][self.rover_view_x + col] == MapBlock.HELIUM3:
                    item_row_index = row
                    break

            state[col + 2] = MAP_HEIGHT - 1 - item_row_index

        state = np.reshape(state, [1, self.state_size])

        return state
"""
# 각 타임스텝마다 상태 전처리
        next_state = np.reshape(next_state, [1, state_size])
        state = np.reshape(state, [1, state_size])
"""

class Environment:

    def __init__(self):

        self.map = None
        self.renederer = None
        self.visual_range = 11
        self.state_size = 1 + 1 + self.visual_range
        self.action_size = 2
        self.rover_view_x = 4

        self.reset()

    def reset(self):
            
        self.map = np.zeros((MAP_HEIGHT, MAP_WIDTH))
        self.map[MAP_HEIGHT-1][:] = MapBlock.HELIUM3

        for col in range((int)(MAP_WIDTH/2+1), MAP_WIDTH):

            if col < MAP_WIDTH/2 + 4:
                new_wall_height = random.randrange(0, 2) # 1부터 6미만 숫자. 1, 2, 3, 4, 5
            else:
                new_wall_height = random.randrange(0, 3) # 1부터 6미만 숫자. 1, 2, 3, 4, 5

            for row in range(MAP_HEIGHT):
                if row < (MAP_HEIGHT - new_wall_height - 1):
                    self.map[row][col] = MapBlock.AIR
                elif row == (MAP_HEIGHT - new_wall_height - 1):
                    self.map[row][col] = MapBlock.HELIUM3
                elif row < MAP_HEIGHT:
                    self.map[row][col] = MapBlock.CRATERS

        self.rover_current_energy = 10
        self.rover_current_x = MAP_WIDTH - 1 - self.visual_range
        self.rover_current_y = MAP_HEIGHT - 1
        self.map_current_x = 0
        self.moving_distance = 0
        self.mineral_sampling = 0

        self.map[self.rover_current_y][self.rover_current_x - self.map_current_x] = MapBlock.ROVER

        obser = Observation(self.state_size)
        obser.rover_current_energy = self.rover_current_energy
        obser.rover_current_y = self.rover_current_y
        obser.visual_range = self.visual_range
        obser.rover_view_x = self.rover_view_x
        obser.map = self.map

        return obser

    def step(self, action):

        reward = 0
        done = False
        dead = False

        self.map[self.rover_current_y][self.rover_current_x - self.map_current_x] = MapBlock.VISIT

        if action == 1:
            if self.rover_current_energy > 0:
                self.rover_current_energy -= 1
                self.rover_current_y -= 1
            else :
                if self.rover_current_y < MAP_HEIGHT - 1:                
                    self.rover_current_y += 1

        else:
            if self.rover_current_y < MAP_HEIGHT - 1:
                self.rover_current_y += 1
    
        self.rover_current_x += 1
        
        current_block = self.map[self.rover_current_y][self.rover_current_x - self.map_current_x]

        new_block = MapBlock.ROVER

        is_get_helium3 = False

        if current_block == MapBlock.AIR:
            new_block = MapBlock.ROVER
        elif current_block == MapBlock.CRATERS:
            new_block = MapBlock.AIR
            done = True
            dead = True
        elif current_block == MapBlock.HELIUM3:
            new_block = MapBlock.ROVER
            self.mineral_sampling += 1
            self.rover_current_energy += 1
            is_get_helium3 = True

        self.map[self.rover_current_y][self.rover_current_x - self.map_current_x] = new_block

        new_wall_height = random.randrange(0, 5) # 1부터 6미만 숫자. 1, 2, 3, 4, 5

        for row in range(MAP_HEIGHT):
            for col in range(MAP_WIDTH-1):
                self.map[row][col] = self.map[row][col+1]

        for row in range(MAP_HEIGHT):
            if row < (MAP_HEIGHT - new_wall_height - 1):
                self.map[row][MAP_WIDTH-1] = MapBlock.AIR
            elif row == (MAP_HEIGHT - new_wall_height - 1):
                self.map[row][MAP_WIDTH-1] = MapBlock.HELIUM3
            elif row < MAP_HEIGHT:
                self.map[row][MAP_WIDTH-1] = MapBlock.CRATERS

        self.map_current_x += 1        

        self.moving_distance = max(self.moving_distance, abs(self.map_current_x))

        if done == True:
            reward = -10
        else:
            if is_get_helium3 == True:
                reward = 2
            else:
                reward = 1

        obser = Observation(self.state_size)
        obser.rover_current_energy = self.rover_current_energy
        obser.rover_current_y = self.rover_current_y
        obser.visual_range = self.visual_range
        obser.rover_view_x = self.rover_view_x
        obser.map = self.map
        
        return obser, reward, done

    def render(self):
        self.renderer.render(self.map, self.rover_current_energy, self.moving_distance, self.mineral_sampling)
        time.sleep(RENDERING_TIME_SLEEP)

    def render_log(self, str_log):
        self.renderer.logging(str_log)

class HumanAgent:
    
    def __init__(self, state_size, action_size, renderer):
        self.renderer = renderer

    def get_action(self, observation):

        ret = 0

        if renderer == None:
            return ret
            
        key = self.renderer.get_key()
        if key == curses.KEY_RIGHT:
            ret = 1

        return ret

    def save_model(self, ep):
        pass

def main_play(screen, env, agent):

    env.renderer.set_screen(screen)

    scores, episodes = [], []

    for ep in range(MAX_EPISODE):

        done = False
        score = 0

        observation = env.reset()

        while not done:
            
            env.render()
            
            action = agent.get_action(observation)

            next_observation, reward, done = env.step(action)
            
            score += reward
            observation = next_observation

        episodes.append(ep)
        scores.append(score)
        
        pylab.plot(episodes, scores, 'b')
        pylab.savefig("./lunar_rover_v101.png")

if __name__=='__main__':

    env = Environment()

    state_size = env.state_size
    action_size = env.action_size

    render = None

    if get_ipython() is None:   
        renderer = CursesRenderer()
    else:
        renderer = StandardRenderer()

    agent = HumanAgent(state_size, action_size, renderer)
    
    env.renderer = renderer

    if get_ipython() is None:   
        curses.wrapper(main_play, env, agent)
    else:
        main_play(None, env, agent)
