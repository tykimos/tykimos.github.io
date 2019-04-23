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

class DQNAgent:
    def __init__(self, state_size, action_size):

        self.load_model = False #True #False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999 #0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 5000 #1000 #학습 전에 메모리에 쌓아둘 리플레이 수

        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=10000) #2000

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./lunar_rover_dqn_v100.h5")

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def post_episode(self):
        self.update_target_model()

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, observation):

        state = observation.get_state()
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def train_sample(self, observation, action, reward, next_observation, done):        
        
        self.memory.append((observation.get_state(), action, reward, next_observation.get_state(), done))

        if len(self.memory) >= self.train_start:
            self.train_model()

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)
    
    def save_model(self, filename):
        self.model.save_weights(filename)

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

def main_train(screen, env, agent):

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

            agent.train_sample(observation, action, reward, next_observation, done)

            score += reward
            observation = next_observation

        agent.post_episode()

        episodes.append(ep)
        scores.append(score)
        
        pylab.plot(episodes, scores, 'b')
        pylab.savefig("./lunar_rover_v101.png")

        str_log = "ep:%5d, score:%5d, memlen:%5d, ep:%5.3f" % (ep, score, len(agent.memory), agent.epsilon)

        env.render_log(str)

        if ep % 1000 == 0:
            agent.save_model("./lunar_rover_dqn_v100.h5")

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
    #agent = ReinforceAgent(state_size, action_size)
    #agent = DQNAgent(state_size, action_size)
    #agent = A2CAgent(state_size, action_size)
    
    env.renderer = renderer

    if get_ipython() is None:   
        #curses.wrapper(main_train, renderer, env, agent)
        curses.wrapper(main_play, env, agent)
    else:
        #main_train(None, renderer, env, agent)
        main_play(None, env, agent)