import curses
import time
import sys
import numpy as np
from random import randint
import random
from enum import Enum, IntEnum

from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from keras.models import Sequential
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from keras import backend as K
import tensorflow as tf
import numpy as np
import random

MAP_WIDTH = 16
MAP_HEIGHT = 16
ROVER_MAX_ENERGY = 40
MAX_EPISODE = 500000

class MapBlock(IntEnum):
    AIR = 0
    VISIT = 1
    GROUND = 2
    CRATERS = 3
    HELIUM3 = 4
    ROVER = 5

MAP_BLOCKS_ICONS = {
    0: '..',
    1: '  ',               
    2: '▓▓',
    3: '░░',
    4: '◆.',
    5: '▞▚',
}

class Observation:

    def __init__(self):
        #self.rover_current_energy = 0
        #self.rover_current_x = 0
        #self.rover_current_y = 0
        #self.rover_direction = 0
        self.map = None

class Environment:
    def __init__(self, screen):

        self.screen = screen

        self.screen.timeout(0)
        self.map = None
        self.action_size = 2

        self.reset()

    def reset(self):
            
        self.map = np.zeros((MAP_HEIGHT, MAP_WIDTH))
        self.map[MAP_HEIGHT-1][:] = MapBlock.GROUND
        self.map[MAP_HEIGHT-2][:] = MapBlock.HELIUM3

        self.rover_current_energy = 5
        self.rover_current_x = int(MAP_WIDTH / 2)
        self.rover_current_y = MAP_HEIGHT - 2
        self.map_current_x = 0
        self.moving_distance = 0
        self.mineral_sampling = 0

        self.map[self.rover_current_y][self.rover_current_x - self.map_current_x] = MapBlock.ROVER

        observation = self.get_observation()

        return observation

    def get_observation(self):
        
        observation = Observation()
        observation.rover_current_energy = self.rover_current_energy
        observation.rover_current_x = self.rover_current_x
        observation.rover_current_y = self.rover_current_y
        observation.map = self.map

        return observation

    def step(self, action):

        reward = 0
        info = {'lives': False}
        done = False
        dead = False

        self.map[self.rover_current_y][self.rover_current_x - self.map_current_x] = MapBlock.VISIT

        if action == 1:
            if self.rover_current_energy > 0:
                self.rover_current_energy -= 1
                self.rover_current_y -= 1
            else :
                if self.rover_current_y < MAP_HEIGHT - 2:                
                    self.rover_current_y += 1

        else:
            if self.rover_current_y < MAP_HEIGHT - 2:
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

        new_wall_height = random.randrange(1, 6) # 1부터 6미만 숫자. 1, 2, 3, 4, 5

        for row in range(MAP_HEIGHT):
            for col in range(MAP_WIDTH-1):
                self.map[row][col] = self.map[row][col+1]

        for row in range(MAP_HEIGHT):
            if row < (MAP_HEIGHT - new_wall_height - 1):
                self.map[row][MAP_WIDTH-1] = MapBlock.AIR
            elif row == (MAP_HEIGHT - new_wall_height - 1):
                self.map[row][MAP_WIDTH-1] = MapBlock.HELIUM3
            elif row < MAP_HEIGHT - 1:
                self.map[row][MAP_WIDTH-1] = MapBlock.CRATERS
            else:
                self.map[row][MAP_WIDTH-1] = MapBlock.GROUND

        self.map_current_x += 1        

        observation = self.get_observation()

        self.moving_distance = max(self.moving_distance, abs(self.map_current_x))

        if done == True:
            reward = -1
        else:
            if is_get_helium3 == True:
                reward = 1 #self.moving_distance + self.mineral_sampling
            else:
                reward = 0

        return observation, reward, done, info

    def render(self):
        
        for row in range(MAP_HEIGHT):
            row_string = ''
            for col in range(MAP_WIDTH):
                row_string += MAP_BLOCKS_ICONS[self.map[row][col]]
            self.screen.addstr(row, 0, row_string)

        self.screen.addstr(0, 0, '%4d♥' % self.rover_current_energy)
        self.screen.addstr(0, 5, '%4dm' % self.moving_distance)
        self.screen.addstr(0, 10, '%4d◆' % self.mineral_sampling)
        self.screen.addstr(16, 0, '')    
        self.screen.refresh()

        #time.sleep(0.1)
        #time.sleep(0.3)
        #time.sleep(1.0)

class HumanAgent:
    def __init__(self, screen):
        self.screen = screen

    def write_summary(self, episode, score, global_step, step):
        pass

    def update_avg_q_max(self, history):
        pass

    def train(self, history, action, reward, next_history, dead, global_step):        
        pass
        
    def get_action(self, observation):

        ret = 0
        key = self.screen.getch()
        if key == curses.KEY_RIGHT:
            ret = 1

        return ret

    def save_model(self, ep):
        pass

class DQNAgent:
    def __init__(self, state_height, state_width, action_size):
        self.render = False
        self.load_model = False
        # 상태와 행동의 크기 정의
        self.state_size = (state_height, state_width, 1)
        self.action_size = action_size
        # DQN 하이퍼파라미터
        #self.epsilon = 0#0.5
        #self.epsilon_start, self.epsilon_end = 0, 0 #0.5, 0.1

        self.epsilon = 0.5
        self.epsilon_start = 0.5
        self.epsilon_end = 0.1

        self.exploration_steps = 100000. #1000000. 
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        self.batch_size = 32
        self.train_start = 5000 #50000 메모리가 이만큼 쌓이면 학습 시작
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        # 리플레이 메모리, 최대 크기 400000
        self.memory = deque(maxlen=400000)
        self.no_op_steps = 30
        # 모델과 타겟모델을 생성하고 타겟모델 초기화
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()

        # 텐서보드 설정
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'lunar_rover_dqn', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("./lunar_rover_dqn.h5")

    # Huber Loss를 이용하기 위해 최적화 함수를 직접 정의
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        prediction = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    def train(self, history, action, reward, next_history, dead, global_step):

        # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
        self.memory.append((history, action, reward, next_history, dead))

        if len(self.memory) >= self.train_start:
            self.train_model()

        # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
        if global_step % self.update_target_rate == 0:
            self.update_target_model()


    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(64, (8, 8), strides=(1, 1), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(1, 1), activation='relu'))
        #model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model

    def update_avg_q_max(self, history):
        self.avg_q_max += np.amax(self.model.predict(np.float32(history / 5.))[0])

    # 타겟 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, history):
        history = np.float32(history / 5.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])


    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 5.)
            next_history[i] = np.float32(mini_batch[i][3] / 5.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        target_value = self.target_model.predict(next_history)

        for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * \
                                        np.amax(target_value[i])

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]

    def write_summary(self, episode, score, global_step, step):
        
        # 각 에피소드 당 학습 정보를 기록
        if global_step > self.train_start:
            stats = [score, self.avg_q_max / float(step), step,
                        self.avg_loss / float(step)]
            for i in range(len(stats)):
                self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]: float(stats[i])})
            summary_str = self.sess.run(self.summary_op)
            self.summary_writer.add_summary(summary_str, episode + 1)

        print("ep:%5d scr:%5d mem:%5d eps:%6.4f step:%5d q:%6.4f loss:%6.4f" %
        (episode, score, len(self.memory), self.epsilon, global_step, self.avg_q_max / float(step), self.avg_loss / float(step)))

        self.avg_q_max = 0
        self.avg_loss = 0


    # 각 에피소드 당 학습 정보를 기록
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def save_model(self, ep):

        # 1000 에피소드마다 모델 저장
        if ep % 1000 == 0:
            self.model.save_weights("./lunar_rover_dqn.h5")            


def main(screen):

    env = Environment(screen)
    action_size = env.action_size

    #agent = HumanAgent(screen)
    
    # DQN 에이전트 생성
    agent = DQNAgent(MAP_HEIGHT, MAP_WIDTH, action_size)

    scores, episodes, global_step = [], [], 0

    for ep in range(MAX_EPISODE):

        step = 0
        score = 0
        observation = env.reset()

        #for _ in range(random.randint(1, agent.no_op_steps)):
        #    observation, _, _, _ = env.step(1)

        done = False

        state = observation.map
        #history = np.stack((state, state, state, state), axis=2)
        history = np.reshape(state, (1, MAP_HEIGHT, MAP_WIDTH, 1))

        while not done:

            env.render()
            
            global_step += 1
            step += 1

             # 현재 관측으로 행동을 선택
            action = agent.get_action(history)

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            observation, reward, done, info = env.step(action)

            # 각 타임스텝마다 상태 전처리
            next_state = observation.map
            next_history = np.reshape(state, (1, MAP_HEIGHT, MAP_WIDTH, 1))

            agent.update_avg_q_max(history)

            reward = np.clip(reward, -1., 1.)

            agent.train(history, action, reward, next_history, done, global_step)

            score += reward
                
        agent.write_summary(ep, score, global_step, step)        
        agent.save_model(ep)

if __name__=='__main__':
    curses.wrapper(main)