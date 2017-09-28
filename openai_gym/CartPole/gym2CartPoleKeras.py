# -*- coding: utf-8 -*-
from math import *
import random
import gym
from gym import wrappers,upload
import numpy as np
from collections import deque
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
import os
# setting
EPISODES = 3000 #test plot epsilon (0.9996**x)*0.5*(1.0+cos(2*pi*x*7/10000)) from 0 to 10000

class DQNAgent:
    def __init__(self, state_size, action_size, EPISODES):
        self.state_size     = state_size
        self.action_size    = action_size
        self.EPISODES       = EPISODES
        self.memory         = deque(maxlen=self.EPISODES)
        self.gamma          = 0.99    # discount rate
        self.epsilon        = 1.0
        self.epsilon_decay  = 0.999
        self.learning_rate  = 0.001
        self.model_backup   = 'CartPole-v1Model.h5'
        self.model          = self._build_model()
        self.mini_epoch     = 3
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    def learn(self, batch_size):
        if len(agent.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma*np.amax(self.model.predict(next_state)[0]) #self.Q_rem = (1.-self.alpha)*self.Q_rem + self.alpha*(reward + self.gamma*np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
    def decay_epsilon(self, ep):
        self.epsilon = (self.epsilon_decay**ep)*0.5*(1.+cos(2.*pi*ep*self.mini_epoch/self.EPISODES))
    def stop_random(self):
        self.epsilon = 0.
    def save_model(self):
        model.save(self.model_backup)
    def load_model(self):
        return self.model_backup
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    #env = wrappers.Monitor(env, os.getcwd()+'/record')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, EPISODES)
    done = False
    batch_size = 64
    f_report=open('Report_CartPole-v1.olo','w') #### open report file
    for ep in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        push=0.
        done=False
        while not done:
            #env.render()
            push+=1.
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
        if ep % EPISODES/200. ==0:
            f_report.write('%i %f %f\n'%(ep, agent.epsilon, push))
        print("episode: {}/{}, score: {}, e: {:.2}".format(ep, EPISODES, push, agent.epsilon))
        agent.learn(batch_size)
        agent.decay_epsilon(ep)
        #if push == 500:
        #    agent.stop_random()
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
    f_report.close()
    env.close()
    #gym.upload(os.getcwd()+'/record', api_key='sk_AULorX5sTLmkSPQsX7PA7w')
