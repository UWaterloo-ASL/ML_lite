#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 08:46:49 2018

@author: daiwei.lin
"""


from Environment.LASROMEnv import LASROMEnv
from LASAgent.LASBaselineAgent import LASBaselineAgent
import numpy as np
from gym import spaces

import logging

# from Learning import Learning

class Env_Example():
    """
    This class represents LAS system.
    """
    def __init__(self, action_dimension, sensors_dimension):
        """
        Create observation and action space:
            observation space: all observation values are within range [0,1]
                # IRs shared by all agents:
                    self.observation_space (gym.spaces.Box): observation space shared by all agents
            action space: all action values are within range [-1,1]
                    self.para_action_space (gym.spaces.Box): action space
        """
        obs_max = np.array([1.] * sensors_dimension)
        obs_min = np.array([0.]*sensors_dimension)
        self.observation_space = spaces.Box(obs_min, obs_max, dtype = np.float32)

        para_act_max = np.array([1] * action_dimension)
        para_act_min = np.array([-1] * action_dimension)
        self.action_space = spaces.Box(para_act_max, para_act_min, dtype=np.float32)

    def reset(self):
        """
        Reset environment and return an observation.
        Here returned observation is a random sample in the observation space
        """

        return self.observation_space.sample()

    def step(self, action):
        """
        Take one time step using given action.
        Return observation, done, info
        """
        return self.observation_space.sample(), 0, True, ''


class ML_LAS_Interface():
    """
    This class is the interface between the ML and LAS system
    """
    def __init__(self, env):
        self.env = env
        self.observation = env.reset()
        self.action = env.action_space.sample()

    def get_observation(self):
        return self.observation

    def reset(self):
        self.env.reset()

    def take_action(self, action):
        self.action = action
        # print("action :", action)
        observation, reward, done, info = self.env.step(action)
        self.observation = observation


if __name__ == '__main__':
    #================#
    # initialization #
    #================#
    logger = logging.getLogger(__name__)
    # V-REP simulator
    ROMenv = LASROMEnv(IP='127.0.0.1',
                       Port=19997,
                       reward_function_type='ir')
    # Example environment
    # env = Env_Example(action_dimension=5, sensors_dimension=10)
    print("env Initialized")
    interface = ML_LAS_Interface(ROMenv)
    print("interface Initialized")

    # Constants
    agent_name = 'LAS_Baseline_Agent'
    x_order_sensor_reading = 2
    load_pretrained_agent_flag = False

    agent = LASBaselineAgent(agent_name,
                             interface.env.observation_space.shape[0],
                             interface.env.action_space.shape[0],
                             num_observation=x_order_sensor_reading,
                             load_pretrained_agent_flag=load_pretrained_agent_flag)
    print("Agent's observation dimension = {}".format(agent.baseline_agent.observation_space.shape[0]))

    #===========#
    # Main loop #
    #===========#
    # Run 1000 steps
    interface.reset()
    for _ in range(1000):
        observation = interface.get_observation()
        # print(observation)
        take_action_flag, action = agent.feed_observation(observation)
        if take_action_flag == True:
                interface.take_action(action)

    print("Training complete")
    # Save learned model
    # logger.info('{}: Interaction is done. Saving learned models...'.format(agent.name))
    # agent.stop()
    # logger.info('{}: Saving learned models done.'.format(agent.name))


