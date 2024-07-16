"""
Author: Amin Tabrizian
Date: July 2024

Code adopted from: https://github.com/nikhilbarhate99/PPO-PyTorch

Description:
This code implements training of LK and LC agents for on-ramp merging on SUMO 
simulator.

"""
import os
import sys
import glob
import time
from datetime import datetime
import argparse
import torch
import numpy as np
import hydra
from omegaconf import DictConfig
from merging import Merging 


def train(cfg):
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    env = Merging(options= cfg, seed= cfg.random_seed) 
    env_name = 'merging'

    ################# training procedure ###############

    
    time_step = 0
    i_episode = 0
    laneID = ''
    epoch = 100
    x = []
    y = []
    # training loop
    for i in range(epoch):
        print(i)
        state = env.reset()
        done = False
        while not done:
            x_single = []
            y_single = []
            action0 = np.random.uniform(-1, 1)
            action1 = np.random.choice([0, 1])
            action = [action0, action1]
            observation, reward, done, _ = env.step(action)
            for i in range(int(len(observation)/6)):
                x_single += observation[6*i:6*i + 3] 
                y_single += observation[6*i + 3:6*i + 6]
            x.append(x_single)
            y.append(y_single)

    np.save('x2.npy', x)
    np.save('y2.npy', y)
        
            





@hydra.main(config_path="../config", config_name="config", version_base="1.1")
def main(cfg):
    train(cfg)

if __name__ == '__main__':
    main()

    
    