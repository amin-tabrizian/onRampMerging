"""
Author: Amin Tabrizian
Date: July 2024

Code adopted from: https://github.com/nikhilbarhate99/PPO-PyTorch

Description:
This code implements training of LK and LC agents for on-ramp merging on SUMO 
simulator.

"""
import os
import glob
import time
from datetime import datetime
import argparse
import torch
from agents.ppo import PPO
import numpy as np
from agents.dqn import DQN
import hydra
from omegaconf import DictConfig
from utils import set_env


def test(cfg):
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    env = set_env(cfg)
    log_freq = eval(cfg.log_freq)
    env_name = 'merging'

    # state and action space dimension
    state_dim = env.observation_space.shape[0]
    ppo_action_dim = 1
    dqn_action_dim = 2


    print("===================================================================")

    ################# training procedure ################

    # initialize the PPO and DQN agent
    ppo_agent = PPO(state_dim, ppo_action_dim, cfg.PPO.lr_actor, 
                    cfg.PPO.lr_critic, cfg.gamma, cfg.PPO.K_epochs, 
                    cfg.PPO.eps_clip, cfg.has_continuous_action_space, 
                    1e-10)
    dqn = DQN(state_dim, dqn_action_dim, cfg.DQN.lr, 1e-10, 
              cfg.DQN.target_iter_replace, cfg.DQN.batch_size, cfg.gamma)
    run_num_pretrained = cfg.random_seed      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    if cfg.mode == 'SLSCPD':
        load_mode = 'SLSC'
        load_delay = ''
    elif cfg.mode == 'PD':
        load_mode = 'Plain'
        load_delay = ''
    else:
        load_mode = cfg.mode
        load_delay = cfg.delay
    load_checkpoint_DQN = directory + "DQN_{}_{}.pth".format(load_mode + str(load_delay), 
                                                             run_num_pretrained)
    load_checkpoint_PPO = directory + "PPO_{}_{}.pth".format(load_mode + str(load_delay), 
                                                             run_num_pretrained)
    dqn.load(load_checkpoint_DQN)
    ppo_agent.load(load_checkpoint_PPO)
    print('Loading models from: ', load_checkpoint_DQN, ' and ', load_checkpoint_PPO)
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("===================================================================")


    
    time_step = 0
    i_episode = 0

    laneID = ''
    colided = 0
    # training loop
    total_ppo_reward = 0
    total_dqn_reward = 0
    while i_episode <= (cfg.total_test_episodes):

        state = env.reset()
        current_ppo_ep_reward = 0
        current_dqn_ep_reward = 0
        done = False
        while not done:

            # select action with policy
            ppo_action = ppo_agent.select_action(state)
            
            if laneID == 'E3_0':
                dqn_action = dqn.select_action(state)
                # print('dqn takes action')
            else:
                dqn_action = 0
            
            
            action = [float(ppo_action[0]), dqn_action]
            observation, reward, done, info = env.step(action)
            ppo_reward = reward[0]
            dqn_reward = reward[1]
           
            laneID = info['lane']
            # print(reward)
            # print(state.shape)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(ppo_reward)
            ppo_agent.buffer.is_terminals.append(done)
            action = info['action']
            ppo_action = action[0]
            dqn_action = action[1]

            
            
            time_step +=1
            current_ppo_ep_reward += ppo_reward
            current_dqn_ep_reward += dqn_reward
            total_ppo_reward += ppo_reward
            total_dqn_reward += dqn_reward



            
            # break; if the episode is over
            if done:
                if info['message']:
                    colided += 1
                break
            state = observation
        print('PPO reward is {} and DQN reward is {}'.format(current_ppo_ep_reward,
                                                            current_dqn_ep_reward))
        i_episode += 1
    print('Number of collisions in {} is {} for mode {} and random seed {}: ' \
          .format(cfg.total_test_episodes, colided, cfg.mode, cfg.random_seed))
    print('Total reward for PPO is {} and for DQN is {}' \
          .format(total_ppo_reward/cfg.total_test_episodes, 
                  total_dqn_reward/cfg.total_test_episodes))
    print('Collision rate is: {}%' \
          .format(round(100*colided/cfg.total_test_episodes, 3)))


    # print total training time
    print("===================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("===================================================================")



@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):
    test(cfg)

if __name__ == '__main__':
    main()

    
    