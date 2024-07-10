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


def train(cfg):
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    env = set_env(cfg)
    log_freq = eval(cfg.log_freq)
    env_name = 'merging'

    # state and action space dimension
    state_dim = env.observation_space.shape[0]
    ppo_action_dim = 1
    dqn_action_dim = 2

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = cfg.random_seed

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + cfg.mode + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    ################### checkpointing ###################
    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(cfg.mode, 
                                                            cfg.random_seed,
                                                            run_num)
    checkpoint_path_dqn = directory + "DQN_{}_{}_{}.pth".format(cfg.mode, 
                                                                cfg.random_seed,
                                                                run_num)
    
    #####################################################

    print("===================================================================")

    ################# training procedure ################

    # initialize the PPO and DQN agent
    ppo_agent = PPO(state_dim, ppo_action_dim, cfg.PPO.lr_actor, 
                    cfg.PPO.lr_critic, cfg.gamma, cfg.PPO.K_epochs, 
                    cfg.PPO.eps_clip, cfg.has_continuous_action_space, 
                    cfg.PPO.action_std)
    dqn = DQN(state_dim, dqn_action_dim, cfg.DQN.lr, cfg.DQN.epsilon, 
              cfg.DQN.target_iter_replace, cfg.DQN.batch_size, cfg.gamma)
    run_num_pretrained = 7      #### set this to load a particular checkpoint num

    # directory = "PPO_preTrained" + '/' + env_name + '/'
    # load_checkpoint_DQN = directory + "DQN_{}_{}_{}.pth".format(env_name, cfg.random_seed, run_num_pretrained)
    # dqn.load(load_checkpoint_DQN)
    # # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("===================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,pporeward,dqnreward,averagereward\n')

    # logging variables
    log_ppo_running_reward = 0
    log_dqn_running_reward = 0
    log_running_episodes = 0
    
    time_step = 0
    i_episode = 0
    best_reward = -float('inf')
    saving_reward = 0
    laneID = ''
    # training loop
    while time_step <= (cfg.max_training_timesteps):

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
            
            if laneID == 'E3_0':
                dqn.store_transition(state, dqn_action,
                                      dqn_reward, observation)
            
            
            time_step +=1
            current_ppo_ep_reward += ppo_reward
            current_dqn_ep_reward += dqn_reward

            # update DQN agent
            if dqn.memory_counter > cfg.memory_capacity:
                dqn.learn()

            # update PPO agent
            if time_step % cfg.PPO.update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if (cfg.has_continuous_action_space and 
                time_step % cfg.PPO.action_std_decay_freq == 0):
                ppo_agent.decay_action_std(cfg.action_std_decay_rate, 
                                           cfg.min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward2 = round(log_dqn_running_reward / log_running_episodes, 4)
                log_avg_reward = round(log_ppo_running_reward / log_running_episodes, 4)
                log_avg_reward_avg = 1/2*(log_avg_reward + log_avg_reward2)

                log_f.write('{},{},{},{},{}\n'.format(i_episode, time_step, 
                                                      log_avg_reward, 
                                                      log_avg_reward2, 
                                                      log_avg_reward_avg))
                log_f.flush()

                log_ppo_running_reward = 0
                log_dqn_running_reward = 0
                log_running_episodes = 0

            # save model weights
            if time_step % cfg.save_model_freq  == 0:
                if  saving_reward > best_reward:
                    best_reward = saving_reward
                    print("---------------------------------------------------")
                    print("Saving model at : " + checkpoint_path)
                    ppo_agent.save(checkpoint_path)
                    dqn.save(checkpoint_path_dqn)
                    print("Model saved")
                    print("Elapsed Time  : ", 
                          datetime.now().replace(microsecond=0) - start_time)
                    print("---------------------------------------------------")
                saving_reward = 0
            
            # break; if the episode is over
            if done:
                break
            state = observation


        log_ppo_running_reward += current_ppo_ep_reward
        log_dqn_running_reward += current_dqn_ep_reward
        saving_reward += log_ppo_running_reward + log_dqn_running_reward
        log_running_episodes += 1

        i_episode += 1
        

    log_f.close()

    # print total training time
    print("===================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("===================================================================")



@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):
    train(cfg)

if __name__ == '__main__':
    main()

    
    