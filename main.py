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

MEMORY_CAPACITY = 20000
def set_env(cfg):
    if cfg.mode == 'SLSC':
        from merging3SLSC import Merging
        env = Merging(options= cfg, seed= cfg.random_seed)

    elif cfg.mode == 'SL':
        from merging3SL import Merging
        env = Merging(options= cfg, seed= cfg.random_seed)

    elif cfg.mode == 'SC':
        from merging3SC import Merging
        env = Merging(options= cfg, seed= cfg.random_seed)

    elif cfg.mode == 'Plain':
        from merging3 import Merging
        env = Merging(options= cfg, seed= cfg.random_seed)

    elif cfg.mode == 'SLSCD':
        from merging3SLSCD import Merging
        d = cfg.d
        env = Merging(options= cfg, seed= cfg.random_seed, d=d)
        
    elif cfg.mode == 'SLSCD_R':
        from merging3SLSCD_R import Merging
        d = cfg.d
        env = Merging(options= cfg, seed= cfg.random_seed, d=d)
        cfg.mode = 'SLSC'
    else:
        raise Exception("Wrong mode!")

env = set_env(cfg)
env_name = 'merging'
# state space dimension
state_dim = env.observation_space.shape[0]
print(state_dim)
    


################################### Training ###################################
def train():
    print("============================================================================================")
    print("training environment name : " + env_name)
    print("============================================================================================")

    
    if has_continuous_action_space:
        # action_dim = env.action_space.shape[0]
        action_dim = 1
    else:
        action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs_e"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = random_seed
    # current_num_files = next(os.walk(log_dir))[2]
    # run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_0.5_' + options.mode + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    # run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(options.mode, random_seed, run_num)
    checkpoint_path_dqn = directory + "DQN_{}_{}_{}.pth".format(options.mode, random_seed, run_num)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    dqn = DQN(state_dim, 2, lr=0.001, epsilon=0.1, target_replace_iter= 8000, batch_size = 128, gamma=0.99)
    run_num_pretrained = 7      #### set this to load a particular checkpoint num

    # directory = "PPO_preTrained" + '/' + env_name + '/'
    # load_checkpoint_DQN = directory + "DQN_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    # dqn.load(load_checkpoint_DQN)
    # # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward,reward2\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0
    log_running_reward2 = 0
    time_step = 0
    i_episode = 0
    best_reward = -float('inf')
    saving_reward = 0
    laneID = ''
    # training loop
    while time_step <= (max_training_timesteps):

        state = env.reset()
        current_ep_reward = 0
        ep_r = 0
        done = False
        while not done:

            # select action with policy
            action0 = ppo_agent.select_action(state)
            
            if laneID == 'E3_0':
                action1 = dqn.select_action(state)
                # print('dqn takes action')
            else:
                action1 = 0
            
            
            action = [float(action0[0]), action1]
            observation, reward, done, info = env.step(action)
            # print(reward)
            # print(state.shape)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward[0])
            ppo_agent.buffer.is_terminals.append(done)
            action = info['action']
            if laneID == 'E3_0':
                dqn.store_transition(state, action[1], reward[1], observation)
            
            laneID = info['lane']
            # print(laneID == )
            ep_r += reward[1]
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))


            time_step +=1
            current_ep_reward += reward[0]

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward2 = round(log_running_reward2 / log_running_episodes, 4)
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}, {}\n'.format(i_episode, time_step, log_avg_reward, log_avg_reward2))
                log_f.flush()

                log_running_reward = 0
                log_running_reward2 = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                # print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                if  saving_reward > best_reward:
                    best_reward = saving_reward
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at : " + checkpoint_path)
                    ppo_agent.save(checkpoint_path)
                    dqn.save(checkpoint_path_dqn)
                    print("model saved")
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print("--------------------------------------------------------------------------------------------")
                saving_reward = 0
            
            # break; if the episode is over
            if done:
                break
            state = observation

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_reward2 += ep_r
        saving_reward += log_running_reward + log_running_reward2
        log_running_episodes += 1

        i_episode += 1
        

    log_f.close()
    # env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")





#################################### Testing ###################################
def test():
    # action space dimension
    env_name = 'merging'
    print("training environment name : " + env_name)
    if has_continuous_action_space:
        action_dim = 1
    else:
        action_dim = env.action_space.n

    
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, 1e-10)
    dqn = DQN(state_dim, 2, lr=0.0001, epsilon=0, target_replace_iter= 100, batch_size = 128, gamma=0.99)
    # preTrained weights directory

    random_seed = options.testSeed             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = options.testSeed    #### set this to load a particular checkpoint num
    
    directory = "PPO_preTrained" + '/' + env_name + '/'

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format('0.5_SLSCD_1', random_seed, run_num_pretrained)
    checkpoint_path_dqn = directory + "DQN_{}_{}_{}.pth".format('0.5_SLSCD_1', random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)
    dqn.load(checkpoint_path_dqn)
    print("--------------------------------------------------------------------------------------------")
    laneID = ''
    test_running_reward = 0
    colided = 0
    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        ep_reward2 = 0
        state = env.reset()
        done = False
        while not done:
            action0 = ppo_agent.select_action(state)
            if laneID == 'E3_0':
                action1 = dqn.select_action(state)
                # print('dqn takes action')
            else:
                action1 = 0
            
            action = [float(action0[0]), action1]
            # print(action)
            # print(action)
            state, reward, done, info = env.step(action)
            laneID = info['lane']
            # print(reward[0], reward[1])
            ep_reward += reward[0]
            ep_reward2 += reward[1]

            if info['message']:
                colided += 1
            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0
        print('Collision rate {:.4f}%'.format(round(colided/total_test_episodes*100, 4)))


    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':
    options = get_options()
    testing = options.test
    if testing:
        test()
    else:
        train()

    
    