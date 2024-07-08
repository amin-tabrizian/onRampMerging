import os
import glob
import time
from datetime import datetime
import argparse
import torch
from agents.ppo import PPO
import numpy as np
from agents.dqn import DQN

MEMORY_CAPACITY = 20000

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", action="store", type=str, dest="text_file", default="")
    parser.add_argument("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    # parser.add_argument("-v", "--version", action="store", type=str, dest="version", default="")
    # parser.add_argument("-k", action="store", type=int, dest="k", default=1)
    parser.add_argument("-t", action="store_true", dest="test", default=False)
    parser.add_argument("-s", "--seed", type= int, dest="seed", default= 0)
    parser.add_argument("--max_ep_len", action="store", type=int, dest="max_ep_len", default=2000)
    parser.add_argument("--action_std", action="store", type=float, dest="action_std", default=0.8)
    parser.add_argument("--action_std_decay_rate", action="store", type=float, dest="action_std_decay_rate", default=0.05)
    parser.add_argument("--min_action_std", action="store", type=float, dest="min_action_std", default=0.05)
    parser.add_argument("--action_std_decay_freq", action="store", type=int, dest="action_std_decay_freq", default=int(1.5e4))
    parser.add_argument("--update_timestep", action="store", type=int, dest="update_timestep", default=2e4)
    parser.add_argument("--K_epochs", action="store", type=int, dest="K_epochs", default=300)
    parser.add_argument("--eps_clip", action="store", type=float, dest="eps_clip", default=0.2)
    parser.add_argument("--gamma", action="store", type=float, dest="gamma", default=0.99)
    parser.add_argument("--lr_actor", action="store", type=float, dest="lr_actor", default=0.0001)
    parser.add_argument("--lr_critic", action="store", type=float, dest="lr_critic", default=0.003)
    parser.add_argument("--total_test_episodes", action="store", type=int, dest="total_test_episodes", default=990)
    parser.add_argument("--total_train_episodes", action="store", type=int, dest="total_train_episodes", default=5000)
    parser.add_argument("--mode", action="store", type=str, dest="mode", default="SLSC")
    parser.add_argument("--testSeed", action="store", type=int, dest="testSeed")
    parser.add_argument("--delay", action="store", type=int, dest="d")
    args = parser.parse_args()
    args = parser.parse_args()
    return args


options = get_options()
# parameters = read_parameters_from_file(options.text_file)



####### initialize environment hyperparameters ######


has_continuous_action_space = True  # continuous action space; else discrete

max_ep_len = options.max_ep_len                  # max timesteps in one episode
max_training_timesteps = int(3.6e5)   # break training loop if timeteps > max_training_timesteps
total_train_episodes = options.total_train_episodes
print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
save_model_freq = int(5e3)          # save model frequency (in num timesteps)

action_std = options.action_std                   # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = options.action_std_decay_rate        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = options.min_action_std               # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = options.action_std_decay_freq  # action_std decay frequency (in num timesteps)
hidden_size = 50
#####################################################

## Note : print/log frequencies should be > than max_ep_len

################ PPO hyperparameters ################
update_timestep = options.update_timestep      # update policy every n timesteps
K_epochs = options.K_epochs               # update policy for K epochs in one PPO update

eps_clip = options.eps_clip          # clip parameter for PPO
gamma = options.gamma            # discount factor

lr_actor = options.lr_actor       # learning rate for actor network
lr_critic = options.lr_critic       # learning rate for critic network

random_seed = options.seed        # set random seed if required (0 = no random seed)
#####################################################

if options.mode == 'SLSC':
    from merging3SLSC import Merging
    env = Merging(options= options, seed= random_seed)

elif options.mode == 'SL':
    from merging3SL import Merging
    env = Merging(options= options, seed= random_seed)

elif options.mode == 'SC':
    from merging3SC import Merging
    env = Merging(options= options, seed= random_seed)

elif options.mode == 'Plain':
    from merging3 import Merging
    env = Merging(options= options, seed= random_seed)

elif options.mode == 'SLSCD':
    from merging3SLSCD import Merging
    d = options.d
    env = Merging(options= options, seed= random_seed, d=d)
    
elif options.mode == 'SLSCD_R':
    from merging3SLSCD_R import Merging
    d = options.d
    env = Merging(options= options, seed= random_seed, d=d)
    options.mode = 'SLSC'
else:
    raise Exception("Wrong mode!")


env_name = 'merging'


# state space dimension
state_dim = env.observation_space.shape[0]
print(state_dim)

################## hyperparameters ##################



total_test_episodes = options.total_test_episodes    # total num of testing episodes




    


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

    
    