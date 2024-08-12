import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

MEMORY_CAPACITY = 20000

class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(128, n_actions)
        self.softmax = nn.Softmax(dim=-1)
        self.out.weight.data.normal_(0, 0.1)   # initialization

        

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        # actions_value = self.softmax(x)
        actions_value = x
        # print(actions_value)
        return actions_value
    

class DQN(object):
    def __init__(self, n_states, n_actions, lr=0.001, epsilon=0.9, target_replace_iter= 100, batch_size = 128, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.epsilon = epsilon
        self.target_replace_iter = target_replace_iter
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, self.n_states * 2 + 2))     # initialize memory
        

        if(torch.cuda.is_available()): 
            self.device = torch.device('cuda:0') 
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(self.device)))
        else:
            print("Device set to : cpu")
            self.device = torch.device('cpu')
        self.device = torch.device('cpu')
        self.eval_net, self.target_net = Net(n_states, n_actions), Net(n_states, n_actions)

        self.eval_net.to(self.device)
        self.target_net.to(self.device)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()


        

    def select_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0)).to(self.device)
        # input only one sample
        if np.random.uniform() > self.epsilon:   # greedy
            actions_value = self.eval_net.forward(x).cpu()
            action = torch.max(actions_value, 1)[1].data.numpy()[0]     # return the argmax
            # print(action)
        else:   # random
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            # self.epsilon = max(self.epsilon*0.99, 0.1)
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :self.n_states]))
        b_a = Variable(torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -self.n_states:]))

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0]   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save(self, checkpoint_path):
        torch.save(self.target_net.state_dict(), checkpoint_path)
    
    def load(self, checkpoint_path):
        self.eval_net.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.target_net.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))