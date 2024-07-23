import numpy as np
import torch
from agents.classifier import CNNClassifier

def getDistance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def flatten(xss):
    flat_list = []
    for xs in xss:
        for x in xs:
            flat_list.append(x)
    return flat_list

def set_env(cfg):
    if cfg.mode == 'SLSC':
        from merging3 import Merging
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
    return env

def predict_driving_style(non_ego_state):

    input_size = 90 
    output_size = 90
    model = CNNClassifier(input_size, output_size)
    model.eval()
    model.load_state_dict(torch.load('slagent.pth'))
    non_ego_state = np.array(non_ego_state).astype(np.float32)
    with torch.no_grad():
        output = model(torch.from_numpy(non_ego_state)).numpy()
        output = np.rint(output).astype(int)
    return output



class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
     
def eval_policy(policy, eval_env, seed, eval_episodes=10):

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
            
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward[0]

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward