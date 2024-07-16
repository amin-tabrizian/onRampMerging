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

