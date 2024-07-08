import numpy as np
import os, sys
import numpy as np
from gym import spaces
import optparse
# from inference import model
import torch
from classifier import model


maxSteps = 200

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary

def getDistance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def safetyCheck(state, action):
    behindVeh = state[0]
    egoidx = int((state.shape[0] - 1)/2)
    egoVeh = state[egoidx]
    frontveh = state[egoidx + 1]
    nexPosition = action[0]/2 + egoVeh[2] + egoVeh[0] 
    moveSafetyFactor = 0.90
    laneChangeSafetyFactor = 0.95
    # Lanechange safet ycheck
    if action[1] > 0 and egoVeh[1] < -14.4:
        if (nexPosition > 1/laneChangeSafetyFactor*(behindVeh[0] + behindVeh[2]) or behindVeh[1] > -14.4) and (nexPosition < laneChangeSafetyFactor*(frontveh[0] + frontveh[2])or frontveh[1] > -14.4):
            action[1] = action[1]
        else:
            action[1] = 0
            action[0] = -1
            print('lane change action suppresed')
        return action
    # Move safety check
    elif egoVeh[1] == frontveh[1] and nexPosition > (frontveh[0] + frontveh[2])*moveSafetyFactor:
        action[0] = -1
        print('changing acceleration')
    
    return action



def getVehList(radius):
    egoPos = traci.vehicle.getPosition('t_0')
    egoVel = traci.vehicle.getSpeed('t_0')
    if egoPos[0] < 0:
        raise Exception()
    vehListInfo = [list(egoPos) + [egoVel]]
    for vehID in traci.vehicle.getIDList():
        if vehID == 't_0':
            continue
        vehPos = traci.vehicle.getPosition(vehID)
        if getDistance(egoPos, vehPos) <= radius:
            vehVel = traci.vehicle.getSpeed(vehID)  
            vehListInfo.append(list(vehPos) + [vehVel])
    vehListInfo = [vehListInfo[0]] + sorted(vehListInfo[1:], key= lambda x: (x[0], x[1]))
    return vehListInfo

def getState(radius, size = 99):
    vehListInfo = getVehList(radius)
    egoStartIdx = 0
    if len(vehListInfo) == 1:
        vehListInfo = np.append(np.append(np.zeros((int((size - 1)/2), 3)), \
                                          vehListInfo), np.zeros((int((size - 1)/2), 3)))
        return vehListInfo
    
    for idx, val in enumerate(vehListInfo[1:]):
        if val[0] > vehListInfo[egoStartIdx][0]:
            greaterStartIdx = idx + 1
            break
    else:
        greaterStartIdx = len(vehListInfo)

    lessList = vehListInfo[1:greaterStartIdx]
    lessList = sorted(lessList, key= lambda x: (-x[1], x[0]), reverse= True)
    greaterList = vehListInfo[greaterStartIdx:] if greaterStartIdx != len(vehListInfo) else []
    greaterList = sorted(greaterList, key= lambda x: (x[1], x[0]))

    if lessList == []:
        vehListInfo = np.append(np.zeros((int((size - 1)/2), 3)), \
                            [vehListInfo[0]] + greaterList, axis = 0)
    elif greaterList == []:
        vehListInfo = np.append(np.append(lessList, \
                        np.zeros((int((size - 1)/2) - len(lessList), 3)), axis = 0), \
                        [vehListInfo[0]], axis = 0)
    else:
        vehListInfo = np.append(np.append(lessList, \
                                np.zeros((int((size - 1)/2) - len(lessList), 3)), axis = 0), \
                                [vehListInfo[0]] + greaterList, axis = 0)
        
    vehListInfo = np.pad(vehListInfo, ((0, size - len(vehListInfo)), (0, 0)), 'constant', constant_values = 0)
    inputs = np.ndarray.flatten(np.array(vehListInfo.copy())).astype(np.float32)

    with torch.no_grad():
        drivingstyle = model(torch.from_numpy(inputs)).numpy()
    drivingstyle = np.rint(drivingstyle).astype(int)
    vehListInfo = np.column_stack((vehListInfo, drivingstyle))
    return vehListInfo


def getReward(vehListInfo, action, laneID):

    r1 = -0.1*np.abs(action[0])  -0.1
    r2 = 0

    if 't_0' in traci.simulation.getCollidingVehiclesIDList():
        if laneID == 'E3_0' and action[1] > 0:
            r2 -= 200
        else:
            r1 -= 200
        return [0.1*r1, 0.1*r2]
    
    elif 't_0' in traci.simulation.getArrivedIDList(): 
        r1 += 200
        return [0.1*r1, 0.1*r2]
    

    if traci.simulation.getEmergencyStoppingVehiclesIDList():
        print('emeregency stop')
        r2 -= 200
        return [0.1*r1, 0.1*r2]
    
    if traci.simulation.getTime() > maxSteps:
        print('max episode length')
        r1 -= 200

    if laneID == 'E3_0' and action[1] > 0:
        r2 += 200

    return [0.1*r1, 0.1*r2]


class Merging():
    def __init__(self, options, seed, radius = 50, render_mode= None, d = 1):
    
        self.done = False
        self.observation = 0
        self.reward = 0
        self.options = options
        self.state = []
        self.ego_inserted = False
        self.seed = seed
        self.radius = radius
        self.size = 31
        self.laneID = ''
        self.d = d
        self.delay_counter = 0


        if self.options.nogui:
            self.sumoBinary = checkBinary('sumo')
        else:
            self.sumoBinary = checkBinary('sumo-gui')

        low_space = np.tile(np.float32([-1000, -1000, 0, 0]), reps= self.size)
        high_space = np.tile(np.float32([1000, 1000, 50, 2]), reps= self.size)
        self.observation_space = spaces.Box(low = low_space, high = high_space, dtype=np.float32)

        self.action_space = spaces.Box(low = -1,  high = 1, shape= (2,))


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


    def reset(self, options=None):
        # super().reset()
        try:
            # traci.start([self.sumoBinary, "-c", "mergingP.sumocfg",
            #             "--tripinfo-output", "tripinfo.xml",  "--no-step-log", \
            #             "--random", "--step-length", "0.5", "--collision.check-junctions"])
            traci.start([self.sumoBinary, "-c", "mergingP.sumocfg",
                        "--tripinfo-output", "tripinfo.xml",  "--no-step-log", \
                        "--random", "--collision.check-junctions", "--collision.action", "remove"])
        except:
            traci.close()
            # traci.start([self.sumoBinary, "-c", "mergingP.sumocfg",
            #             "--tripinfo-output", "tripinfo.xml",  "--no-step-log", \
            #             "--random", "--step-length", "0.5", "--collision.check-junctions"])
            traci.start([self.sumoBinary, "-c", "mergingP.sumocfg",
                        "--tripinfo-output", "tripinfo.xml",  "--no-step-log", \
                        "--random", "--collision.check-junctions", "--collision.action", "remove"])
        
        self.done = False
        self.ego_inserted = False
        self.counter = 1
        while not self.ego_inserted:
            traci.simulationStep()
            if 't_0' in traci.simulation.getDepartedIDList():
                self.ego_inserted = True
                traci.vehicle.setSpeedMode("t_0", 32)
                traci.vehicle.setLaneChangeMode('t_0', 0b000000000000)
                self.state = getState(self.radius, self.size)
                # print('ego inserted')
                

            for vehicle_name in traci.simulation.getDepartedIDList():
                if 'f_1' in vehicle_name:
                    traci.vehicle.setTau(vehicle_name, np.random.uniform(0.1, 0.7))
                    traci.vehicle.setMaxSpeed(vehicle_name, np.random.uniform(10, 13))
                    traci.vehicle.setSpeedMode(vehicle_name, 32)
                if 'f_2' in vehicle_name:
                    traci.vehicle.setTau(vehicle_name, np.random.normal((0.6 + 1.8)/2, 0.1))
                    traci.vehicle.setMaxSpeed(vehicle_name, np.random.uniform(8, 11))
        
        self.observation_history = np.zeros((self.d, self.state.shape[0], self.state.shape[1]))
        self.observation_history[0] = self.state
        observation = np.ndarray.flatten(self.state)
        return observation
        
        

    def step(self, action):
        info = False
        self.delay_counter = (self.delay_counter + 1)%self.d
        self.observation_history[self.delay_counter] = self.state
        return_idx = max(0, self.counter - self.d)%self.d
        action = safetyCheck(self.observation_history[return_idx], action)
        
        if not self.done:
            if traci.vehicle.getLaneID('t_0') == 'E3_0':
                self.laneID = 'E3_0'
                if action[1] > 0:
                    traci.vehicle.changeLane('t_0', 1, 0)
            else:
                self.laneID = ''

            traci.vehicle.setAcceleration(vehID='t_0',duration= 1, acceleration= action[0])
            for vehicle_name in traci.simulation.getDepartedIDList():
                if 'f_1' in vehicle_name:
                    traci.vehicle.setTau(vehicle_name, np.random.uniform(0.1, 0.7))
                    traci.vehicle.setMaxSpeed(vehicle_name, np.random.uniform(10, 13))
                    traci.vehicle.setSpeedMode(vehicle_name, 32)
                if 'f_2' in vehicle_name:
                    traci.vehicle.setTau(vehicle_name, np.random.normal((0.6 + 1.8)/2, 0.1))
                    traci.vehicle.setMaxSpeed(vehicle_name, np.random.uniform(8, 11))
            
            self.counter += 1
            traci.simulationStep()

            if 't_0' in traci.simulation.getCollidingVehiclesIDList():
                info = True
                self.done = True
                observationArray = self.state
            elif 't_0' in traci.simulation.getArrivedIDList() or \
                traci.simulation.getTime() > maxSteps:
                self.done = True
                observationArray = self.state
            elif  traci.simulation.getTime() > maxSteps:
                print('Terminating episode since it exceeded maximum time.')
                info = True
                self.done = True
                observationArray = getState(self.radius, self.size)
                self.state = observationArray
            elif 't_0' in traci.simulation.getEmergencyStoppingVehiclesIDList():
                self.done = True
                info = True
                observationArray = getState(self.radius, self.size)
                self.state = observationArray
            else:
                observationArray = getState(self.radius, self.size)
                self.state = observationArray

        self.reward = getReward(observationArray, action, self.laneID)
        
        if self.done:
            traci.close()
        
        return np.ndarray.flatten(self.observation_history[return_idx]), self.reward, self.done, {'message': info, 'lane': self.laneID, 'action': action}
    

    def render(self):
        print('nothing can be rendered')

