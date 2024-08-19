import numpy as np
import os, sys
import numpy as np
import optparse
# from inference import model
from gym import spaces
import torch
from utils import *


maxSteps = 200

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary


def safetyCheck(state, action):
    egoidx = int((len(state) - 1)/2) - 1
    egoVeh = state[egoidx:egoidx+3]
    frontVeh = state[egoidx + 3:egoidx + 6]
    behindVeh = state[egoidx - 6: egoidx - 3]
    nexPosition = action[0]/2 + egoVeh[2] + egoVeh[0] 
    moveSafetyFactor = 0.90
    laneChangeSafetyFactor = 0.95
    # Lanechange safet ycheck
    if action[1] > 0 and egoVeh[1] < -14.4:
        if (nexPosition > 1/laneChangeSafetyFactor*(behindVeh[0] + behindVeh[2]) or behindVeh[1] > -14.4) and (nexPosition < laneChangeSafetyFactor*(frontVeh[0] + frontVeh[2])or frontVeh[1] > -14.4):
            action[1] = action[1]
        else:
            action[1] = 0
            action[0] = -1
            print('lane change action suppresed')
        return action
    # Move safety check
    elif egoVeh[1] == frontVeh[1] and nexPosition > (frontVeh[0] + frontVeh[2])*moveSafetyFactor:
        action[0] = -1
        print('changing acceleration')
    return action

def getState(radius, size= 99, mode= 'Plain'):

    # Initialize behind and ahead vehicles list
    veh_behind_list = []
    veh_ahead_list = []
    lower_radius = 10
    veh_behind_number = 0
    veh_ahead_number = 0
    nonComList = []

    # Get ego vehicles' pos and vel
    egoPos = traci.vehicle.getPosition('t_0')
    egoVel = traci.vehicle.getSpeed('t_0')
    if egoPos[0] < 0:
        raise Exception()
    ego = [list(egoPos) + [egoVel]]

    # Get surrounding vehicles pos and vel
    for vehID in traci.vehicle.getIDList():
        if vehID == 't_0':
            continue
        vehPos = traci.vehicle.getPosition(vehID)


        # Add to the list if the vehicle is in radius
        if getDistance(egoPos, vehPos) <= radius:
            
            vehVel = traci.vehicle.getSpeed(vehID)
            vehList = list(vehPos) + [vehVel]
            
            if ('n_0' in vehID \
                and getDistance(egoPos, vehPos) <= lower_radius) \
                or ('n_0' not in vehID):
                if  vehPos < egoPos:
                    veh_behind_list.append(vehList)
                    veh_behind_number += 1
                else:
                    veh_ahead_list.append(vehList)
                    veh_ahead_number += 1

    # Sort behind and ahead vehicle lists. 
    veh_behind_list = sorted(veh_behind_list, 
                             key= lambda x: [-x[1], x[0]])
    veh_ahead_list = sorted(veh_ahead_list, 
                            key= lambda x: [-x[1], -x[0]],
                            reverse=True)
    raw_state_non_ego = flatten(veh_behind_list + veh_ahead_list)
    raw_state = flatten(veh_behind_list + ego + veh_ahead_list)

    # Zero padd
    behind_padding = list(np.zeros(3*(int((size - 1)/2) - veh_behind_number)))
    ahead_padding = list(np.zeros(3*(int((size - 1)/2) - veh_ahead_number)))
    

    if 'SL' in mode:
        padded_state = behind_padding + raw_state_non_ego + ahead_padding
        driving_style = list(1*predict_driving_style(padded_state))
        padded_state_driving_style = []

        for i in np.arange(0, int(len(padded_state)/3)):
            if i == 15:
                padded_state_driving_style += flatten(ego)
            padded_state_driving_style += padded_state[3*i:3*i+3]
            padded_state_driving_style += driving_style[3*i:3*i+3]
        padded_state = padded_state_driving_style
    else:
        padded_state = behind_padding + raw_state + ahead_padding
    return padded_state



def getReward(state, action, laneID):
    # ego_start_idx = int((len(state) - 1)/2) -1
    # ego_pos = state[ego_start_idx:ego_start_idx + 2]
    r1 = -0.01*np.abs(action[0]) - 1
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
    def __init__(self, options, seed, radius = 50, render_mode= None):
        mode = options.mode
        self.done = False
        self.observation = []
        self.reward = 0
        self.options = options
        self.d = options.delay
        self.state = []
        self.ego_inserted = False
        self.seed = seed
        self.radius = radius
        self.size = 31
        self.laneID = ''

        if self.options.nogui:
            self.sumoBinary = checkBinary('sumo')
        else:
            self.sumoBinary = checkBinary('sumo-gui')
        if  mode == 'SLSC' or mode == 'SL':
            low_space = np.tile(np.float32([-1000]), 
                                reps= (self.size - 1)*6 + 3)
            high_space = np.tile(np.float32([1000]), 
                                reps= (self.size - 1)*6 + 3)
            self.observation_space = spaces.Box(low = low_space, 
                                                high = high_space, 
                                                dtype=np.float32)
        elif mode == 'Plain' :
            low_space = np.tile(np.float32([-1000, -1000, 0]), reps= self.size)
            high_space = np.tile(np.float32([1000, 1000, 50]), reps= self.size)
            self.observation_space = spaces.Box(low = low_space, 
                                                high = high_space, 
                                                dtype=np.float32)
        elif 'D' in mode:
            low_space = np.tile(np.float32([-1000]), 
                                reps= (self.size - 1)*6 + 3)
            low_space = np.append(low_space, np.tile(np.float32([-1, -1]), 
                                reps=self.d))
            high_space = np.tile(np.float32([1000]), 
                                reps= (self.size - 1)*6 + 3)
            high_space = np.append(high_space, np.tile(np.float32([1, 1]), 
                                reps=self.d))
            self.observation_space = spaces.Box(low = low_space, 
                                                high = high_space, 
                                                dtype=np.float32)


        self.action_space = spaces.Box(low = -1,  high = 1, shape= (2,))


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


    def reset(self, options=None):
        HOME = '/home/amin/onRampMerging'
        SUMO = '/sumo_files/mergingP.sumocfg'
        # super().reset()
        try:
            # traci.start([self.sumoBinary, "-c", "mergingP.sumocfg",
            #             "--tripinfo-output", "tripinfo.xml",  "--no-step-log", \
            #             "--random", "--step-length", "0.1", "--collision.check-junctions"])
            traci.start([self.sumoBinary, "-c", HOME + SUMO,
                        "--tripinfo-output", "tripinfo.xml",  "--no-step-log", \
                        "--random", "--collision.check-junctions", "--collision.action", "remove"])
        except:
            traci.close()
            # traci.start([self.sumoBinary, "-c", "mergingP.sumocfg",
            #             "--tripinfo-output", "tripinfo.xml",  "--no-step-log", \
            #             "--random", "--step-length", "0.1", "--collision.check-junctions"])
            traci.start([self.sumoBinary, "-c", HOME + SUMO,
                        "--tripinfo-output", "tripinfo.xml",  "--no-step-log", \
                        "--random", "--collision.check-junctions", "--collision.action", "remove"])
        
        self.done = False
        self.ego_inserted = False
        self.counter = 0
        while not self.ego_inserted:
            traci.simulationStep()
            if 't_0' in traci.simulation.getDepartedIDList():
                self.ego_inserted = True
                traci.vehicle.setSpeedMode("t_0", 32)
                traci.vehicle.setLaneChangeMode('t_0', 0b000000000000)
                self.state = getState(self.radius, self.size, self.options.mode)
                # print('ego inserted')
                

            for vehicle_name in traci.simulation.getDepartedIDList():
                if 'f_1' in vehicle_name:
                    traci.vehicle.setTau(vehicle_name, np.random.uniform(0.1, 0.7))
                    traci.vehicle.setMaxSpeed(vehicle_name, np.random.uniform(10, 13))
                    traci.vehicle.setSpeedMode(vehicle_name, 32)
                if 'f_2' in vehicle_name:
                    traci.vehicle.setTau(vehicle_name, np.random.normal((0.6 + 1.8)/2, 0.1))
                    traci.vehicle.setMaxSpeed(vehicle_name, np.random.uniform(8, 11))
        if 'D' in self.options.mode:
            self.action_history =np.zeros((self.d, 2))
            self.observation_history = np.zeros((self.d, len(self.state)))
            self.delay_counter = 0
            self.observation_history[0] = self.state
            return np.append(self.state, np.ndarray.flatten(self.action_history))
        return self.state
        
        

    def step(self, action):
        info = False
        if self.options.mode == "SLSCD":
            self.delay_counter = (self.delay_counter + 1)%self.d
            self.observation_history[self.delay_counter] = self.state
            return_idx = max(0, self.counter - self.d)%self.d
            action = safetyCheck(self.observation_history[return_idx], action)
            self.action_history[self.delay_counter - 1] = action
        elif self.options.mode == "SLD":
            self.delay_counter = (self.delay_counter + 1)%self.d
            self.observation_history[self.delay_counter] = self.state
            return_idx = max(0, self.counter - self.d)%self.d
            self.action_history[self.delay_counter - 1] = action

        elif self.options.mode == "SLSC":
            action = safetyCheck(self.state, action)
        
        if not self.done:
            if traci.vehicle.getLaneID('t_0') == 'E3_0':
                self.laneID = 'E3_0'
                if action[1] > 0:
                    traci.vehicle.changeLane('t_0', 1, 0)
            else:
                self.laneID = ''

            traci.vehicle.setAcceleration(vehID='t_0',duration= 1, 
                                          acceleration= action[0])
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
            elif 't_0' in traci.simulation.getArrivedIDList():
                self.done = True
                observationArray = self.state
            elif  traci.simulation.getTime() > maxSteps:
                print('Terminating episode since it exceeded maximum time.')
                info = True
                self.done = True
                observationArray = getState(self.radius, self.size, self.options.mode)
                self.state = observationArray
            elif 't_0' in traci.simulation.getEmergencyStoppingVehiclesIDList():
                self.done = True
                info = True
                observationArray = getState(self.radius, self.size, self.options.mode)
                self.state = observationArray
            else:
                observationArray = getState(self.radius, self.size, self.options.mode)
                self.state = observationArray

        self.reward = getReward(observationArray, action, self.laneID)
        
        if self.done:
            traci.close()
        
        if 'D' in self.options.mode:
            return np.append(self.observation_history[return_idx], self.action_history), \
                    self.reward, self.done, \
                    {'message': info, 'lane': self.laneID, 'action': action}

        return observationArray, self.reward, self.done, {'message': info, 'lane': self.laneID, 'action': action}


    def render(self):
        print('nothing can be rendered')

