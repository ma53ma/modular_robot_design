import numpy as np
import random
import math
import csv
import pybullet as p
import pybullet_data
import os

import torch
from torch import nn
import torch.optim as optim
from create_xml import make_xml

steps_done = 0
prev_action = 0

# reading in CSV file of modules and converting to list
with open('mini_mods.csv') as f:
    reader = csv.reader(f)
    modules = list(reader)

# class for DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size, lr, gamma):
        super(DQN, self).__init__()
        self.model = nn.Sequential(nn.Linear(state_size, 64),
                                    nn.ReLU(), nn.Linear(64, 32),
                                    nn.ReLU(), nn.Linear(32, action_size))
        self.optimizer = optim.Adam(self.parameters(),lr)
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.gamma = gamma

    def forward(self, state):
        return self.model(state)

    def step(self, a, curr, mod_size):
        global prev_action

        # determining action
        action = choose_action(self,a, mod_size, curr, prev_action)

        # storing previous action
        prev_action = action

        # one-hot vectors for each module type we could potentially add
        options = [torch.from_numpy(np.array([1, 0, 0, 0, 0, 0])).type(torch.FloatTensor),
                   torch.from_numpy(np.array([0, 1, 0, 0, 0, 0])).type(torch.FloatTensor),
                   torch.from_numpy(np.array([0, 0, 1, 0, 0, 0])).type(torch.FloatTensor),
                   torch.from_numpy(np.array([0, 0, 0, 1, 0, 0])).type(torch.FloatTensor),
                   torch.from_numpy(np.array([0, 0, 0, 0, 1, 0])).type(torch.FloatTensor),
                   torch.from_numpy(np.array([0, 0, 0, 0, 0, 1])).type(torch.FloatTensor)]

        # updating the current module with the chosen action
        next_a = a
        next_a[curr: curr + mod_size] = options[action]

        # forward pass through DQN for old arrangement
        state_vals = self.forward(a)

        # forward pass through DQN for arrangement with new module
        next_state_vals = self.forward(next_a)

        # obtaining reward for new arrangement and distance from goal if terminal arrangement
        cost, dist  = reward(next_a, curr, mod_size)
        r = cost
        # Bellman's equation for updating Q values
        target = r + self.gamma * torch.max(next_state_vals)

        # Computing loss
        loss = self.loss_fn(state_vals[action], target)

        # clear old gradients
        self.optimizer.zero_grad()

        # back-propagate loss
        loss.backward()

        # optimizer step
        self.optimizer.step()

        return loss.item(), dist

def masking(a,probs,curr,mod_size, prev_action):
    if curr == 0: # if this is the first module, have it's previous action be a bracket so that an actuator will be chosen
        prev_action = 1
    if curr == len(a) - mod_size: # if this is the last module, mask the actuator
        mask_range = [0, 1]
    elif prev_action > 0: # if the previous module was not an actuator, mask all but the actuator
        mask_range = [1,mod_size]
    else: # if the previous module was an actuator, mask actuator and gripper
        mask_range = [0,1]
        probs[mod_size - 1] = 0
    probs[mask_range[0]:mask_range[1]] = [0] * (mask_range[1] - mask_range[0])

    # normalize probabilities
    sum = np.sum(probs)
    probs = [i / sum for i in probs]

    return probs

def choose_action(network,a, mod_size, curr, prev_action):
    # get Q values for arrangement
    values = network.forward(a).detach().numpy()

    # boltzmann Exploration
    probs = [0] * mod_size
    sum = 0
    t = 0.5
    for i in range(mod_size):
        sum += np.exp(values[i] / t)
    for i in range(mod_size):
        probs[i] = np.exp(values[i] / t) / sum

    # perform action masking
    probs = masking(a,probs, curr,mod_size, prev_action)

    # pick action
    act = np.random.choice(range(mod_size), p=probs)
    return act

# reward function
def reward(a, curr, mod_size):
    masses = [.315, .215, .215, .402, .402, .05]
    mass_weight = 0.1
    act_weight = 0.025
    cost = 0
    dist = 0
    a = a.numpy()
    for i in range(len(a)):
        if a[i] == 1: # accounting for mass of each element
            cost -= mass_weight*masses[i % len(masses)]
        if i % len(masses) == 0: # accounting for number of actuators
            cost -= act_weight
    if curr == (len(a) - mod_size): # if terminal module, obtain terminal reward
        dist, r = term_reward(a, mod_size)
        cost += r
    return cost, dist

def term_reward(a, mod_size):
    arrangement = [''] * int(len(a) / mod_size)
    info = ['DQN-Result', '0.0.1']
    for i in range(int(len(a) / mod_size)):
        mod = a[i * mod_size:(i + 1) * mod_size] # current module
        for j in range(len(mod)):
            val = mod[j]
            if val == 1:
                # trimming values from CSV for modules
                if j <= 2 or j == 5: # if module is actuator, bracket, or gripper just get first 2 items
                    arrangement[i] = modules[j][0:2]
                else: # if module is link get first four items
                    arrangement[i] = modules[j][0:4]
                break
    #print('arrangement: ', arrangement)

    if arrangement[-1] != modules[-1][0:2]: # if last module is not end-effector
        return 0, -1 # just returning dist from goal as 0 and reward as -1
    else:
        make_xml(arrangement, info) # generate the xacro for the arm
        cmd = 'rosrun xacro xacro custom.xacro > custom.urdf'
        os.system(cmd)
        dist = sim() # do pybullet simulation for IK

        #print('dist: ', dist)
        return dist, math.exp(-dist) # return distance from goal and softened reward from 0 - 1


def sim():
    cubeStartPos = [0,0,0]
    cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
    robotId = p.loadURDF("custom.urdf",cubeStartPos, cubeStartOrientation)

    goal = [random.uniform(-.50,.50),random.uniform(-.50,.50),random.uniform(0,.50)] # randomizing location of goal
    finalJoint = p.getNumJoints(robotId)

    # inverse kinematics
    angles = p.calculateInverseKinematics(robotId, finalJoint - 1, goal)

    p.setRealTimeSimulation(1) # don't need if doing p.DIRECT for server, but do need for p.GUI

    # forward kinematics
    for i in range(len(angles)):
        p.resetJointState(robotId, 3 * i + 2, angles[i])
    endEffPos = p.getLinkState(robotId, finalJoint - 1)[0]

    # determine dist from tip of EE to goal
    dist = np.linalg.norm(np.array([a_i - b_i for a_i, b_i in zip(endEffPos, goal)]))

    # delete arm from simulation
    p.removeBody(robotId)
    #print('dist: ', dist)
    return dist

def print_formatting(test_a, mod_size):
    moduled_output = np.zeros(shape=(int(len(test_a) / mod_size), mod_size))
    for i in range(int(len(test_a) / mod_size)):
        moduled_output[i] = test_a[i * mod_size:(i + 1) * mod_size]
    results = moduled_output.tolist()
    return results

if __name__ == '__main__':
    # parameters
    lr = 1e-4
    gamma = .95
    train_episodes = 500
    test_episodes = 1

    physicsClient = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version, p.DIRECT is faster
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")

    # module list
    mod1 = torch.from_numpy(np.array([1, 0, 0, 0, 0, 0])).type(torch.FloatTensor)
    mod2 = torch.from_numpy(np.array([0, 1, 0, 0, 0, 0])).type(torch.FloatTensor)
    mod3 = torch.from_numpy(np.array([0, 0, 1, 0, 0, 0])).type(torch.FloatTensor)
    mod4 = torch.from_numpy(np.array([0, 0, 0, 1, 0, 0])).type(torch.FloatTensor)
    mod5 = torch.from_numpy(np.array([0, 0, 0, 0, 1, 0])).type(torch.FloatTensor)
    mod6 = torch.from_numpy(np.array([0, 0, 0, 0, 0, 1])).type(torch.FloatTensor)
    actions = [mod1,mod2,mod3, mod4, mod5, mod6]
    mod_size = len(actions)

    # length of arm
    arm_size = 10

    # arrangement set-up
    empty = np.array([0, 0, 0, 0, 0, 0] * arm_size)
    a = torch.from_numpy(empty).type(torch.FloatTensor)

    target_net = DQN(len(a), mod_size,lr,gamma) # initialize network

    # training
    total_loss = 0
    total_dist = 0
    term_times = 0
    for ep in range(train_episodes):
        print('ep: ', ep)
        for curr in range(0, len(a), mod_size):
            #print('curr: ', curr)
            l,dist = target_net.step(a,curr, mod_size)
            total_loss += l
            total_dist += dist
            if dist > 0:
                term_times += 1
        print('mean loss: ', total_loss / (ep + 1))
        print('mean distance: ', total_dist / (term_times + 1))
        a = torch.from_numpy(empty).type(torch.FloatTensor)

    # testing
    final_dist = 0
    results = []
    test_a = torch.from_numpy(empty).type(torch.FloatTensor)
    for test in range(test_episodes):
        for curr in range(0,len(test_a), mod_size):
            values = target_net.forward(test_a).detach().numpy()
            probs = masking(a, values, curr, mod_size, prev_action)
            action = np.argmax(probs)
            prev_action = action
            test_a[curr: curr + mod_size] = actions[action]
        test_a = test_a.numpy()
        results = print_formatting(test_a, mod_size)
        final_dist = term_reward(test_a,mod_size)
        test_a = torch.from_numpy(empty).type(torch.FloatTensor)
    p.disconnect()

    #print('results: ', results)
    print('final distance: ', final_dist[0])