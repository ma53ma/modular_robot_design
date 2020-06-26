import numpy as np
from create_xml import make_xml
import os
from pybullet_sim import sim
import math
import torch

def masking(probs,mod_size, prev_action):
    if prev_action > 0: # if the previous module was not an actuator, mask all but the actuator
        mask_range = [1,mod_size]
    else: # if the previous module was an actuator, mask actuator
        mask_range = [0,1]
    probs[mask_range[0]:mask_range[1]] = [0] * (mask_range[1] - mask_range[0])

    # normalize probabilities
    sum = np.sum(probs)
    probs = [i / sum for i in probs]

    return probs

def choose_action(network,a, mod_size, prev_action, goal, steps_done):
    # get Q values for arrangement
    with torch.no_grad():
        values = network.forward(a, goal, 0).detach().numpy()

    # boltzmann exploration
    probs = [0] * mod_size
    sum = 0
    t_end = 0.05
    t_begin = 0.5
    t_decay = 500
    t = t_end + (t_begin - t_end)*np.exp(-steps_done / t_decay)
    for i in range(mod_size):
        sum += np.exp(values[i] / t)
    for i in range(mod_size):
        probs[i] = np.exp(values[i] / t) / sum

    # perform action masking
    probs = masking(probs, mod_size, prev_action)
    print('probabilities for actions: ', probs)

    # pick action
    act = np.random.choice(range(mod_size), p=probs)
    return act

# reward function
def reward(a, curr, mod_size, goal, action, modules):
    masses = [.315, .215, .215, (0.4 * (float(modules[3][2]) + 0.03) + 0.26), (0.4 * (float(modules[4][2]) + 0.03) + 0.26), 0.0]
    mass_weight = 0.1
    act_weight = 0.025
    rew = 0
    dist = 0
    a = a.numpy()
    pos = (0.0, 0.0, 0.0)
        #if a[i] == 1: # accounting for mass of each element
        #    cost -= mass_weight*masses[i % len(masses)]
    mod = a[curr:curr + mod_size]
    if mod[0] == 1: # accounting for number of actuators
        for i in range(0, len(a), mod_size):
            curr_mod = a[i:i + mod_size]
            if curr_mod[0] == 1:
                rew -= act_weight
    if action == mod_size - 1 or curr == len(a) - mod_size: # if putting end effector on or at last module, obtain terminal reward
        dist, term_r, end_eff_pos = term_reward(a, mod_size, goal, curr, modules)
        rew += term_r
        pos = end_eff_pos
    return rew, dist, pos

def term_reward(a, mod_size, goal, curr, modules):
    arrangement = [''] * int((curr / mod_size) + 1)
    info = ['DQN-Result', '0.0.1']
    for i in range(curr + mod_size):
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
    if arrangement[-1] != modules[-1][0:2]: # if last module is not end-effector
        return 0, -1, (0.0, 0.0, 0.0) # just returning dist from goal as 0 and reward as -1
    else:
        make_xml(arrangement, info) # generate the xacro for the arm
        cmd = 'rosrun xacro xacro custom.xacro > custom.urdf'
        os.system(cmd) # convert xacro to urdf
        dist, end_eff_pos = sim(goal) # do pybullet simulation for IK
        rew = binary_rew(dist)
        return dist, rew, end_eff_pos

def pos_neg_rew(dist):
    return 2*math.exp(-20*dist) - .25

def soft_rew(dist):
    return math.exp(-10*dist)

def binary_rew(dist):
    if dist < .05:
        return 1
    else:
        return 0