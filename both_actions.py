import numpy as np
import random
from both_xml import make_xml
import os
from both_sim import sim
import math
import csv
import torch as T
import xacro
import os.path
import subprocess
import tempfile

# opening up .csv file with specific information about the module types
with open('dqn_sac.csv') as f:
    reader = csv.reader(f)
    modules = list(reader)

# function that performs action masking based on previous action
def masking(probs,mod_size, prev_action):
    if prev_action > 0: # if the previous module was not an actuator, mask all but the actuator
        mask_range = [1,mod_size]
    else: # if the previous module was an actuator, mask actuator
        mask_range = [0,1]
    probs[mask_range[0]:mask_range[1]] = [-float('inf')] * (mask_range[1] - mask_range[0])

    return probs

# function that choose the continuous action from the SAC networks
def sac_choose_action(env, max_q_val, action_space, ep, agent):
    #print('SAC CHOOSE ACTION')
    if ep < env.explore_episodes: # if within exploration period
        if max_q_val == 0 or max_q_val == 1 or max_q_val == 3:
            action = np.array([0])
        else:
            random_length = random.uniform(action_space[0][0], action_space[0][1])
            #random_twist = random.uniform(action_space[1][0], action_space[1][1])
            action = np.array([random_length])#, random_twist])
            #print('sac action is: ', action)
    else: # if within exploitation period
        action = agent.select_action(env.Z, max_q_val, env,evaluate=env.evaluate)

    #print('env active l cnt before: ', env.active_l_cnt)
    #env.active_l_cnt = env.active_l_cnt + 1
    #print('env active l cnt after: ', env.active_l_cnt)
    #print('sac action: ', action)
    return action

# function that choose the discrete action from the DQN network
def dqn_choose_action(env, agent,a, mod_size, prev_action, goal, ep, curr):
    # get Q values for arrangement
    with T.no_grad():
        q_vals, Z = agent.dqn.forward(a, env.variables, goal, T.tensor(np.array([curr])).type(T.FloatTensor), 0, env)
        values = q_vals.detach().numpy()
    values = masking(values, mod_size, prev_action)
    if ep < env.explore_episodes: # in exploration phase
        if prev_action > 0:
            act = 0
        else:
            choices = list(range(1, env.mod_size))
            act = np.random.choice(choices)
        return act, Z
    else: # in exploitation phase
        # boltzmann exploration
        sum = 0
        t_end = 0.01
        t_begin = 0.5
        t_decay = 1000
        t = t_end + (t_begin - t_end)*np.exp(-ep / t_decay)
        #print('values before boltzmann: ', values)
        max_val = np.amax(values)
        values = np.subtract(values, max_val)
        #print('values before subtract: ', values)

        # normalizing
        for i in range(mod_size):
            sum += np.exp(values[i] / t)
        for i in range(mod_size):
            values[i] = np.exp(values[i] / t) / sum
        #print('values after boltzmann: ', values)

        # perform action masking
        #print('probabilities for actions: ', values)

        # pick action
        act = np.random.choice(range(mod_size), p=values)
        return act, Z

# reward function for discrete actions
def dqn_reward(variables, a, curr, mod_size, goal, action, modules, env):
    ## NOT USED FOR LINKS
    # masses for each module type
    masses = [.315, .215, 0.0, 0.0]
    # weights for mass and actuator penalties
    mass_weight = 0.1
    act_weight = 0.025
    rew = 0
    a = a.numpy()
    mod = a[curr:curr + mod_size]
    # penalizing for mass of current module
    for j in range(len(mod)):
        if mod[j] == 1:
            rew -= mass_weight*masses[j]
    # penalizing for actuator if added
    if mod[0] == 1: # accounting for number of actuators
        rew -= act_weight
    # penalizing if last possible module added is not a gripper
    if curr == len(a) - mod_size and mod[-1] != 1:
        print('not terminal')
        rew -= 1
    return rew

# reward function for continuous actions
def sac_reward(next_a, env, curr, next_variables, goal):
    # weight for length of link
    length_weight = .1
    rew = 0
    pos_dist = 0
    orient_dist = 0
    end_eff_pos = (0.0, 0.0, 0.0)
    if env.prev_action == env.mod_size - 2: # penalizing for length of link if link is added
        #for i in range(0, len(next_variables), 2):
        #    if next_variables[i] > 0:
        # print('next_variables[(env.active_l_cnt - 1)*env.num_vars]:', next_variables[(env.active_l_cnt - 1)*env.num_vars])
        rew -= ((0.4 * (next_variables[(env.active_l_cnt - 1)*env.num_vars] + 0.03) + 0.26)*length_weight)
    elif env.prev_action == env.mod_size - 1: # adding in terminal reward if gripper is added
        pos_dist,orient_dist, term_rew, end_eff_pos = sac_term_reward(next_a, env, curr, goal, next_variables)
        rew += term_rew
    return pos_dist, orient_dist,rew, end_eff_pos

# terminal reward function for when gripper is added
def sac_term_reward(next_a, env, curr, goal, next_variables):
    # configuring syntax for an xacro
    arrangement = [''] * int((len(next_a) / env.mod_size))
    info = ['DQN-SAC-Result', '0.0.1']
    action_tuple = next_variables.reshape((env.l_cnt,env.num_vars))
    #print('action tuple: ', action_tuple)
    xml_tuple = []
    for i in range(len(action_tuple)):
        #if action_tuple[i][1] > (np.pi/2):
        #    action_tuple[i][1] = np.pi/2
        #else:
        #    action_tuple[i][1] = 0
        #xml_tuple.append((str(action_tuple[i][0]), '${' + str(action_tuple[i][1]) + '}'))
        xml_tuple.append((str(action_tuple[i][0])))
    l_cnt = 0
    #print('xml tuple: ', xml_tuple)
    #print('state: ', env.state)
    for i in range(int((len(next_a) / env.mod_size))):
        mod = next_a[i * env.mod_size:(i + 1) * env.mod_size]  # current module
        # print('mod: ', mod)
        for j in range(len(mod)):
            val = mod[j]
            if val == 1:
                #print('j: ', j)
                if j <= 1 or j == (env.mod_size - 1): # if module is actuator, bracket, or gripper just get first 2 items
                    arrangement[i] = modules[j][0:2]
                else: # case of link
                    link_tuple = modules[j][0:2]
                    #print('link tuple: ', link_tuple)
                    action = xml_tuple[l_cnt]
                    #print('action: ', action)
                    link_tuple.append(action)
                    link_tuple.append(modules[j][2])
                    #link_tuple.append(action[1])
                    #print('link tuple: ', link_tuple)
                    arrangement[i] = link_tuple
                if j != 0:
                    l_cnt += 1
                    print('l_cnt: ', l_cnt)
                break
    #print('arrangement: ', arrangement)
    make_xml(arrangement, info)  # generate the xacro for the arm
    cmd = 'xacro dqn_sac.xacro > dqn_sac.urdf'
    os.system(cmd)  # convert xacro to urdf
    pos_dist, orient_dist, end_eff_pos = sim(goal, env.pos_epsilon, env.orient_epsilon)  # do pybullet simulation for IK
    # finding reward for simulation result
    if env.orientation:
        rew = binary_orient_rew(pos_dist, orient_dist, env)
        if rew == 1.5:
            env.successful_goals.append((env.goal, arrangement))
    else:
        rew = binary_rew(pos_dist, env)
    return pos_dist, orient_dist, rew, end_eff_pos

# different reward functions for simulation
def pos_neg_soft(dist, env):
    if dist < env.epsilon:
        return pos_soft_rew(dist)
    else:
        return neg_soft_rew(dist)

def pos_soft_rew(dist):
    return .5*math.exp(-40*dist) + 1

def neg_soft_rew(dist):
    return -.8*dist

def soft_rew(dist):
    return math.exp(-10*dist)

def binary_rew(pos_dist, env):
    if pos_dist < env.pos_epsilon:
        return 1.5
    else:
        return 0

def binary_orient_rew(pos_dist, orient_dist, env):
    if pos_dist < env.pos_epsilon and orient_dist < env.orient_epsilon:
        return 1.5
    else:
        return 0

def tiered_binary_orient_rew(pos_dist, orient_dist, env):
    if pos_dist < env.pos_epsilon and orient_dist < env.orient_epsilon:
        return 1.5
    elif pos_dist < env.pos_epsilon:
        return 1.0
    elif orient_dist < env.orient_epsilon:
        return .1
    else:
        return 0