import csv
import os
import random
import math

from ddpg_create_xml import make_xml
import numpy as np
import torch as T
from pybullet_sim import sim

with open('ddpg_mods.csv') as f:
    reader = csv.reader(f)
    modules = list(reader)

EPS_END = .05
EPS_START = .9
EPS_DECAY = 1000
def choose_action(actor, state, variables, goal, env, noise, ep):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * ep / EPS_DECAY)
    print('sample: ', sample)
    print('threshold: ', eps_threshold)
    if sample > eps_threshold:
        print('exploitation')
        actor.eval()
        mu = actor(state, variables, goal, 0)
        print('mu: ', mu)
        noise = T.tensor(noise(ep), dtype=T.float)
        print('noise tensor: ', noise)
        mu_prime = mu + noise
        print('mu prime: ', mu_prime)
        if mu_prime[0] > env.action_bounds[0]:
            mu_prime[0] = env.action_bounds[0]
        if mu_prime[0] <= 0:
            mu_prime[0] = 0.05
        if mu_prime[1] > env.action_bounds[1]:
            mu_prime[1] = env.action_bounds[1]
        if mu_prime[1] <= 0:
            mu_prime[1] = 0
        #print('mu prime: ', mu_prime)
        actor.train()
        #print('mu prime: ', mu_prime)
        return mu_prime.detach().numpy()
    else:
        print('exploration')
        random_length = random.uniform(0, env.action_bounds[0])
        random_twist = random.uniform(0, env.action_bounds[1])
        random_action = np.array([random_length, random_twist])
        return random_action

def reward(env, curr, next_variables):
    act_weight = 0.025
    rew = 0
    dist = 0
    end_eff_pos = (0.0, 0.0, 0.0)
    mod = env.state[curr:curr + env.mod_size]
    if mod[0] == 1:  # accounting for number of actuators
        for i in range(0, len(env.state), env.mod_size):
            curr_mod = env.state[i:i + env.mod_size]
            if curr_mod[0] == 1:
                rew -= act_weight
    #print('curr: ', curr)
    #print('comparing env.active_l_cnt: ', env.active_l_cnt)
    #print('with env.l_cnt: ', env.l_cnt)
    if env.active_l_cnt == env.l_cnt:
        dist, term_rew, end_eff_pos = term_reward(env, curr, next_variables)
        rew += term_rew
    return dist, rew, end_eff_pos

def term_reward(env, curr, next_variables):
    arrangement = [''] * int((len(env.state) / env.mod_size))
    info = ['DDPG-Result', '0.0.1']
    action_tuple = next_variables.reshape((env.l_cnt,2))
    xml_tuple = []
    for i in range(len(action_tuple)):
        xml_tuple.append((str(action_tuple[i][0]), '${' + str(action_tuple[i][1]) + '}'))
    l_cnt = 0
    #print('xml tuple: ', xml_tuple)
    for i in range(int((len(env.state) / env.mod_size))):
        mod = env.state[i * env.mod_size:(i + 1) * env.mod_size]  # current module
        # print('mod: ', mod)
        for j in range(len(mod)):
            val = mod[j]
            if val == 1:
                #print('j: ', j)
                if j <= 2 or j == 4: # if module is actuator, bracket, or gripper just get first 2 items
                    arrangement[i] = modules[j][0:2]
                else: # case of link
                    link_tuple = modules[j][0:2]
                    action = xml_tuple[l_cnt]
                    link_tuple.append(action[0])
                    link_tuple.append(action[1])
                    arrangement[i] = link_tuple
                    l_cnt += 1
                break
    #print('arrangement: ', arrangement)
    make_xml(arrangement, info)  # generate the xacro for the arm
    cmd = 'rosrun xacro xacro custom.xacro > custom.urdf'
    os.system(cmd)  # convert xacro to urdf
    dist, end_eff_pos = sim(env.goal, env.epsilon)  # do pybullet simulation for IK
    rew = binary_rew(dist, env)
    return dist, rew, end_eff_pos

def binary_rew(dist, env):
    if dist < env.epsilon:
        return 1
    else:
        return 0