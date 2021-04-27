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
EPS_DECAY = 500

def choose_action(actor, state, variables, env, ep):
    sample = random.random()
    #    sample = 0
    #threshold = (.2*math.exp(-ep/4000))
    threshold = .2
    #if sample < threshold:
    if ep < env.explore_episodes:
        # gaussian distribution
        #random_length = np.random.normal(loc=.375,scale=.125, size=1)
        #random_twist = np.random.normal(loc=np.pi,scale=np.pi/3, size=1)
        #mu = T.tensor(np.array([random_length, random_twist]),dtype=T.float).view(env.n_actions)
        # uniform distribution
        random_length = random.uniform(0.05, env.action_bounds[0])
        random_twist = random.uniform(0, env.action_bounds[1])
        mu = T.tensor(np.array([random_length, random_twist]),dtype=T.float).view(env.n_actions)
    else:
        mu = actor.forward(state, variables, 0, env).view(env.n_actions)
        print('else mu: ', mu)
    print('mu: ', mu)
    length_noise = np.random.normal(scale=.03333, size=1)
    twist_noise = np.random.normal(scale=.125, size=1)
    noise = T.tensor(np.array([length_noise, twist_noise]),dtype=T.float).view(env.n_actions)
    #print('noise: ', noise)
    mu_prime = mu + noise
    mu_prime[0] = T.clamp(mu_prime[0], 0.05, env.action_bounds[0])
    mu_prime[1] = T.clamp(mu_prime[1], 0, env.action_bounds[1])
    print('mu prime: ', mu_prime)
    return mu_prime.cpu().detach().numpy()

def reward(env, curr, next_variables):
    #act_weight = 0.025
    length_weight = .1
    rew = 0
    dist = 0
    #mod = env.state[curr:curr + env.mod_size]
    #if mod[0] == 1:  # accounting for number of actuators
    #    for i in range(0, len(env.state), env.mod_size):
    #        curr_mod = env.state[i:i + env.mod_size]
    #        if curr_mod[0] == 1:
    #            rew -= act_weight
    #print('rew after actuators: ', rew)
    for i in range(0, len(next_variables), 2):
        rew -= next_variables[i] * length_weight
    #print('rew after lengths: ', rew)
    #print('curr: ', curr)
    #print('comparing env.active_l_cnt: ', env.active_l_cnt)
    #print('with env.l_cnt: ', env.l_cnt)
    if env.active_l_cnt == env.l_cnt:
        dist, term_rew = term_reward(env, curr, next_variables)
        rew += term_rew
    #print('rew after terminal: ', rew)
    return dist, rew

def term_reward(env, curr, next_variables):
    length_tot = 0.0
    twist_tot = 0.0
    for i in range(0, len(next_variables), env.num_d_vars):
        set_vars = next_variables[i:i+env.num_d_vars]
        length_tot += set_vars[0]
        twist_tot += set_vars[1]
    print('length_tot: ', length_tot)
    print('length if ', .8 < length_tot < .9)
    print('twist tot: ', twist_tot)
    print('twist if : ', (np.pi + .3) < twist_tot < (np.pi + .7))
    if .8 < length_tot < .9 and (np.pi + .3) < twist_tot < (np.pi + .7):
        return 0, 1
    else:
        return 1, 0

def pos_neg_soft(dist, env):
    if dist < env.epsilon:
        return 1
    else:
        return neg_soft_rew(dist)

def pos_soft_rew(dist):
    return 1

def neg_soft_rew(dist):
    return -.75*dist

def binary_rew(dist, env):
    if dist < env.epsilon:
        return 1
    else:
        return 0