import argparse
import numpy as np
import random
import math
import csv
import pybullet as p
import pybullet_data
import copy

import torch as T
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from collections import namedtuple
from both_actions import masking, dqn_choose_action, sac_choose_action, dqn_reward, sac_reward, sac_term_reward
from pybullet_sim import sim
from both_agent import Agent, ReplayMemory

writer = SummaryWriter()

# reading in CSV file of modules and converting to list
with open('dqn_sac.csv') as f:
    reader = csv.reader(f)
    modules = list(reader)

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--gamma', type=float, default=1.0, metavar='G',
                    help='discount factor for reward (default: 1.0)')
parser.add_argument('--actor_lr', type=float, default=0.01, metavar='G',
                    help='actor learning rate (default: 0.0003)')
parser.add_argument('--critic_lr', type=float, default=0.01, metavar='G',
                    help='critic learning rate (default: 0.0003)')
parser.add_argument('--dqn_lr', type=float, default = 0.006, metavar='G',
                    help='dqn learning rate (default: 0.004)')
parser.add_argument('--alpha', type=float, default=0.1, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')

args = parser.parse_args()

class env():
    def __init__(self, arm_size, num_vars, action_space):
        self.orientation = False
        # module list
        self.actions = [T.from_numpy(np.array([1, 0, 0, 0])).type(T.FloatTensor),
                        T.from_numpy(np.array([0, 1, 0, 0])).type(T.FloatTensor),
                        T.from_numpy(np.array([0, 0, 1, 0])).type(T.FloatTensor),
                        T.from_numpy(np.array([0, 0, 0, 1])).type(T.FloatTensor)]
        self.mod_size = len(self.actions)
        self.successful_goals = []
        self.successful_arrangements = []
        self.state = T.from_numpy(np.array([0, 0, 0, 0] * arm_size)).type(T.FloatTensor)
        self.min_goal_range = .1
        self.max_goal_range = .5
        self.pos = T.from_numpy(np.array([random.uniform(self.min_goal_range, self.max_goal_range), random.uniform(self.min_goal_range, self.max_goal_range),
                         random.uniform(self.min_goal_range, self.max_goal_range)])).type(T.FloatTensor)
        self.orient_list = [[0,0,0,1],
                            [0,0,math.sqrt(2)/2,math.sqrt(2)/2],
                            [0,0,-1,0],
                            [0,0,-math.sqrt(2)/2,math.sqrt(2)/2]]
                            #[0, 0, 0.383, 0.924],
                            #[0, 0, -0.383, 0.924]]
                            #[0,math.sqrt(2)/2,0,math.sqrt(2)/2],
                            #[0,-math.sqrt(2)/2,0,math.sqrt(2)/2]]
        self.orient = T.from_numpy(np.array(random.choice(self.orient_list))).type(T.FloatTensor)
        if self.orientation:
            self.goal = T.cat((self.pos,self.orient),0)
        else:
            self.goal = self.pos
        self.arm_size = arm_size
        self.variables = []
        self.num_vars = num_vars
        self.prev_action = 1
        self.curr_action = 1
        self.action_space = action_space
        self.max_validation_rew = 0

        self.batch_size = 25

        self.buffer_goal = []
        self.buffer_reward = []
        self.failed = 0
        self.sequence = []
        self.pos_epsilon = .02
        self.orient_epsilon = .05
        self.l_cnt = int(self.arm_size/2)
        self.explore_episodes = 500
        self.ep = 0
        self.target_update_interval = 10
        self.learn_step_cntr = 0
        self.cont_vars = [0]#*self.num_vars
        self.active_l_cnt = 0
        self.evaluate = False
        self.ep = 0

    def reset(self):
        a = np.array([0, 0, 0, 0] * self.arm_size)
        self.state = T.from_numpy(a).type(T.FloatTensor)
        self.prev_action = 1
        pos = np.array([random.uniform(self.min_goal_range, self.max_goal_range), random.uniform(self.min_goal_range, self.max_goal_range),
                         random.uniform(self.min_goal_range, self.max_goal_range)])
        self.pos = T.from_numpy(pos).type(T.FloatTensor)  # randomizing location of goal
        self.orient = T.from_numpy(np.array(random.choice(self.orient_list))).type(T.FloatTensor)
        if self.orientation:
            self.goal = T.cat((self.pos,self.orient),0)
        else:
            self.goal = self.pos
        self.variables = np.zeros((self.l_cnt* self.num_vars))
        self.buffer_goal = []
        self.sequence = []
        self.failed = 0
        self.active_l_cnt = 0
        self.cont_vars = [0]#*self.num_vars

    def step(self, a, curr, goal, action):
        done = (action == self.mod_size - 1)
        # storing previous action
        self.prev_action = action

        # updating the current module with the chosen action
        next_a = copy.deepcopy(a)
        next_a[curr: curr + self.mod_size] = self.actions[action]
        next_vars = copy.deepcopy(self.variables)
        if action == self.mod_size - 2:
            # if action was a link, add link length to variables
            next_vars[(self.active_l_cnt - 1) * self.num_vars:(self.active_l_cnt) * self.num_vars] = self.cont_vars

        # obtaining reward for new arrangement and distance from goal if terminal arrangement
        dqn_r  = dqn_reward(next_vars, next_a, curr, self.mod_size, goal, action, modules, env)
        sac_pos_dist, sac_orient_dist, sac_r, sac_pos = sac_reward(next_a, env, curr, next_vars, goal)
        dqn_r += sac_r

        if done:
            if self.orientation:
                if sac_pos_dist > self.pos_epsilon or sac_orient_dist > self.orient_epsilon:
                    self.buffer_goal = T.from_numpy(np.array(sac_pos)).type(T.FloatTensor)
                    self.failed = 1
            else:
                if sac_pos_dist > self.pos_epsilon:
                    self.buffer_goal = T.from_numpy(np.array(sac_pos)).type(T.FloatTensor)
                    self.failed = 1

        mask = T.zeros(1, self.mod_size, dtype=T.bool)
        if action == 0:
            mask[0][0:1] = True
        else:
            mask[0][1:self.mod_size] = True

        return next_a, next_vars, dqn_r, done, mask, sac_r, sac_pos_dist, sac_orient_dist

    def test_step(self, curr):
        #sum = 0
        with T.no_grad():
            q_vals, env.Z = agent.dqn.forward(self.state, self.variables, self.goal, T.tensor(np.array([curr])).type(T.FloatTensor), 0, env)
            qvals = q_vals.detach().numpy()
        qvals = masking(qvals, self.mod_size, self.prev_action)
        action = np.argmax(qvals)

        if action != 0:
            self.active_l_cnt = self.active_l_cnt + 1
        self.cont_vars = sac_choose_action(self, action, self.action_space, ep, agent)
        self.variables[(self.active_l_cnt - 1) * self.num_vars:(self.active_l_cnt) * self.num_vars] = self.cont_vars

        self.prev_action = action
        self.state[curr: curr + self.mod_size] = self.actions[action]

        return action

def validation(env, ep):
    env.evaluate = True
    print('STARTING VALIDATION AT: ', ep)
    validation_rew = 0
    for goal in val_goals:
        for curr in range(0, len(env.state), env.mod_size):
            action = env.test_step(curr)
            if action != 0:
                writer.add_scalar('Validation Distance' + str(goal) + '/component ' + str((curr / env.mod_size) + 1), action, ep)
            #print('validation action is: ', action)
            if action == env.mod_size - 1:
                final_pos_dist, final_orient_dist, final_rew, end_eff_pos = sac_term_reward(env.state, env, curr, goal, env.variables)
                test_a = env.state.numpy()
                validation_rew += final_rew
                if goal == val_goals[0]:
                    print('val goal 0')
                    val_arrangements1.append((ep,test_a))
                    #print('new val arrangements: ', test_a)
                    writer.add_scalar('Validation Distance/(.1,.1,.1)', final_pos_dist, ep)
                    for i in range(0, int(env.arm_size/2), env.num_vars):
                        writer.add_scalar('Validation Distance(.1,.1,.1)/ vars' + str(i), env.variables[i:i+env.num_vars], ep)
                elif goal == val_goals[1]:
                    print('val goal 1')
                    writer.add_scalar('Validation Distance/(.2,.2,.2)', final_pos_dist, ep)
                    for i in range(0, int(env.arm_size/2), env.num_vars):
                        writer.add_scalar('Validation Distance(.2,.2,.2)/ vars' + str(i), env.variables[i:i+env.num_vars], ep)
                    val_arrangements2.append((ep,test_a))
                elif goal == val_goals[2]:
                    print('val goal 2')
                    writer.add_scalar('Validation Distance/(.3,.3,.3)', final_pos_dist, ep)
                    for i in range(0, int(env.arm_size/2), env.num_vars):
                        writer.add_scalar('Validation Distance(.3,.3,.3)/ vars' + str(i), env.variables[i:i+env.num_vars], ep)
                    val_arrangements3.append((ep,test_a))
                elif goal == val_goals[3]:
                    print('val goal 3')
                    writer.add_scalar('Validation Distance/(.4,.4,.4)', final_pos_dist, ep)
                    for i in range(0, int(env.arm_size/2), env.num_vars):
                        writer.add_scalar('Validation Distance(.4,.4,.4)/ vars' + str(i), env.variables[i:i+env.num_vars], ep)
                    val_arrangements4.append((ep,test_a))
                else:
                    print('val goal 4')
                    writer.add_scalar('Validation Distance/(.5,.5,.5)', final_pos_dist, ep)
                    for i in range(0, int(env.arm_size/2), env.num_vars):
                        writer.add_scalar('Validation Distance(.5,.5,.5)/ vars' + str(i), env.variables[i:i+env.num_vars], ep)
                    val_arrangements5.append((ep,test_a))
                break
        env.reset()
    print('validation reward: ', validation_rew)
    writer.add_scalar('Validation Distance/validation reward', validation_rew, ep)
    T.save(agent.link_policy, "./save_model/latest_link_policy.pth")
    T.save(agent.critic, "./save_model/latest_critic.pth")
    T.save(agent.dqn, "./save_model/latest_dqn.pth")
    if validation_rew >= env.max_validation_rew:
        env.max_validation_rew = validation_rew
        print('model saved')
        print('new max validation reward: ', validation_rew)
        T.save(agent.link_policy, "./save_model/best_link_policy.pth")
        T.save(agent.critic, "./save_model/best_critic.pth")
        T.save(agent.dqn, "./save_model/best_dqn.pth")
    env.evaluate = False
    print('ENDING VALIDATION AT: ', ep)

def make_arrangement(env, arrangement_nums):
    arrangement = [''] * int((len(arrangement_nums) / env.mod_size))
    for i in range(int(curr / env.mod_size) + 1):
        mod = arrangement_nums[i * env.mod_size:(i + 1) * env.mod_size]  # current module
        # print('mod: ', mod)
        for j in range(len(mod)):
            val = mod[j]
            if val == 1:
                # trimming values from CSV for modules
                if j < 2 or j == (env.mod_size - 1):  # if module is actuator, bracket, or gripper just get first 2 items
                    arrangement[i] = modules[j][0:2]
                else:  # if module is link get first four items
                    arrangement[i] = modules[j][0:3]
                # print('arrangement: ', arrangement)
                break
    return arrangement

if __name__ == '__main__':
    # parameters
    DQN_LR_DECAY = 8000
    SAC_LR_DECAY = 3000
    gamma = 1
    train_episodes = 3500
    test_episodes = 100
    print('initializing env')
    env = env(arm_size=10, num_vars=1, action_space=np.array([[0.1, 0.5]])) # initialize environment
    print('env initialized')
    if env.orientation:
        val_goals = [[.1, .1, .1,0,0,0,1], [.2, .2, .2,0,0,0,1], [.3, .3, .3,0,0,0,1], [.4, .4, .4,0,0,0,1], [.5,.5,.5,0,0,0,1]]
    else:
        val_goals = [[.1, .1, .1], [.2, .2, .2], [.3, .3, .3], [.4, .4, .4], [.5,.5,.5]]

    val_arrangements1 = []
    val_arrangements2 = []
    val_arrangements3 = []
    val_arrangements4 = []
    val_arrangements5 = []

    print('initializing agent')
    agent = Agent(len(env.state), env.action_space, len(env.goal), args, env.num_vars, env.l_cnt, env)
    print('agent initialized')
    env.Z_size = agent.Z_size
    lambda_dqn = lambda ep: math.exp(-ep / DQN_LR_DECAY)
    agent.dqn_scheduler = optim.lr_scheduler.LambdaLR(agent.dqn_optim, lambda_dqn)
    lambda_sac = lambda ep: math.exp(-ep / SAC_LR_DECAY)
    agent.actuator_policy_scheduler = optim.lr_scheduler.LambdaLR(agent.actuator_policy_optim, lambda_sac)
    agent.bracket_policy_scheduler = optim.lr_scheduler.LambdaLR(agent.bracket_policy_optim, lambda_sac)
    agent.link_policy_scheduler = optim.lr_scheduler.LambdaLR(agent.link_policy_optim, lambda_sac)
    agent.gripper_policy_scheduler = optim.lr_scheduler.LambdaLR(agent.gripper_policy_optim, lambda_sac)
    agent.critic_scheduler = optim.lr_scheduler.LambdaLR(agent.critic_optim, lambda_sac)

    buffer = ReplayMemory(100)
    # training
    total_loss = 0
    dist = 0
    TARGET_UPDATE = 10
    total_dqn_rew = 0
    total_sac_rew = 0
    sac_iter = 3
    dqn_iter = 3
    #print('env.mod_size: ', env.mod_size)
    for ep in range(train_episodes):
        env.ep = ep
        if ep == 1500:
            sac_iter = 2
            dqn_iter = 2
        if ep == 3750:
            sac_iter = 1
            dqn_iter = 1
        print('')
        print('EPISODE: ', ep)
        print('')
        env.reset()
        if (ep+10 % 50) == 0:
            validation(env, ep)
        ep_dqn_rew = 0
        ep_sac_rew = 0
        ep_dqn_loss = 0
        ep_sac_loss = 0
        for curr in range(0, len(env.state), env.mod_size):
            env.c = T.tensor(np.array([curr])).type(T.FloatTensor)
            dqn_action, env.Z = dqn_choose_action(env, agent, env.state, env.mod_size, env.prev_action, env.goal, ep, curr)
            if dqn_action != 0:
                env.active_l_cnt = env.active_l_cnt + 1
            #print('dqn action:', dqn_action)
            #print('env.mod_size - 2: ', env.mod_size - 2)
            #print(dqn_action != env.mod_size - 2)
            #print('CHOOSING SAC ACTION')
            sac_action = sac_choose_action(env, dqn_action, env.action_space, ep, agent)
            env.cont_vars = sac_action
            #print('active l cnt: ', env.active_l_cnt)
            next_a, next_variables, dqn_r, done, mask, sac_r, sac_pos_dist, sac_orient_dist = env.step(env.state,curr, env.goal, dqn_action)
            #print('sac dist: ', sac_dist)
            ep_dqn_rew += dqn_r
            ep_sac_rew += sac_r
            print('state:',env.state)
            print('vars:',T.tensor(np.array(env.variables)).type(T.FloatTensor))
            print('next state', next_a)
            print('next vars:',T.tensor(np.array(next_variables)).type(T.FloatTensor))
            print('goal:',env.goal)
            print('c:',env.c)
            print('Z:',env.Z)
            print('dqn act',T.tensor(np.array([dqn_action])).type(T.LongTensor))
            print('sac act:',T.tensor(np.array(sac_action)).type(T.FloatTensor).view(env.num_vars))
            print('dqn r:',T.tensor(np.array([dqn_r])).type(T.FloatTensor).view(1))
            print('sac r:',T.tensor(np.array(sac_r)).type(T.FloatTensor).view(1))
            print('done:',T.tensor(np.array([1 - done])).type(T.LongTensor))
            print('mask: ',mask)
            buffer.push(env.state, T.tensor(np.array(env.variables)).type(T.FloatTensor),
                        next_a, T.tensor(np.array(next_variables)).type(T.FloatTensor),
                        env.goal, env.c, env.Z, T.tensor(np.array([dqn_action])).type(T.LongTensor),
                        T.tensor(np.array(sac_action)).type(T.FloatTensor).view(env.num_vars),
                        T.tensor(np.array([dqn_r])).type(T.FloatTensor).view(1),
                        T.tensor(np.array(sac_r)).type(T.FloatTensor).view(1),
                        T.tensor(np.array([1 - done])).type(T.LongTensor),mask)
            if done:
                if env.failed:
                    print('failed')
                    print('goal point was: ', env.goal)
                    print('end effector position was: ', env.buffer_goal)
                    print('pos distance: ', sac_pos_dist)
                    if env.orientation:
                        print('orient distance: ', sac_orient_dist)
                    # do I need to do something about the DQN action here?
                    env.sequence.append((env.state, T.tensor(np.array(env.variables)).type(T.FloatTensor),
                                         next_a, T.tensor(np.array(next_variables)).type(T.FloatTensor), env.c, env.Z,
                                         T.tensor(np.array([dqn_action])).type(T.LongTensor),
                                         T.tensor(np.array(sac_action)).type(T.FloatTensor),
                                         T.tensor(np.array([1.5])).type(T.FloatTensor),
                                         T.tensor(np.array([1.5])).type(T.FloatTensor),
                                         done, mask))
                    for part in env.sequence:
                        #print('state:', part[0])
                        #print('vars:', part[1])
                        #print('next state', part[2])
                        #print('next vars:', part[3])
                        #print('goal:', env.buffer_goal)
                        #print('c:', part[4])
                        #print('Z:', part[5])
                        #print('dqn act', part[6])
                        #print('sac act:', part[7])
                        #print('dqn r:', part[8])
                        #print('sac r:', part[9])
                        #print('done:', T.tensor(np.array([1 - part[10]])).type(T.LongTensor))
                        #print('mask: ', part[11])
                        buffer.push(part[0], part[1], part[2], part[3],
                                    env.buffer_goal, part[4], part[5], part[6], part[7], part[8], part[9],
                                    T.tensor(np.array([1 - part[10]])).type(T.LongTensor), part[11])
                else:
                    print('succeeded')
                    print('goal point was: ', env.goal)
                    print('pos distance: ', sac_pos_dist)
                    #print('length of successful goals: ', len(env.successful_goals))
                    if env.orientation:
                        print('orient distance:', sac_orient_dist)
                writer.add_scalar('Distance/position training distance', sac_pos_dist, ep)
                if env.orientation:
                    writer.add_scalar('Distance/orientation training distance', sac_orient_dist, ep)
                break
            else:
                env.sequence.append((env.state, T.tensor(np.array(env.variables)).type(T.FloatTensor),
                                         next_a, T.tensor(np.array(next_variables)).type(T.FloatTensor), env.c, env.Z,
                                         T.tensor(np.array([dqn_action])).type(T.LongTensor),
                                         T.tensor(np.array(sac_action)).type(T.FloatTensor),
                                         T.tensor(np.array([dqn_r])).type(T.FloatTensor),
                                         T.tensor(np.array([sac_r])).type(T.FloatTensor),
                                         done, mask))
            #for i in range(2):
            dqn_loss = 0
            for i in range(dqn_iter):
                dqn_loss += agent.dqn_optimize_model(buffer, env, ep)
            sac_loss = 0
            for i in range(sac_iter):
                sac_loss += agent.sac_optimize_model(buffer, env, ep)
            #print('dqn_loss: ', dqn_loss)
            #print('sac_loss: ', sac_loss)
            ep_dqn_loss += (dqn_loss/dqn_iter)
            ep_sac_loss += (sac_loss/sac_iter)
            env.state = copy.deepcopy(next_a)
            env.variables = copy.deepcopy(next_variables)
        total_dqn_rew += ep_dqn_rew
        total_sac_rew += ep_sac_rew
        total_loss += (ep_dqn_loss + ep_sac_loss)
        #if ep > (train_episodes/2) and total_dqn_rew < 0:
        #    break
        writer.add_scalar('Loss/dqn train loss', ep_dqn_loss,ep)
        writer.add_scalar('Loss/sac train loss', ep_sac_loss,ep)
        writer.add_scalar('Reward/dqn train reward', ep_dqn_rew, ep)
        writer.add_scalar('Reward/sac train reward', ep_sac_rew, ep)
        writer.add_scalar('Reward/dqn train total reward', total_dqn_rew, ep)
        writer.add_scalar('Reward/sac train total reward', total_sac_rew, ep)

    agent.dqn = T.load("./save_model/best_dqn.pth")
    agent.policy = T.load("./save_model/best_policy.pth")
    agent.critic = T.load("./save_model/best_critic.pth")

    print('SUCCESSFUL GOALS AND ARRANGEMENTS')
    for pair in env.successful_goals:
        print('successful goal:', pair[0])
        print('successful arrangement: ', pair[1])
    print('')

    env.evaluate = True
    # testing
    results = []
    for test in range(test_episodes):
        print('goal: ', env.goal)
        env.reset()
        for curr in range(0,len(env.state), env.mod_size):
            action = env.test_step(curr)
            #print('action: ', action)
            if action == env.mod_size - 1:
                final_pos_dist, final_orient_dist, rew, end_eff_pos = sac_term_reward(env.state, env, curr, env.goal, env.variables)
                #sac_term_reward(next_a, env, curr, goal, next_variables):
                #print('final distance: ', final_dist[0])
                writer.add_scalar('Distance/Test Position Distance', final_pos_dist, test)
                writer.add_scalar('Distance/Test Orientation Distance', final_orient_dist, test)
                break
    cntr = 0
    print('goal: ', val_goals[cntr])
    for i in range(len(val_arrangements1)):
        arrangement = val_arrangements1[i]
        print(arrangement[0], make_arrangement(env, arrangement[1]))
    cntr +=1
    print('goal: ', val_goals[cntr])
    for i in range(len(val_arrangements2)):
        arrangement = val_arrangements2[i]
        print(arrangement[0], make_arrangement(env, arrangement[1]))
    cntr += 1
    print('goal: ', val_goals[cntr])
    for i in range(len(val_arrangements3)):
        arrangement = val_arrangements3[i]
        print(arrangement[0], make_arrangement(env, arrangement[1]))
    cntr += 1
    print('goal: ', val_goals[cntr])

    for i in range(len(val_arrangements4)):
        arrangement = val_arrangements4[i]
        print(arrangement[0], make_arrangement(env, arrangement[1]))
    cntr += 1
    print('goal: ', val_goals[cntr])
    for i in range(len(val_arrangements5)):
        arrangement = val_arrangements5[i]
        print(arrangement[0], make_arrangement(env, arrangement[1]))


