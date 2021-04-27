import numpy as np
import random
import math
import csv
import pybullet as p
import pybullet_data
import copy


import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from collections import namedtuple
from dqn_sac_actions import masking
from dqn_sac_actions import dqn_choose_action
from dqn_sac_actions import sac_choose_action
from dqn_sac_actions import reward
from dqn_sac_actions import term_reward
from pybullet_sim import sim

steps_done = 0

writer = SummaryWriter()
# reading in CSV file of modules and converting to list
with open('dqn_sac.csv') as f:
    reader = csv.reader(f)
    modules = list(reader)

# class for DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size, lr, gamma, target_size):
        super(DQN, self).__init__()
        self.goal_layer = nn.Linear(target_size, 9)
        self.a_layer = nn.Linear(state_size, 64)
        self.model = nn.Sequential(nn.Linear(73, 128), nn.ReLU(), nn.Linear(128, 64),nn.ReLU(), nn.Linear(64, 32),
                                    nn.ReLU(), nn.Linear(32, action_size))
        #self.optimizer = optim.Adam(self.parameters(),lr)
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)


    def forward(self, state, goal, batch):
        a_res = self.a_layer(state)
        #print('a_res: ', a_res)
        goal_res = self.goal_layer(goal)
        #print('goal_res: ', goal_res)
        if batch:
            tot_res = torch.cat((a_res, goal_res),1)
        else:
            tot_res = torch.cat((a_res, goal_res), 0)
        #print('tot_res', tot_res)
        return self.model(tot_res)

class env():
    def __init__(self, arm_size):
        a = np.array([0, 0, 0, 0, 0, 0] * arm_size)
        self.state = torch.from_numpy(a).type(torch.FloatTensor)
        goal = np.array([random.uniform(-.5, 0.5), random.uniform(-.5, 0.5),
                         random.uniform(0.0, 0.5)])
        #goal_or = [random.uniform(-1,1) for i in range(3)]
        #mag = sum(x ** 2 for x in goal_or) ** .5
        #goal_or = [x / mag for x in goal_or]
        self.goal = torch.from_numpy(goal).type(torch.FloatTensor)
        self.arm_size = arm_size
        self.prev_action = 1
        self.variables = []

        # module list
        self.actions = [torch.from_numpy(np.array([1, 0, 0, 0, 0, 0])).type(torch.FloatTensor),
                        torch.from_numpy(np.array([0, 1, 0, 0, 0, 0])).type(torch.FloatTensor),
                        torch.from_numpy(np.array([0, 0, 1, 0, 0, 0])).type(torch.FloatTensor),
                        torch.from_numpy(np.array([0, 0, 0, 1, 0, 0])).type(torch.FloatTensor),
                        torch.from_numpy(np.array([0, 0, 0, 0, 1, 0])).type(torch.FloatTensor),
                        torch.from_numpy(np.array([0, 0, 0, 0, 0, 1])).type(torch.FloatTensor)]
        self.mod_size = len(self.actions)
        self.batch_size = 25

        self.buffer_goal = []
        self.buffer_reward = []
        self.failed = 0
        self.sequence = []
        self.epsilon = .01

    def reset(self):
        a = np.array([0, 0, 0, 0, 0, 0] * self.arm_size)
        self.state = torch.from_numpy(a).type(torch.FloatTensor)
        self.prev_action = 1
        goal = np.array([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5),
                         random.uniform(0.0, 0.5)])
        self.goal = torch.from_numpy(goal).type(torch.FloatTensor)  # randomizing location of goal
        self.buffer_goal = []
        self.sequence = []
        self.failed = 0
        self.variables = np.zeros((self.l_cnt* self.num_d_vars))

    def step(self, a, curr, goal, action):
        global steps_done
        done = 0
        steps_done += 1
        if action > 0:
            if action == self.mod_size - 1:
                done = 1
        # storing previous action
        self.prev_action = action

        # updating the current module with the chosen action
        next_a = copy.deepcopy(a)
        next_a[curr: curr + self.mod_size] = self.actions[action]
        if action == 4:
            next_vars[curr / self.mod_size] = self.link_vars

        # obtaining reward for new arrangement and distance from goal if terminal arrangement
        r, dist, pos  = reward(next_vars, next_a, curr, self.mod_size, goal, action, modules, env)

        if done:
            if dist > self.epsilon:
                self.buffer_goal = torch.from_numpy(np.array(pos)).type(torch.FloatTensor)
                self.failed = 1

        if action == 0:
            mask = torch.zeros(1, self.mod_size, dtype=torch.bool)
            mask[0][0:1] = True
            #print('mask: ', mask)
        else:
            mask = torch.zeros(1, self.mod_size, dtype=torch.bool)
            mask[0][1:self.mod_size] = True
            #print('mask: ', mask)

        return next_a, r, dist, done, mask

    def test_step(self, policy_net, curr):
        #sum = 0
        with torch.no_grad():
            qvals = policy_net.forward(self.state, self.goal, 0).detach().numpy()
        #print('test step values: ', qvals)
        #for i in range(self.mod_size):
        #    qvals[i] = np.exp(qvals[i])
        qvals = masking(qvals, self.mod_size, self.prev_action)
        #print('qvals: ', qvals)
        action = np.argmax(qvals)
        self.prev_action = action
        self.state[curr: curr + self.mod_size] = self.actions[action]
        return action

Transition = namedtuple('Transition',
                        ('state', 'variables','action', 'next_state', 'next_variables','reward', 'goal', 'done', 'mask'))

class ReplayMemory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def validation(env, ep):
    for goal in val_goals:
        #print('for validation goal: ', goal)
        for curr in range(0, len(env.state), env.mod_size):
            action = env.test_step(policy_net, curr)
            if action == env.mod_size - 1:
                final_dist = term_reward(env.state, env.mod_size, goal, curr, modules, env)
                test_a = env.state.numpy()
                #print('')
                #print('arrangement: ', test_a)
                ##print('distance: ', final_dist[0])
                #print('')
                #print('goal: ', goal)
                #print('action: ', action)
                #print('arrangement: ', test_a)
                if goal == val_goals[0]:
                    val_arrangements1.append((ep,test_a))
                    #print('new val arrangements[0]: ', val_arrangements[0])
                    writer.add_scalar('Validation Distance/(.1,.1,.1)', final_dist[0], ep)
                elif goal == val_goals[1]:
                    writer.add_scalar('Validation Distance/(.2,.2,.2)', final_dist[0], ep)
                    val_arrangements2.append((ep,test_a))
                elif goal == val_goals[2]:
                    writer.add_scalar('Validation Distance/(.3,.3,.3)', final_dist[0], ep)
                    val_arrangements3.append((ep,test_a))
                elif goal == val_goals[3]:
                    writer.add_scalar('Validation Distance/(.4,.4,.4)', final_dist[0], ep)
                    val_arrangements4.append((ep,test_a))
                else:
                    writer.add_scalar('Validation Distance/(.5,.5,.5)', final_dist[0], ep)
                    val_arrangements5.append((ep,test_a))
                break
        env.reset()

def optimize_model(buffer, env, policy_net, target_net, optimizer):
    if len(buffer) < env.batch_size:
        return 0
    transitions = buffer.sample(env.batch_size)
    batch = Transition(*zip(*transitions))
    state_batch = torch.cat(batch.state).view(env.batch_size,len(env.state))
    action_batch = torch.cat(batch.action).view(env.batch_size,1)
    goal_batch = torch.cat(batch.goal).view(env.batch_size,len(env.goal))
    mask_batch = torch.cat(batch.mask).view(env.batch_size, env.mod_size)
    state_action_values = policy_net(state_batch, goal_batch, 1).gather(1, action_batch)

    next_state_batch = torch.cat(batch.next_state).view(env.batch_size,len(env.state))
    next_state_vals = target_net(next_state_batch, goal_batch, 1).detach()
    next_state_vals[mask_batch] = -float('inf')

    max_next_state_vals = next_state_vals.max(1)[0]
    done_batch = torch.cat(batch.done)
    reward_batch = torch.cat(batch.reward)

    expected_state_action_values = reward_batch + policy_net.gamma * max_next_state_vals * done_batch

    # Computing loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()

    # back-propagate loss
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    # optimizer step
    optimizer.step()
    scheduler.step()

    return loss.item()

def make_arrangement(env, arrangement_nums):
    arrangement = [''] * int((len(arrangement_nums) / env.mod_size))
    for i in range(int(curr / env.mod_size) + 1):
        mod = arrangement_nums[i * env.mod_size:(i + 1) * env.mod_size]  # current module
        # print('mod: ', mod)
        for j in range(len(mod)):
            val = mod[j]
            if val == 1:
                # trimming values from CSV for modules
                if j <= 2 or j == 5:  # if module is actuator, bracket, or gripper just get first 2 items
                    arrangement[i] = modules[j][0:2]
                else:  # if module is link get first four items
                    arrangement[i] = modules[j][0:4]
                # print('arrangement: ', arrangement)
                break
    return arrangement

if __name__ == '__main__':
    # parameters
    dqn_lr = 4e-3
    DQN_LR_DECAY = 8000
    gamma = 1
    train_episodes = 5000
    test_episodes = 100
    val_goals = [[.1, .1, .1], [.2, .2, .2], [.3, .3, .3], [.4, .4, .4], [.5,.5,.5]]
    val_arrangements1 = []
    val_arrangements2 = []
    val_arrangements3 = []
    val_arrangements4 = []
    val_arrangements5 = []
    env = env(arm_size=8) # initialize environment

    agent = SAC(len(env.state), action_space, len(env.goal), args, env.n_actions, env.l_cnt)

    # initialize policy network where we are frequently updating weights
    policy_net = DQN(len(env.state), env.mod_size,lr,gamma, len(env.goal))

    # initialize target network where we are periodically updating weights
    target_net = DQN(len(env.state), env.mod_size,lr,gamma, len(env.goal))
    target_net.load_state_dict(policy_net.state_dict()) # sets target net equal to policy net to begin
    target_net.eval()

    lambda1 = lambda ep: math.exp(-ep / DQN_LR_DECAY)
    scheduler = optim.lr_scheduler.LambdaLR(policy_net.optimizer, lambda1)

    buffer = ReplayMemory(100)
    # training
    total_loss = 0
    dist = 0
    TARGET_UPDATE = 10
    total_rew = 0
    for ep in range(train_episodes):
        env.reset()
        if ep % 50 == 0:
            validation(env, ep)
        for curr in range(0, len(env.state), env.mod_size):
            action = dqn_choose_action(policy_net, env.state, env.mod_size, env.prev_action, env.goal, ep)
            if action == 4:
                env.link_vars = sac_choose_action(env, action_space, ep, agent)
            next_a, r, dist, done, mask = env.step(env.state,curr, env.goal, action)
            total_rew += r
            #print('reward: ', r)
            #print('total reward: ', total_rew)
            #curr_mask = torch.zeros(1, env.mod_size, dtype=torch.bool)
            #print('PUSHING TO BUFFER:')
            #print('state: ', env.state)
            #print('action: ', torch.tensor(np.array([action])).type(torch.LongTensor))
            #print('next state: ', next_a)
            #print('reward: ', torch.tensor(np.array([r])).type(torch.FloatTensor))
            #print('goal: ', env.goal)
            #print('done: ', torch.tensor(np.array([1 - done])).type(torch.LongTensor))
            buffer.push(env.state, torch.tensor(np.array([action])).type(torch.LongTensor), next_a,
                        torch.tensor(np.array([r])).type(torch.FloatTensor),
                        env.goal, torch.tensor(np.array([1 - done])).type(torch.LongTensor), mask)
            if done:
                if env.failed:
                    #print('')
                    #print('failed')
                    #print('goal point was: ' + str(env.goal) + " while end eff position was " + str(env.buffer_goal))
                    #print('distance was: ', dist)
                    env.sequence.append((env.state, torch.tensor(np.array([action])).type(torch.LongTensor), next_a,
                                        torch.tensor(np.array([1.0])).type(torch.FloatTensor), done, mask))
                    for part in env.sequence:
                        #print('PUSHING TO BUFFER:')
                        #print('state: ', part[0])
                        #print('action: ', part[1])
                        #print('next state: ', part[2])
                        #print('reward: ', part[3])
                        #print('goal: ', env.buffer_goal)
                        #print('done: ', torch.tensor(np.array([1 - part[4]])).type(torch.LongTensor))
                        #print('new goal of: ', env.buffer_goal)
                        buffer.push(part[0], part[1], part[2], part[3], env.buffer_goal, torch.tensor(np.array([1 - part[4]])).type(torch.LongTensor), part[5])
                #policy_net_net.lr = policy_net_net.lr*math.exp(-ep/LR_DECAY)
                #else:
                    #print('succeeded')
                    #print('goal point was: ' + str(env.goal))
                    #print('distance: ', dist)
                writer.add_scalar('Distance/train', dist, ep)
                break
            else:
                env.sequence.append((env.state, torch.tensor(np.array([action])).type(torch.LongTensor), next_a,
                        torch.tensor(np.array([r])).type(torch.FloatTensor), done, mask))
            #for i in range(2):
            l = optimize_model(buffer, env, policy_net, target_net, policy_net.optimizer)
            total_loss += l
            env.state = copy.deepcopy(next_a)

        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        writer.add_scalar('Total Loss/train', total_loss,ep)
        writer.add_scalar('Total rew/train', total_rew, ep)

    # testing
    results = []
    for test in range(test_episodes):
        #print('goal: ', env.goal)
        for curr in range(0,len(env.state), env.mod_size):
            action = env.test_step(target_net,curr)
            #print('action: ', action)
            if action == env.mod_size - 1:
                final_dist = term_reward(env.state, env.mod_size, env.goal, curr, modules, env)
                #print('final distance: ', final_dist[0])
                writer.add_scalar('Distance/test', final_dist[0], test)
                break
        env.reset()
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


