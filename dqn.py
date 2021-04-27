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
from actions import masking
from actions import choose_action
from actions import reward
from actions import term_reward
from pybullet_sim import sim

steps_done = 0

writer = SummaryWriter()
# reading in CSV file of modules and converting to list
with open('mini_mods.csv') as f:
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
        self.goal = torch.from_numpy(goal).type(torch.FloatTensor)
        self.arm_size = arm_size
        self.prev_action = 1
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
        self.epsilon = .05

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

    def step(self, a, curr, goal, action):
        global steps_done
        done = 0
        mask = []
        steps_done += 1
        if action > 0:
            if action == self.mod_size - 1:
                done = 1
        # storing previous action
        self.prev_action = action

        # updating the current module with the chosen action
        next_a = copy.deepcopy(a)
        next_a[curr: curr + self.mod_size] = self.actions[action]

        # obtaining reward for new arrangement and distance from goal if terminal arrangement
        r, dist, pos  = reward(next_a, curr, self.mod_size, goal, action, modules, env)

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
                        ('state', 'action', 'next_state', 'reward', 'goal', 'done', 'mask'))

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

def validation(env):
    for goal in val_goals:
        print('for validation goal: ', goal)
        for curr in range(0, len(env.state), env.mod_size):
            action = env.test_step(policy_net, curr)
            if action == env.mod_size - 1:
                final_dist = term_reward(env.state, env.mod_size, goal, curr, modules, env)
                test_a = env.state.numpy()
                print('')
                print('arrangement: ', test_a)
                print('distance: ', final_dist[0])
                print('')
                if goal == val_goals[0]:
                    val_arrangements[0].append(test_a)
                    writer.add_scalar('Validation Distance/(.1,.1,.1)', final_dist[0], ep)
                elif goal == val_goals[1]:
                    writer.add_scalar('Validation Distance/(.2,.2,.2)', final_dist[0], ep)
                    val_arrangements[1].append(test_a)
                elif goal == val_goals[2]:
                    writer.add_scalar('Validation Distance/(.3,.3,.3)', final_dist[0], ep)
                    val_arrangements[2].append(test_a)
                elif goal == val_goals[3]:
                    writer.add_scalar('Validation Distance/(.4,.4,.4)', final_dist[0], ep)
                    val_arrangements[3].append(test_a)
                else:
                    writer.add_scalar('Validation Distance/(.5,.5,.5)', final_dist[0], ep)
                    val_arrangements[4].append(test_a)
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

if __name__ == '__main__':
    # parameters
    lr = 2e-3
    LR_DECAY = 1000
    gamma = 1
    train_episodes = 500
    test_episodes = 50
    val_goals = [[.1, .1, .1], [.2, .2, .2], [.3, .3, .3], [.4, .4, .4], [.5,.5,.5]]
    val_arrangements = [[]] * len(val_goals)
    env = env(arm_size=8) # initialize environment

    # initialize policy network where we are frequently updating weights
    policy_net = DQN(len(env.state), env.mod_size,lr,gamma, len(env.goal))

    # initialize target network where we are periodically updating weights
    target_net = DQN(len(env.state), env.mod_size,lr,gamma, len(env.goal))
    target_net.load_state_dict(policy_net.state_dict()) # sets target net equal to policy net to begin
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters(), lr=lr)
    lambda1 = lambda ep: math.exp(-ep / LR_DECAY)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda1)

    buffer = ReplayMemory(100)
    # training
    total_loss = 0
    dist = 0
    TARGET_UPDATE = 10
    for ep in range(train_episodes):
        env.reset()
        print('ep: ', ep)
        #print('goal: ', env.goal)
        # every 100 episodes, do validation checks
        if ep % 25 == 0:
            validation(env)
        for curr in range(0, len(env.state), env.mod_size):
            action = choose_action(policy_net, env.state, env.mod_size, env.prev_action, env.goal, ep)
            next_a, r, dist, done, mask = env.step(env.state,curr, env.goal, action)
            #curr_mask = torch.zeros(1, env.mod_size, dtype=torch.bool)
            #if done:
            #    curr_mask[0][0:env.mod_size] = True
            #print('PUSHING TO BUFFER:')
            #print('state: ', env.state)
            #print('action: ', torch.tensor(np.array([action])).type(torch.LongTensor))
            #print('next state: ', next_a)
            #print('reward: ', torch.tensor(np.array([r])).type(torch.FloatTensor))
            #print('goal: ', env.goal)
            #print('done: ', torch.tensor(np.array([1 - done])).type(torch.LongTensor))
            buffer.push(env.state, torch.tensor(np.array([action])).type(torch.LongTensor), next_a,
                        torch.tensor(np.array([r])).type(torch.FloatTensor), env.goal, torch.tensor(np.array([1 - done])).type(torch.LongTensor), mask)
            if done:
                if env.failed:
                    #print('')
                    print('failed')
                    print('goal point was: ' + str(env.goal) + " while end eff position was " + str(env.buffer_goal))
                    print('distance was: ', dist)
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
                #policy_net.lr = policy_net.lr*math.exp(-ep/LR_DECAY)
                else:
                    print('succeeded')
                    print('goal point was: ' + str(env.goal))
                    print('distance: ', dist)
                writer.add_scalar('Distance/train', dist, ep)
                break
            else:
                env.sequence.append((env.state, torch.tensor(np.array([action])).type(torch.LongTensor), next_a,
                        torch.tensor(np.array([r])).type(torch.FloatTensor), done, mask))
            l = optimize_model(buffer, env, policy_net, target_net, optimizer)
            total_loss += l
            env.state = copy.deepcopy(next_a)

        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        writer.add_scalar('Total Loss/train', total_loss,ep)

    # testing
    results = []
    for test in range(test_episodes):
        print('goal: ', env.goal)
        for curr in range(0,len(env.state), env.mod_size):
            action = env.test_step(target_net,curr)
            print('action: ', action)
            if action == env.mod_size - 1:
                final_dist = term_reward(env.state, env.mod_size, env.goal, curr, modules, env)
                print('final distance: ', final_dist[0])
                writer.add_scalar('Distance/test', final_dist[0], test)
                break
        env.reset()

    for i in range(len(val_arrangements)):
        print('goal ', i)
        print(val_arrangements[i])

