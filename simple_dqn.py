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
from torch.utils.tensorboard import SummaryWriter

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
        self.goal_layer = nn.Linear(target_size, 64)
        self.a_layer = nn.Linear(state_size, 64)
        self.model = nn.Sequential(nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32),
                                    nn.ReLU(), nn.Linear(32, action_size))
        self.optimizer = optim.Adam(self.parameters(),lr)
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.gamma = gamma

    def forward(self, state, goal):
        a_res = self.a_layer(state)
        goal_res = self.goal_layer(goal)
        tot_res = torch.cat((a_res, goal_res),0)
        return self.model(tot_res)

class env():
    def __init__(self, arm_size):
        a = np.array([0, 0, 0, 0, 0, 0] * arm_size)
        self.a = torch.from_numpy(a).type(torch.FloatTensor)
        goal = np.array([random.uniform(.25, .5), random.uniform(.25, .5),
                         random.uniform(0.0, .25)])
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
        self.buffer_record = []
        self.buffer_goal = 0
        self.epsilon = .1

    def reset(self):
        a = np.array([0, 0, 0, 0, 0, 0] * self.arm_size)
        self.a = torch.from_numpy(a).type(torch.FloatTensor)
        self.prev_action = 1
        goal = np.array([random.uniform(-.4, .4), random.uniform(-.4, .4),
                         random.uniform(0.0, .25)])
        self.goal = torch.from_numpy(goal).type(torch.FloatTensor)  # randomizing location of goal
        self.buffer_record = []
        self.buffer_goal = 0


    def step(self, model, a, curr, goal):
        global steps_done
        done = 0
        # determining action
        #print('')
        #print('state: ', a)
        action = choose_action(model, a, self.mod_size, self.prev_action, goal, steps_done)
        #print('action: ', action)
        steps_done += 1
        if action == self.mod_size - 1:
            done = 1
        # storing previous action
        self.prev_action = action

        # updating the current module with the chosen action
        next_a = copy.deepcopy(a)
        next_a[curr: curr + self.mod_size] = self.actions[action]
        #print('a: ', a)
        #print('next a: ', next_a)
        # forward pass through DQN for old arrangement
        state_vals = model.forward(a, goal)

        # forward pass through DQN for arrangement with new module
        next_state_vals = model.forward(next_a, goal)

        # obtaining reward for new arrangement and distance from goal if terminal arrangement
        cost, dist, pos  = reward(next_a, curr, self.mod_size, goal, action, modules)
        r = cost
        pos = torch.from_numpy(np.array(pos)).type(torch.FloatTensor)
        # Bellman's equation for updating Q values
        target = r + model.gamma * torch.max(next_state_vals)
        #print('reward: ', r)
        #print('')
        # Computing loss
        loss = model.loss_fn(state_vals[action], target)
        self.a = copy.deepcopy(next_a)
        # keeping track of episode for replay buffer

        # clear old gradients
        model.optimizer.zero_grad()

        # back-propagate loss
        loss.backward()

        # optimizer step
        model.optimizer.step()

        #print('updated vals: ', model.forward(a, goal))
        return loss.item(), dist, done

    def test_step(self, target_net, curr):
        values = target_net.forward(self.a, self.goal).detach().numpy()
        qvals = masking(values, self.mod_size, self.prev_action)
        action = np.argmax(qvals)
        self.prev_action = action
        self.a[curr: curr + self.mod_size] = self.actions[action]
        return action

def print_formatting(test_a, mod_size):
    moduled_output = np.zeros(shape=(int(len(test_a) / mod_size), mod_size))
    for i in range(int(len(test_a) / mod_size)):
        moduled_output[i] = test_a[i * mod_size:(i + 1) * mod_size]
    results = moduled_output.tolist()
    return results

def validation(env):
    for goal in val_goals:
        for curr in range(0, len(env.a), env.mod_size):
            action = env.test_step(target_net, curr)
            if action == env.mod_size - 1:
                final_dist = term_reward(env.a, env.mod_size, goal, curr, modules)
                test_a = env.a.numpy()
                print('')
                print('for validation goal: ', goal)
                print('arrangement: ', test_a)
                print('distance: ', final_dist[0])
                print('')
                if goal == val_goals[0]:
                    writer.add_scalar('Validation Distance/(.1,.1,.1)', final_dist[0], ep)
                elif goal == val_goals[1]:
                    writer.add_scalar('Validation Distance/(.3,.3,.3)', final_dist[0], ep)
                else:
                    writer.add_scalar('Validation Distance/(.5,.5,.5)', final_dist[0], ep)
                break
        env.reset()

if __name__ == '__main__':
    # parameters
    lr = 1e-4
    gamma = .95
    train_episodes = 600
    test_episodes = 50
    val_goals = [[.1, .1, .1], [.3, .3, .3], [.5, .5, .5]]

    env = env(arm_size=8) # initialize environment

    # arrangement set-up
    target_net = DQN(len(env.a), env.mod_size,lr,gamma, len(env.goal)) # initialize network

    # training
    total_loss = 0
    term_times = 0
    dist = 0
    buffer = {}
    buffer_count = 0
    for ep in range(train_episodes):
        print('ep: ', ep)
        if (ep + 100) % 100 == 0:
            validation(env)
        else:
            #print('goal: ', env.goal)
            for curr in range(0, len(env.a), env.mod_size):
                l,dist, done = env.step(target_net, env.a,curr, env.goal)
                total_loss += l
                if dist > 0:
                    term_times += 1
                if done:
                    print('distance: ', dist)
                    #print('buffer count: ', buffer_count)
                    writer.add_scalar('Distance/train', dist, ep)
                    break
        env.reset()
        writer.add_scalar('Total Loss/train', total_loss,ep)

    # testing
    final_dist = (0, 0)
    results = []
    for test in range(test_episodes):
        for curr in range(0,len(env.a), env.mod_size):
            action = env.test_step(target_net,curr)
            if action == env.mod_size - 1:
                final_dist = term_reward(env.a, env.mod_size, env.goal, curr, modules)
                break
        test_a = env.a.numpy()
        results = print_formatting(test_a, env.mod_size)
        env.reset()
        writer.add_scalar('Distance/test', final_dist[0], test)

    #print('results: ', results)
    print('final distance: ', final_dist[0])