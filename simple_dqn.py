import numpy as np
import random
import math
import csv
import pybullet as p
import pybullet_data


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
        self.model = nn.Sequential(nn.ReLU(), nn.Linear(128, 64),
                                    nn.ReLU(), nn.Linear(64, action_size))
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

    def reset(self):
        a = np.array([0, 0, 0, 0, 0, 0] * self.arm_size)
        self.a = torch.from_numpy(a).type(torch.FloatTensor)
        self.prev_action = 1
        goal = np.array([random.uniform(.25, .5), random.uniform(.25, .5),
                         random.uniform(0.0, .25)])
        self.goal = torch.from_numpy(goal).type(torch.FloatTensor)  # randomizing location of goal

    def step(self, model, a, curr, mod_size, goal):
        global steps_done
        done = 0
        # determining action
        action = choose_action(model, a, mod_size, curr, self.prev_action, goal, steps_done)
        steps_done += 1
        if action == mod_size - 1:
            done = 1
        # storing previous action
        self.prev_action = action

        # updating the current module with the chosen action
        next_a = a
        next_a[curr: curr + mod_size] = self.actions[action]

        # forward pass through DQN for old arrangement
        state_vals = model.forward(a, goal)

        # forward pass through DQN for arrangement with new module
        next_state_vals = model.forward(next_a, goal)

        # obtaining reward for new arrangement and distance from goal if terminal arrangement
        cost, dist  = reward(next_a, curr, mod_size, goal, action, modules)
        r = cost
        #print('reward: ', r)
        # Bellman's equation for updating Q values
        target = r + model.gamma * torch.max(next_state_vals)

        # Computing loss
        loss = model.loss_fn(state_vals[action], target)

        # clear old gradients
        model.optimizer.zero_grad()

        # back-propagate loss
        loss.backward()

        # optimizer step
        model.optimizer.step()

        return loss.item(), dist, done

    def test_step(self, target_net, curr,):
        values = target_net.forward(self.a, self.goal).detach().numpy()
        probs = masking(self.a, values, curr, self.mod_size, self.prev_action)
        action = np.argmax(probs)
        self.prev_action = action
        self.a[curr: curr + self.mod_size] = self.actions[action]
        return action

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
    train_episodes = 250
    test_episodes = 50

    #physicsClient = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version, p.DIRECT is faster
    #p.setAdditionalSearchPath(pybullet_data.getDataPath())
    #planeId = p.loadURDF("plane.urdf")

    env = env(arm_size=8) # initialize environment

    # arrangement set-up

    target_net = DQN(len(env.a), env.mod_size,lr,gamma, len(env.goal)) # initialize network

    # training
    total_loss = 0
    total_dist = 0
    term_times = 0
    dist = 0
    for ep in range(train_episodes):
        print('ep: ', ep)
        #print('goal: ', env.goal)
        if ep == 0 or (ep % 100 == 0 and ep > 99):
            #print('here')
            #print('len(env.a): ', len(env.a))
            #print()
            goals= [[.1, .1, .1],
                    [.3, .3, .3],
                    [.5, .5,.5]]
            for goal in goals:
                #print('goal: ', goal)
                prev_action = 1
                for curr in range(0, len(env.a), env.mod_size):
                    action = env.test_step(target_net, curr)
                    #print('env.a: ', env.a)
                    if action == env.mod_size - 1:
                        final_dist = term_reward(env.a, env.mod_size, goal, curr, modules)
                        test_a = env.a.numpy()
                        print('')
                        print('for validation goal: ', goal)
                        print('arrangement: ', test_a)
                        print('distance: ', final_dist[0])
                        print('')
                        break
                env.reset()
        else:
            print('goal: ', env.goal)
            for curr in range(0, len(env.a), env.mod_size):
                #print('curr: ', curr)
                l,dist, done = env.step(target_net, env.a,curr, env.mod_size, env.goal)
                total_loss += l
                total_dist += dist
                if dist > 0:
                    term_times += 1
                if done:
                    print('distance: ', dist)
                    writer.add_scalar('Distance/train', dist, ep)
                    break
        env.reset()
        writer.add_scalar('Total Loss/train', total_loss,ep)

    # testing
    final_dist = (0, 0)
    results = []
    prev_action = 1
    for test in range(test_episodes):
        print('env.goal: ', env.goal)
        for curr in range(0,len(env.a), env.mod_size):
            action = env.test_step(target_net,curr)
            if action == env.mod_size - 1:
                final_dist = term_reward(env.a, env.mod_size, env.goal, curr, modules)
                if test % 5 == 0:
                    print('action: ', action)
                    print('final_dist: ', final_dist[0])
                break
        test_a = env.a.numpy()
        results = print_formatting(test_a, env.mod_size)
        env.reset()
        writer.add_scalar('Distance/test', final_dist[0], test)

    #print('results: ', results)
    print('final distance: ', final_dist[0])