import numpy as np
import queue as q
import random
import math

import torch
from torch import nn
import torch.optim as optim

steps_done = 0

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

    # Will use this to determine next module to add
    def forward(self, state):
        return self.model(state)

    def step(self, a, curr):
        # obtaining values for state and next state
        action = choose_action(self,a, curr)

        #print('current arrangment: ', a)

        #print(action)

        options = [torch.from_numpy(np.array([1, 0, 0, 0, 0])).type(torch.FloatTensor),
                   torch.from_numpy(np.array([0, 1, 0, 0, 0])).type(torch.FloatTensor),
                   torch.from_numpy(np.array([0, 0, 1, 0, 0])).type(torch.FloatTensor),
                   torch.from_numpy(np.array([0, 0, 0, 1, 0])).type(torch.FloatTensor),
                   torch.from_numpy(np.array([0, 0, 0, 0, 1])).type(torch.FloatTensor)]
        mod_size = len(options)
        next_a = a
        next_a[curr: curr + mod_size] = options[action]
        #print('next arrangement: ', next_a)

        state_vals = self.forward(a) # has to go after changing next_a for some reason


        next_state_vals = self.forward(next_a)

        #print('current values: ', state_vals)
        #print('next values: ', next_state_vals)


        r = reward(next_a, curr)
        #print('reward for next a: ', r)
        target = r + self.gamma * torch.max(next_state_vals)
        #print('')
        loss = self.loss_fn(state_vals[action], target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def choose_action(network,a,curr):
    global steps_done
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    values = network.forward(a).detach().numpy()
    #print('actio values: ',network.forward(a))
    '''    if curr == 0:
        return random.randrange(0, 5)
    if sample > eps_threshold:  # using policy
        with torch.no_grad():
            value = np.argmax(values[0:5])
            print('policy action: ', value)
            return value
    else:  # picking at random
        value = random.randrange(0, 5)
        print('random action: ', value)
        return value
'''

    probs = [0] * mod_size
    sum = 0
    t = 0.5
    for i in range(mod_size):
        sum += np.exp(values[i]/t)

    for i in range(mod_size):
        probs[i] = np.exp(values[i]/t)/sum
    act = np.random.choice([0,1,2,3,4],p=probs)
    #print('action: ', act)
    return act


# reward function
def reward(a, curr):
    #print('a: ', a)
    values = a.numpy()
    if curr == len(a) - mod_size:
        for i in range(len(values)):
            if i == mod_size:
                break
            count = 0
            while i < len(values):
                count += values[i]
                i = i + mod_size
            if count == int(len(a) / mod_size):
                return 1
        return -1
    else:
        return 0

if __name__ == '__main__':
    mod1 = torch.from_numpy(np.array([1, 0, 0, 0, 0])).type(torch.FloatTensor) # joint
    mod2 = torch.from_numpy(np.array([0, 1, 0, 0, 0])).type(torch.FloatTensor) # actuator
    mod3 = torch.from_numpy(np.array([0, 0, 1, 0, 0])).type(torch.FloatTensor) # link
    mod4 = torch.from_numpy(np.array([0, 0, 0, 1, 0])).type(torch.FloatTensor) # bracket
    mod5 = torch.from_numpy(np.array([0, 0, 0, 0, 1])).type(torch.FloatTensor) # gripper

    actions = [mod1,mod2,mod3, mod4, mod5]
    mod_size = len(actions)
    empty = np.array([0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0])
    a = torch.from_numpy(empty).type(torch.FloatTensor)
    lr = 1e-3
    gamma = .95
    episodes = 1000
    test_episodes = 100
    target_net = DQN(len(a), mod_size,lr,gamma)

    for ep in range(episodes):
        for curr in range(0, len(a), mod_size):
            l = target_net.step(a,curr)
        a = torch.from_numpy(empty).type(torch.FloatTensor)


    # testing
    test_a = torch.from_numpy(empty).type(torch.FloatTensor)
    results = []
    for test in range(test_episodes):
        for curr in range(0,len(test_a), mod_size):
            if curr == 0:
                action = random.randrange(0, 5)
            else:
                values = target_net.forward(test_a).detach().numpy()
                arrangement = test_a.numpy()
                action = np.argmax(values[0:5])
            test_a[curr: curr + mod_size] = actions[action]
        test_a = test_a.numpy()
        #print('test_a: ', test_a)
        #print('len(test_a): ', len(test_a))
        #print('mod_size: ', mod_size)
        moduled_output = np.zeros(shape=(int(len(test_a)/mod_size),mod_size))
        for i in range(int(len(test_a) / mod_size)):
            moduled_output[i] = test_a[i*mod_size:(i + 1)*mod_size]
        results.append(moduled_output.tolist())
        test_a = torch.from_numpy(empty).type(torch.FloatTensor)
    for result in results:
        print(result)






