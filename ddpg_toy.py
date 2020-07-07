import os
import csv
import copy
import math
from collections import namedtuple
import random
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ddpg_create_xml import make_xml
from pybullet_sim import sim
from torch.utils.tensorboard import SummaryWriter
from ddpg_actions import choose_action
from ddpg_actions import reward
from ddpg_actions import term_reward

writer = SummaryWriter()

class OUActionNoise(object):
    def __init__(self,mu, theta=0.15, sigma=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 =x0
        self.x_prev = None
        self.reset()

    def __call__(self, ep):
        # can just call object to get noise
        #x = self.x_prev + self.theta*(self.x_prev)*self.dt + self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        # this is how you create temporal correlations
        #self.x_prev = x
        length_noise = np.random.uniform(-0.15, 0.15)*math.exp(-ep/LR_DECAY)
        twist_noise = np.random.uniform(-1.0,1.0)*math.exp(-ep/LR_DECAY)
        return (length_noise, twist_noise)

    def reset(self):
       self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

Transition = namedtuple('Transition',
                        ('state', 'variables','next_variables', 'action', 'reward', 'goal', 'done'))
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
#
class Actor(nn.Module):
    def __init__(self, input_size, goal_size, var_size, n_actions, fc1_dims, fc2_dims, fc3_dims, fc4_dims, fc5_dims, a_dims, g_dims, v_dims, lr_actor, action_bound):
        super(Actor, self).__init__()
        self.input_size = input_size
        self.goal_size = goal_size
        self.var_size = var_size
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims
        self.g_dims = g_dims
        self.a_dims = a_dims
        self.v_dims = v_dims
        self.lr = lr_actor
        self.action_bound = action_bound

        self.fc1 = nn.Linear(self.a_dims + self.g_dims + self.v_dims, self.fc1_dims)
        #f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        #T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        #T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        #f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        #T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        #T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)

        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)

        self.fc5 = nn.Linear(self.fc4_dims, self.fc5_dims)

        self.g1 = nn.Linear(self.goal_size, self.g_dims)
        self.a1 = nn.Linear(self.input_size, self.a_dims)
        self.v1 = nn.Linear(self.var_size, self.v_dims)

        #f3 = 0.003
        # mu is our action policy
        self.mu = nn.Linear(self.fc5_dims, self.n_actions)
        #T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        #T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr = lr_actor)
        # need a device? configure GPU

    def forward(self, state, variables, goal, batch):
        goal_output = self.g1(goal)
        state_output = self.a1(state)
        # print('variables: ', variables)
        if batch:
            tensor_variables = variables
            link_vals = T.from_numpy(np.zeros((len(variables), self.v_dims))).type(T.FloatTensor)
        else:
            tensor_variables = T.from_numpy(variables).type(T.FloatTensor)
            link_vals = T.from_numpy(np.array([0.0]*self.v_dims)).type(T.FloatTensor)
        link_vals = T.add(link_vals, self.v1(tensor_variables))
        if batch:
            concat_state = T.cat((state_output, goal_output, link_vals), 1)
        else:
            concat_state = T.cat((state_output, goal_output, link_vals), 0)
        #concat_state = F.relu(concat_state)
        output1 = self.fc1(concat_state)
        output1 = F.relu(output1)
        output2 = self.fc2(output1)
        output2 = F.relu(output2)
        output3 = self.fc3(output2)
        output3 = F.relu(output3)
        output4 = self.fc4(output3)
        output4 = F.relu(output4)
        output5 = self.fc5(output4)
        output5 = F.relu(output5)
        # using sigmoid to get it between 0 and 1
        m = nn.Sigmoid()
        final_output = m(self.mu(output5))
        # final_output = self.model(concat_state)
        #print('non bounded final_output: ', final_output)
        final_output = T.mul(final_output, T.tensor(self.action_bound))
        #print('bounded output: ', final_output)
        return final_output

class Critic(nn.Module):
    def __init__(self, input_size, goal_size, var_size, n_actions, fc1_dims, fc2_dims, fc3_dims, fc4_dims, fc5_dims, a_dims, g_dims, v_dims, lr_critic):
        super(Critic, self).__init__()
        self.input_size = input_size
        self.goal_size = goal_size
        self.var_size = var_size
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims
        self.lr = lr_critic
        self.g_dims = g_dims
        self.a_dims = a_dims
        self.v_dims = v_dims

        self.fc1 = nn.Linear(self.a_dims + self.g_dims + self.v_dims, self.fc1_dims)
        #f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        #T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        #T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        #f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        #T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        #T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)

        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)

        self.fc5 = nn.Linear(self.fc4_dims, self.fc5_dims)

        self.g1 = nn.Linear(self.goal_size, self.g_dims)
        self.a1 = nn.Linear(self.input_size, self.a_dims)
        self.v1 = nn.Linear(self.var_size, self.v_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc5_dims)
        # f3 = 0.003
        # scalar value, only one output (value of the action chosen)
        self.q = nn.Linear(self.fc5_dims, 1)
        #.nn.init.uniform_(self.q.weight.data, -f3, f3)
        #T.nn.init.uniform_(self.q.bias.data, -f3, f3)

        #self.state_model = nn.Sequential(self.fc1, nn.ReLU(), self.fc2)
        #self.action_model = nn.Sequential(self.action_value, nn.ReLU())
        self.optimizer = optim.Adam(self.parameters(), lr = lr_critic)

        # self device??

    def forward(self, state, variables, goal, action, batch):
        goal_output = self.g1(goal)
        state_output = self.a1(state)
        # print('variables: ', variables)
        if batch:
            tensor_variables = variables
            link_vals = T.from_numpy(np.zeros((len(variables), self.v_dims))).type(T.FloatTensor)
        else:
            tensor_variables = T.from_numpy(variables).type(T.FloatTensor)
            link_vals = T.from_numpy(np.array([0.0]*self.v_dims)).type(T.FloatTensor)
        link_vals = T.add(link_vals, self.v1(tensor_variables))
        if batch:
            concat_state = T.cat((state_output, goal_output, link_vals), 1)
        else:
            concat_state = T.cat((state_output, goal_output, link_vals), 0)
        concat_state = F.relu(concat_state) # added
        state_value = self.fc1(concat_state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc3(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc4(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc5(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

class env():
    def __init__(self, arm_size, n_actions, action_bounds, num_d_vars):
        a = np.array([0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0])
        finished_a = np.array([1, 0, 0, 0, 0,
                      0, 0, 0, 1, 0,
                      1, 0, 0, 0, 0,
                      0, 0, 0, 1, 0,
                      1, 0, 0, 0, 0,
                      0, 0, 0, 1, 0,
                      1, 0, 0, 0, 0,
                      0, 0, 0, 0, 1])
        self.a_add = [T.from_numpy(np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0])).type(T.FloatTensor),
                 T.from_numpy(np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0])).type(T.FloatTensor),
                 T.from_numpy(np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0])).type(T.FloatTensor),
                 T.from_numpy(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])).type(T.FloatTensor)]
        self.state = T.from_numpy(a).type(T.FloatTensor)
        self.finished_state = T.from_numpy(finished_a).type(T.FloatTensor)
        self.variables = []
        goal = np.array([random.uniform(0.3, 0.4), random.uniform(0.3, 0.4),
                         random.uniform(0.3, 0.4)])
        self.goal = T.from_numpy(goal).type(T.FloatTensor)
        self.arm_size = arm_size
        self.num_d_vars = num_d_vars
        self.prev_action = 1
        self.n_actions = n_actions
        self.action_bounds = action_bounds
        # module list
        self.actions = [T.from_numpy(np.array([1, 0, 0, 0, 0])).type(T.FloatTensor),
                        T.from_numpy(np.array([0, 1, 0, 0, 0])).type(T.FloatTensor),
                        T.from_numpy(np.array([0, 0, 1, 0, 0])).type(T.FloatTensor),
                        T.from_numpy(np.array([0, 0, 0, 1, 0])).type(T.FloatTensor),
                        T.from_numpy(np.array([0, 0, 0, 0, 1])).type(T.FloatTensor)]
        self.mod_size = len(self.actions)
        self.batch_size = 25

        self.buffer_goal = []
        self.failed = 0
        self.sequence = []
        self.epsilon = .05
        self.gamma = 1
        self.l_cnt = 0
        self.active_l_cnt = 0
        self.tau = 0.001

    def reset(self):
        a = np.array([0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0])
        self.state = T.from_numpy(a).type(T.FloatTensor)
        self.prev_action = 1
        self.variables = np.zeros((self.l_cnt* self.num_d_vars))
        goal = np.array([random.uniform(0.3, 0.4), random.uniform(0.3, 0.4),
                         random.uniform(0.3, 0.4)])
        self.goal = T.from_numpy(goal).type(T.FloatTensor)  # randomizing location of goal
        self.buffer_goal = []
        self.sequence = []
        self.failed = 0
        self.active_l_cnt = 0

    def step(self, curr, action):
        done = 0
        if self.active_l_cnt == self.l_cnt:
            done = 1
        next_variables = copy.deepcopy(self.variables)
        next_variables[(self.active_l_cnt - 1)*self.num_d_vars:(self.active_l_cnt)*self.num_d_vars] = np.array(action)
        dist, rew, end_eff_pos = reward(self, curr, next_variables)

        if done:
            if dist > self.epsilon:
                self.buffer_goal = T.from_numpy(np.array(end_eff_pos)).type(T.FloatTensor)
                self.failed = 1

        return next_variables, rew, dist, done

    def test_step(self, actor, state, variables, goal):
        actor.eval()
        action = actor(state, variables, goal, 0).detach().numpy()
        self.variables[(self.active_l_cnt - 1) * self.num_d_vars:(self.active_l_cnt) * self.num_d_vars] = np.array(action)

def optimize_model(memory, env, actor, target_actor, critic, target_critic):
    if memory.position < env.batch_size:
        return
    transitions = memory.sample(env.batch_size)
    batch = Transition(*zip(*transitions))
    state_batch = T.cat(batch.state).view(env.batch_size,len(env.state))
    variables_batch = T.cat(batch.variables).view(env.batch_size, env.l_cnt*env.num_d_vars)
    next_variables_batch = T.cat(batch.next_variables).view(env.batch_size, env.l_cnt*env.num_d_vars)
    action_batch = T.cat(batch.action)
    reward_batch = T.cat(batch.reward).view(env.batch_size, 1)
    goal_batch = T.cat(batch.goal).view(env.batch_size, len(env.goal))
    done_batch = T.cat(batch.done).view(env.batch_size, 1)

    print('state batch: ', state_batch)
    print('action batch: ', action_batch)
    print('variables batch: ', variables_batch)
    print('reward batch: ', reward_batch)
    print('next variables batch: ', next_variables_batch)


    target_actor.eval()
    target_critic.eval()
    critic.eval()

    # calculate target actions from bellman's equation?
    target_actions = target_actor.forward(state_batch, next_variables_batch, goal_batch, 1)

    # for the new state, getting target action actions from target actor network
    # what actions should target critic take based on target actor's estimates of actions
    target_critic_value = target_critic.forward(state_batch, next_variables_batch, goal_batch, target_actions, 1)
    # what was the estimate of the values of the states and actions we encountered in our sample of replay buffer
    critic_value = critic.forward(state_batch, variables_batch, goal_batch, action_batch, 1)

    # calculating our target values using Bellman's Equation
    target = reward_batch + env.gamma * target_critic_value * done_batch

    # already performed evaluations, now we want to calculate loss
    critic.train()

    # whenever we calculate loss, want to zero gradients so gradients from
    # prev. steps don't accumulate and interfere with current calculations
    # Critic loss
    critic.optimizer.zero_grad()
    critic_loss = F.mse_loss(target, critic_value)
    critic_loss.backward()
    critic.optimizer.step()
    critic_scheduler.step()

    # setting to evaluation mode for calculation for loss for actor network
    critic.eval()

    # Actor loss
    actor.optimizer.zero_grad()
    mu = actor.forward(state_batch, variables_batch, goal_batch, 1)
    actor.train()

    # implicitly calculates chain rule for actor/critic derivatives
    actor_loss = T.mean(-critic.forward(state_batch, variables_batch, goal_batch, mu, 1))
    actor_loss.backward()
    actor.optimizer.step()
    actor_scheduler.step()

    soft_update_network_parameters(actor, target_actor, critic, target_critic, env.tau)

def soft_update_network_parameters(actor, target_actor, critic, target_critic, tau):
    # tau is param that lets update of target network to gradually approach the evaluation networks
    # good for nice and slow convergence

    actor_params = actor.named_parameters()
    critic_params = critic.named_parameters()
    target_actor_params = target_actor.named_parameters()
    target_critic_params = target_critic.named_parameters()

    critic_state_dict = dict(critic_params)
    actor_state_dict = dict(actor_params)
    target_critic_dict = dict(target_critic_params)
    target_actor_dict = dict(target_actor_params)

    for name in critic_state_dict:
        critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1 - tau)*target_critic_dict[name].clone()

    # iterates over dictionary, looks at key, and updates values from this network
    target_critic.load_state_dict(critic_state_dict)

    for name in actor_state_dict:
        actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1 - tau)*target_actor_dict[name].clone()

    # iterates over dictionary, looks at key, and updates values from this network
    target_actor.load_state_dict(actor_state_dict)

def hard_update_network_parameters(actor, target_actor, critic, target_critic):
    # hard update version
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

if __name__ == '__main__':
    TARGET_UPDATE = 10
    LR_DECAY = 1500

    buffer = ReplayMemory(100)
    env = env(arm_size=8, n_actions=2, action_bounds= [.75,2*np.pi], num_d_vars=2)
    noise = OUActionNoise(mu=np.zeros(env.n_actions))

    #for curr in range(0, len(env.state), env.mod_size):
    #    mod = env.state[curr:curr + env.mod_size]
    #    if mod[-2] == 1:
    #        env.l_cnt += 1
    env.l_cnt = 3
    n_actions = 2
    var_size = n_actions * env.l_cnt
    lr_actor = .0001
    lr_critic = .001
    train_episodes = 1000
    test_episodes = 50

    actor = Actor(input_size=len(env.state), goal_size=len(env.goal), var_size=var_size, n_actions=n_actions,fc1_dims=400,
                  fc2_dims=300, fc3_dims=128, fc4_dims=32, fc5_dims=8, a_dims= 64, g_dims= 9, v_dims= 12, lr_actor=lr_actor,action_bound=[.75,2*np.pi])
    target_actor = Actor(input_size=len(env.state), goal_size=len(env.goal), var_size=var_size, n_actions=n_actions,fc1_dims=400,
                         fc2_dims=300, fc3_dims=128, fc4_dims=32, fc5_dims=8, a_dims= 64, g_dims= 9, v_dims= 12, lr_actor=lr_actor,action_bound=[.75,2*np.pi])
    critic = Critic(input_size=len(env.state), goal_size=len(env.goal), var_size=var_size, n_actions=n_actions,fc1_dims=400,
                    fc2_dims=300, fc3_dims=128, fc4_dims=32, fc5_dims=8, a_dims= 64, g_dims= 9, v_dims= 12, lr_critic=lr_critic)
    target_critic = Critic(input_size=len(env.state), goal_size=len(env.goal), var_size=var_size, n_actions=n_actions,fc1_dims=400,
                           fc2_dims=300, fc3_dims=128, fc4_dims=32, fc5_dims=8, a_dims= 64, g_dims= 9, v_dims= 12, lr_critic=lr_critic)

    lambda1 = lambda ep: math.exp(-ep / LR_DECAY)
    actor_scheduler = optim.lr_scheduler.LambdaLR(actor.optimizer, lambda1)
    critic_scheduler = optim.lr_scheduler.LambdaLR(critic.optimizer, lambda1)

    # initialize networks to have the same features
    soft_update_network_parameters(actor, target_actor, critic, target_critic, tau=1)

    for ep in range(train_episodes):
        noise.reset()
        env.reset()
        print('episode: ', ep)
        for curr in range(0, len(env.state), env.mod_size):
            #print('curr: ', curr)
            #print('len of state', len(env.state))
            # only want to call step if we are at a link
            mod = env.finished_state[curr:curr + env.mod_size]
            #print('mod: ', mod)
            if mod[-2] != 1:
                continue
            #print('state before: ', env.state)
            #print('env.a_add[env.active_l_cnt]: ', env.a_add[env.active_l_cnt])
            env.state[curr - env.mod_size: curr + env.mod_size] = env.a_add[env.active_l_cnt]
            env.active_l_cnt += 1
            if env.active_l_cnt == env.l_cnt:
                env.state[curr + env.mod_size: curr + 3 * env.mod_size] = env.a_add[env.active_l_cnt]
            #print('state after: ', env.state)

            action = choose_action(actor, env.state, env.variables, env.goal, env, noise, ep)
            next_variables, rew, dist, done = env.step(curr, action)
            print('PUSHING TO BUFFER')
            print('state: ', env.state)
            print('variables: ', T.tensor(np.array(env.variables)).type(T.FloatTensor))
            print('next variables: ', T.tensor(np.array(next_variables)).type(T.FloatTensor))
            print('action: ', T.tensor(np.array([action])).type(T.FloatTensor))
            print('rew: ', T.tensor(np.array([rew])).type(T.FloatTensor))
            print('goal: ', env.goal)
            print('1 - done: ', T.tensor(np.array([1 - done])).type(T.LongTensor))
            buffer.push(env.state, T.tensor(np.array(env.variables)).type(T.FloatTensor), T.tensor(np.array(next_variables)).type(T.FloatTensor), T.tensor(np.array([action])).type(T.FloatTensor),
                        T.tensor(np.array([rew])).type(T.FloatTensor), env.goal,
                        T.tensor(np.array([1 - done])).type(T.LongTensor))
            if done:
                if env.failed:
                    print('failed')
                    print('goal point was: ' + str(env.goal) + " while end eff position was " + str(env.buffer_goal))
                    print('distance was: ', dist)
                    env.sequence.append((env.state,  T.tensor(np.array(env.variables)).type(T.FloatTensor),
                                         T.tensor(np.array(next_variables)).type(T.FloatTensor),
                                         T.tensor(np.array([action])).type(T.FloatTensor),
                                         T.tensor(np.array([1.0])).type(T.FloatTensor),
                                         T.tensor(np.array([1 - done])).type(T.LongTensor)))
                    print('PUSHING TO BUFFER FROM FAILED')
                    for part in env.sequence:
                        print('state: ', part[0])
                        print('variables: ', part[1])
                        print('next variables: ', part[2])
                        print('action: ', part[3])
                        print('rew: ', part[4])
                        print('goal: ', env.buffer_goal)
                        print('1 - done: ', part[5])
                        buffer.push(part[0], part[1], part[2], part[3], part[4], env.buffer_goal, part[5])
                else:
                    print('succeeded')
                    print('goal point was: ' + str(env.goal))
                    print('distance: ', dist)
                writer.add_scalar('Distance/train', dist, ep)
                break
            else:
                env.sequence.append((env.state,  T.tensor(np.array(env.variables)).type(T.FloatTensor),
                                     T.tensor(np.array(next_variables)).type(T.FloatTensor),
                                     T.tensor(np.array([action])).type(T.FloatTensor),
                                     T.tensor(np.array([rew])).type(T.FloatTensor),
                                     T.tensor(np.array([1 - done])).type(T.LongTensor)))
            optimize_model(buffer,env, actor, target_actor, critic, target_critic)
            env.variables = copy.deepcopy(next_variables)
            #print('env variables copied as: ', env.variables)
        if ep % TARGET_UPDATE == 0:
            hard_update_network_parameters(actor,target_actor,critic,target_critic)

    results = []
    for test in range(test_episodes):
        env.reset()
        #print('goal: ', env.goal)
        for curr in range(0, len(env.state), env.mod_size):
            mod = env.state[curr:curr + env.mod_size]
            if mod[-2] != 1:
                continue
            env.active_l_cnt += 1
            env.test_step(actor, env.state, env.variables, env.goal)
            if env.active_l_cnt == env.l_cnt:
                dist, rew, end_eff_pos = term_reward(env, curr, env.variables)
                print('final distance: ', dist)
                writer.add_scalar('Distance/test', dist, test)
                break
