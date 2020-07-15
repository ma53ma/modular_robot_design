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
from toy_td3_actions import choose_action
from toy_td3_actions import reward
from toy_td3_actions import term_reward

writer = SummaryWriter()

class ActionNoise(object):
    def __init__(self):
        self.noise_decay = 900

    def __call__(self, ep, actor, critic):
        # can just call object to get noise
        #x = self.x_prev + self.theta*(self.x_prev)*self.dt + self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        # this is how you create temporal correlations
        #self.x_prev = x
        #if .03333*math.exp(-(ep - explore_episodes)/self.noise_decay) < .03333:
        #    length_std = .03333*math.exp(-(ep - explore_episodes)/self.noise_decay)
        #else:
        #    length_std = .03333
        '''
        length_std = min(.03333*math.exp(-(ep - explore_episodes)/self.noise_decay),.03333)
        length_noise = np.random.normal(0,length_std,1)
        twist_std = min(.13333*math.exp(-(ep - explore_episodes)/self.noise_decay),.13333)
        #print('length noise', length_noise)
        #if .13333*math.exp(-(ep - explore_episodes)/self.noise_decay) < .13333:
        #    twist_std = .13333*math.exp(-(ep - explore_episodes)/self.noise_decay)
        #else:
        #    twist_std = .13333
        twist_noise = np.random.normal(0,twist_std,1)
        return np.array([length_noise, twist_noise])
        '''

        noise_modifier = min(math.exp(-(ep - explore_episodes)/self.noise_decay),1)
        with T.no_grad():
            for param in actor.parameters():
                #print('actor param before: ', param)
                actor_noise = T.randn(param.size()) * 0.05*noise_modifier
                #print('actor noise: ', actor_noise)
                param.add_(actor_noise)
                #print('actor param after: ', param)
            for param in critic.parameters():
                #print('critic param: ', param)
                critic_noise = T.randn(param.size()) * 0.05*noise_modifier
                param.add_(critic_noise)
                #print('critic noise: ', critic_noise)

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
        #print('position: ', self.position)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
#
class Actor(nn.Module):
    def __init__(self, input_size, goal_size, var_size, n_actions, fc1_dims, fc2_dims, a_dims, v_dims, lr_actor, action_bound):
        super(Actor, self).__init__()
        self.input_size = input_size
        self.goal_size = goal_size
        self.var_size = var_size
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.a_dims = a_dims
        self.pre_process_dims = v_dims
        self.lr = lr_actor
        self.init_lr = lr_actor
        self.action_bound = action_bound

        self.var_pre_process = nn.Linear(self.n_actions, self.pre_process_dims)
        self.bn1 = nn.LayerNorm(self.pre_process_dims)

        self.fc1 = nn.Linear(self.a_dims + self.goal_size + self.pre_process_dims*self.n_actions, self.fc1_dims)
        #f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        #T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        #T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        #f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        #T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        #T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.a1 = nn.Linear(self.input_size, self.a_dims)

        #f3 = 0.003
        # mu is our action policy
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        #T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        #T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr = lr_actor)
        # need a device? configure GPU

    def forward(self, state, variables, goal, batch, env):
        if batch:
            tensor_variables = variables
            pre_process_vars = T.from_numpy(np.zeros((env.batch_size, env.l_cnt * self.pre_process_dims))).type(
                T.FloatTensor)
            for i in range(0, env.num_d_vars * env.l_cnt, env.num_d_vars):
                d_var_set = tensor_variables[:, i:i + env.num_d_vars]
                mask = T.tensor(tuple(map(lambda s: s[0] > 0,
                                          d_var_set)), dtype=T.bool)
                output = self.var_pre_process(d_var_set)
                good_pre_process_vars = T.from_numpy(np.zeros((env.batch_size, self.pre_process_dims))).type(
                    T.FloatTensor)
                good_pre_process_vars[mask] = output[mask]
                pre_process_vars[:, i * int(self.pre_process_dims / self.n_actions):i * int(
                    self.pre_process_dims / self.n_actions) + self.pre_process_dims] = good_pre_process_vars
        else:
            tensor_variables = T.from_numpy(variables).type(T.FloatTensor)
            pre_process_vars = T.from_numpy(np.array([0.0] * (env.l_cnt * self.pre_process_dims))).type(T.FloatTensor)
            for i in range(0, env.num_d_vars * env.l_cnt, env.num_d_vars):
                d_var_set = tensor_variables[i:i + env.num_d_vars]
                output = self.var_pre_process(d_var_set)
                if d_var_set[0] > 0:
                    pre_process_vars[i * int(self.pre_process_dims / self.n_actions):i * int(
                        self.pre_process_dims / self.n_actions) + self.pre_process_dims] = output
        state_output = self.a1(state)
        if batch:
            concat_state = T.cat((state_output, goal, pre_process_vars), 1)
        else:
            concat_state = T.cat((state_output, goal, pre_process_vars), 0)
        #concat_state = F.relu(concat_state)
        output1 = self.fc1(concat_state)
        output1 = F.relu(output1)
        output2 = self.fc2(output1)
        output2 = F.relu(output2)
        # using sigmoid to get it between 0 and 1
        output = self.mu(output2)
        final_output = T.sigmoid(output)
        #if not batch:
            #print('pre sigmoid action: ', final_output)
            #print('state: ', state)
            #print('state output: ', state_output)
            #print('goal: ', goal)
            #print('pre_process_vars: ', pre_process_vars)
        #final_output = m(action)
        #if not batch:
        #    print('final output: ', final_output)
        final_output = T.mul(final_output, T.tensor(self.action_bound))
        #print('goal: ', goal[0])
        if not batch and (goal[0] - .2 == 0 or goal[0] - .3 == 0 or goal[0] - .4 == 0):
            print('mu output: ', output)
            if env.variables[0] == 0:
                writer.add_scalar('Mu length 1 output/' + str(goal), output[0], ep)
                writer.add_scalar('Mu twist 1 output/' + str(goal), output[1], ep)
            else:
                writer.add_scalar('Mu length 2 output/' + str(goal), output[0], ep)
                writer.add_scalar('Mu twist 2 output/' + str(goal), output[1], ep)
            print('bounded output: ', final_output)
        return final_output

class Critic(nn.Module):
    def __init__(self, input_size, goal_size, var_size, n_actions, fc1_dims, fc2_dims, a_dims, v_dims, lr_critic):
        super(Critic, self).__init__()
        self.input_size = input_size
        self.goal_size = goal_size
        self.var_size = var_size
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.pre_process_dims = v_dims
        self.lr = lr_critic
        self.init_lr = lr_critic
        self.a_dims = a_dims

        self.var_pre_process = nn.Linear(self.n_actions, self.pre_process_dims)

        self.fc1 = nn.Linear(self.a_dims + self.goal_size + self.pre_process_dims*self.n_actions + self.n_actions, self.fc1_dims)
        #f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        #T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        #T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.fc2 = nn.Linear(self.fc1_dims + self.n_actions, self.fc2_dims)
        #f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        #T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        #T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        #self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)

        #self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)

        #self.fc5 = nn.Linear(self.fc4_dims, self.fc5_dims)

        self.a1 = nn.Linear(self.input_size, self.a_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        # f3 = 0.003
        # scalar value, only one output (value of the action chosen)
        self.q = nn.Linear(self.fc2_dims, 1)
        #.nn.init.uniform_(self.q.weight.data, -f3, f3)
        #T.nn.init.uniform_(self.q.bias.data, -f3, f3)

        #self.state_model = nn.Sequential(self.fc1, nn.ReLU(), self.fc2)
        #self.action_model = nn.Sequential(self.action_value, nn.ReLU())
        self.optimizer = optim.Adam(self.parameters(), lr = lr_critic)

    def forward(self, state, variables, goal, action, batch, env):
        if batch:
            tensor_variables = variables
            pre_process_vars = T.from_numpy(np.zeros((env.batch_size, env.l_cnt * self.pre_process_dims))).type(
                T.FloatTensor)
            for i in range(0, env.num_d_vars * env.l_cnt, env.num_d_vars):
                # print(i)
                # print('env.num_d_vars: ', env.num_d_vars)
                # print(len(variables))
                #print('tensor variables: ', tensor_variables)
                d_var_set = tensor_variables[:, i:i + env.num_d_vars]
                #print('d_var_set: ', d_var_set)
                mask = T.tensor(tuple(map(lambda s: s[0] > 0,
                                          d_var_set)), dtype=T.bool)
                #print('mask: ', mask)
                # if d_var_set[0] > 0:
                output = self.var_pre_process(d_var_set)
                #print('output: ', output)
                good_pre_process_vars = T.from_numpy(np.zeros((env.batch_size, self.pre_process_dims))).type(
                    T.FloatTensor)
                # print(good_pre_process_vars)
                good_pre_process_vars[mask] = output[mask]
                #print('good_pre_process_vars', good_pre_process_vars)
                pre_process_vars[:, i * int(self.pre_process_dims / self.n_actions):i * int(
                    self.pre_process_dims / self.n_actions) + self.pre_process_dims] = good_pre_process_vars
        else:
            tensor_variables = T.from_numpy(variables).type(T.FloatTensor)
            pre_process_vars = T.from_numpy(np.array([0.0] * (env.l_cnt * self.pre_process_dims))).type(T.FloatTensor)
            #print('pre process vars before: ', pre_process_vars)
            for i in range(0, env.num_d_vars * env.l_cnt, env.num_d_vars):
                #print('tensor variables: ', tensor_variables)
                d_var_set = tensor_variables[i:i + env.num_d_vars]
                output = self.var_pre_process(d_var_set)
                if d_var_set[0] > 0:
                    pre_process_vars[i * int(self.pre_process_dims / self.n_actions):i * int(
                        self.pre_process_dims / self.n_actions) + self.pre_process_dims] = output
                #print('pre process vars after: ', pre_process_vars)
        #goal_output = self.g1(goal)
        state_output = self.a1(state)
        if batch:
            concat_state = T.cat((state_output, goal, pre_process_vars, action), 1)
        else:
            concat_state = T.cat((state_output, goal, pre_process_vars, action), 0)
        # concat_state = F.relu(concat_state) # added
        state_value = self.fc1(concat_state)
        state_value = F.relu(state_value)
        if batch:
            action_state_concat = T.cat((state_value, action), 1)
        else:
            action_state_concat = T.cat((state_value, action), 0)
        #state_value = self.fc2(state_value)
        #state_value = F.relu(state_value)
        #state_value = self.fc3(state_value)
        #state_value = F.relu(state_value)
        #state_value = self.fc4(state_value)
        #state_value = F.relu(state_value)
        #state_value = self.fc5(state_value)

        #action_value = F.relu(self.action_value(action))
        #print('state value: ', state_value)
        #print('action: ', action)
        state_action_value = F.relu(self.fc2(action_state_concat))
        #state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)
        #print('q value: ', state_action_value.view(env.batch_size))

        return state_action_value

class env():
    def __init__(self, arm_size, n_actions, action_bounds, num_d_vars):
        a = np.array([0, 0, 0, 0, 0,
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
                      0, 0, 0, 0, 1])
        self.a_add = [T.from_numpy(np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0])).type(T.FloatTensor),
                 T.from_numpy(np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0])).type(T.FloatTensor),
                 T.from_numpy(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])).type(T.FloatTensor)]
        self.state = T.from_numpy(a).type(T.FloatTensor)
        self.finished_state = T.from_numpy(finished_a).type(T.FloatTensor)
        self.variables = []
        self.val_goals = [[0.2, .2, .2], [.3, .3, .3], [.4, .4, .4]]
        goal = np.array([random.uniform(-.4, 0.4), random.uniform(-.4, 0.4),
                         random.uniform(0.0, 0.4)])
        #goal = np.array([0.3479, 0.3065, 0.3976])
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
        self.tau = 0.005
        self.explore_episodes = 0
        self.ep = 0
        self.noise = 0.05
        self.update_actor_interval = 10
        self.learn_step_cntr = 0


    def reset(self):
        a = np.array([0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0])
        self.state = T.from_numpy(a).type(T.FloatTensor)
        self.prev_action = 1
        self.variables = np.zeros((self.l_cnt* self.num_d_vars))
        goal = np.array([random.uniform(-.4, 0.4), random.uniform(-.4, 0.4),
                         random.uniform(0.0, 0.4)])
        #goal = np.array([0.3479, 0.3065, 0.3976])
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
        print('action: ', action)
        #print('next_variables: ', next_variables)
        next_variables[(self.active_l_cnt - 1)*self.num_d_vars:(self.active_l_cnt)*self.num_d_vars] = np.array(action)
        dist, rew, end_eff_pos = reward(self, curr, next_variables, self.goal)

        if done:
            if dist > self.epsilon:
                self.buffer_goal = T.from_numpy(np.array(end_eff_pos)).type(T.FloatTensor)
                self.failed = 1

        return next_variables, rew, dist, done

    def test_step(self, actor, state, variables, goal):
        action = actor(state, variables, goal, 0, env).detach().numpy()
        self.variables[(self.active_l_cnt - 1) * self.num_d_vars:(self.active_l_cnt) * self.num_d_vars] = np.array(action)

def validation(env):
    for goal in env.val_goals:
        print('for validation goal: ', goal)
        env.reset()
        for curr in range(0, len(env.state), env.mod_size):
            #print('state: ', env.state)
            mod = env.finished_state[curr:curr + env.mod_size]
            if mod[-2] != 1:
                continue
            env.state[curr - env.mod_size: curr + env.mod_size] = env.a_add[env.active_l_cnt]
            env.active_l_cnt += 1
            if env.active_l_cnt == env.l_cnt:
                env.state[curr + env.mod_size: curr + 3 * env.mod_size] = env.a_add[env.active_l_cnt]
            env.test_step(actor, env.state, env.variables, T.from_numpy(np.array(goal)).type(
                T.FloatTensor))
            if env.active_l_cnt == env.l_cnt:
                dist, rew, end_eff_pos = term_reward(env, curr, env.variables, goal)
                print('final distance: ', dist)
                print('final variables: ', env.variables)

                if goal == env.val_goals[0]:
                    writer.add_scalar('Validation Distance/(.2,.2,.2)', dist, ep)
                    for i in range(0, len(env.variables), env.num_d_vars):
                        if i == 0:
                            writer.add_scalar('Validation Goal 1 Variables/Length 1', env.variables[i],ep)
                            writer.add_scalar('Validation Goal 1 Variables/Twist 1', env.variables[i + 1], ep)
                        else:
                            writer.add_scalar('Validation Goal 1 Variables/Length 2', env.variables[i],ep)
                            writer.add_scalar('Validation Goal 1 Variables/Twist 2', env.variables[i + 1], ep)
                elif goal == env.val_goals[1]:
                    writer.add_scalar('Validation Distance/(.3,.3,.3)', dist, ep)
                    for i in range(0, len(env.variables), env.num_d_vars):
                        if i == 0:
                            writer.add_scalar('Validation Goal 2 Variables/Length 1', env.variables[i], ep)
                            writer.add_scalar('Validation Goal 2 Variables/Twist 1', env.variables[i + 1], ep)
                        else:
                            writer.add_scalar('Validation Goal 2 Variables/Length 2', env.variables[i], ep)
                            writer.add_scalar('Validation Goal 2 Variables/Twist 2', env.variables[i + 1], ep)
                else:
                    writer.add_scalar('Validation Distance/(.4,.4,.4)', dist, ep)
                    for i in range(0, len(env.variables), env.num_d_vars):
                        if i == 0:
                            writer.add_scalar('Validation Goal 3 Variables/Length 1', env.variables[i],ep)
                            writer.add_scalar('Validation Goal 3 Variables/Twist 1', env.variables[i + 1], ep)
                        else:
                            writer.add_scalar('Validation Goal 3 Variables/Length 2', env.variables[i],ep)
                            writer.add_scalar('Validation Goal 3 Variables/Twist 2', env.variables[i + 1], ep)
                break

def optimize_model(memory, env, actor, target_actor, critic_1, target_critic_1, critic_2, target_critic_2, ep):
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

    final_mask = T.tensor(tuple(map(lambda s: s == 1,
                                          batch.reward)),dtype=T.bool)
    #print('final state batch: ', state_batch[final_mask])
    #print('final action batch: ', action_batch[final_mask])
    #print('variables batch: ', variables_batch)
    #print('reward batch: ', reward_batch)
    #print('next variables batch: ', next_variables_batch)
    #print('masked next variables batch: ', done_batch)

    # calculate target actions from bellman's equation?
    target_actions = target_actor.forward(state_batch, next_variables_batch, goal_batch, 1, env)

    length_smoothing = T.clamp(T.tensor(np.random.normal(scale=env.noise, size=1), dtype=T.float),-.15,.15)
    twist_smoothing = T.clamp(T.tensor(np.random.normal(scale=np.pi / 6, size=1), dtype=T.float),-0.5,0.5)
    smoothing = T.cat((length_smoothing,twist_smoothing),0).view(env.n_actions)
    noisy_target_actions = target_actions + smoothing
    print('noisy target actions: ', noisy_target_actions)
    #min = T.min(noisy_target_actions, T.tensor([[0.0, 0.0]],dtype=T.float))
    #print('min: ', min)
    clamped_target_actions = T.max(T.min(noisy_target_actions, T.tensor(env.action_bounds,dtype=T.float)), T.tensor([[0.05, 0.0]], dtype=T.float))
    #clamped_target_actions = T.tensor(np.array([]),dtype=T.float)
    #clamped_target_actions = T.cat((clamped_target_actions, T.clamp(noisy_target_actions[0], 0.05, env.action_bounds[0])),0)
    #clamped_target_actions = T.cat((clamped_target_actions, T.clamp(noisy_target_actions[1], 0, env.action_bounds[1])),0)
    print('clamped target actions: ', clamped_target_actions)
    # for the new state, getting target action actions from target actor network
    # what actions should target critic take based on target actor's estimates of actions
    target_critic_value_1 = target_critic_1.forward(state_batch, next_variables_batch, goal_batch, clamped_target_actions, 1, env)
    target_critic_value_2 = target_critic_2.forward(state_batch, next_variables_batch, goal_batch, clamped_target_actions, 1, env)
    #print('target critic value 1: ', target_critic_value_1)
    #print('target critic value 2: ', target_critic_value_2)

    # what was the estimate of the values of the states and actions we encountered in our sample of replay buffer
    critic_value_1 = critic_1.forward(state_batch, variables_batch, goal_batch, action_batch, 1, env)
    critic_value_2 = critic_2.forward(state_batch, variables_batch, goal_batch, action_batch, 1, env)

    target_critic_value = T.min(target_critic_value_1, target_critic_value_2)
    # calculating our target values using Bellman's Equation
    #print('variables batch: ', variables_batch)
    #print('next variables batch: ', next_variables_batch)
    print('target critic value: ', target_critic_value)
    target = reward_batch + env.gamma * target_critic_value * done_batch
    print('target: ', target.view(env.batch_size, 1))
    #print('critic value: ', critic_value.view(env.batch_size, 1))
    #  whenever we calculate loss, want to zero gradients so gradients from
    # prev. steps don't accumulate and interfere with current calculations
    # Critic loss
    critic_1.optimizer.zero_grad()
    critic_2.optimizer.zero_grad()
    q1_loss = F.mse_loss(target,critic_value_1)
    q2_loss = F.mse_loss(target,critic_value_2)
    #print('target: ', target)
    #print('critic value 1: ', critic_value_1)
    #print('critic value 2: ', critic_value_2)
    critic_loss =  q1_loss + q2_loss
    #print('critic loss: ', critic_loss)
    writer.add_scalar('Critic Loss/train', critic_loss, ep)
    critic_loss.backward()
    for param in critic_1.parameters():
        if param.grad is not None:
            param_norm = T.norm(param.grad)
            param = T.div(param,param_norm)
    critic_1.optimizer.step()
    critic_2.optimizer.step()

    env.learn_step_cntr += 1

    if env.learn_step_cntr % env.update_actor_interval != 0:
        return


    #critic_scheduler.step()
    #critic.lr = critic.init_lr*math.exp(-ep/LR_DECAY)

    # Actor loss
    actor.optimizer.zero_grad()

    # implicitly calculates chain rule for actor/critic derivatives
    actor_loss = critic_1.forward(state_batch, variables_batch, goal_batch,
                                        actor.forward(state_batch, variables_batch, goal_batch, 1, env), 1, env)
    #print('actor loss: ', actor_loss)
    actor_loss = -T.mean(actor_loss)
    writer.add_scalar('Actor Loss/train', actor_loss, ep)
    actor_loss.backward()
    #for param in critic.parameters():
    #    if param.grad.data is not None:
    #        param.grad.data.clamp_(-1, 1)
    actor.optimizer.step()
    #actor_scheduler.step()
    #actor.lr = actor.init_lr*math.exp(-ep/LR_DECAY)

    soft_update_network_parameters(actor, target_actor, critic_1, target_critic_1, critic_2, target_critic_2, env.tau)

def soft_update_network_parameters(actor, target_actor, critic_1, target_critic_1, critic_2, target_critic_2, tau):
    if tau is None:
        tau = self.tau

    actor_params = actor.named_parameters()
    critic_1_params = critic_1.named_parameters()
    critic_2_params = critic_2.named_parameters()
    target_actor_params = target_actor.named_parameters()
    target_critic_1_params = target_critic_1.named_parameters()
    target_critic_2_params = target_critic_2.named_parameters()

    critic_1_dict = dict(critic_1_params)
    critic_2_dict = dict(critic_2_params)
    actor_dict = dict(actor_params)
    target_actor_dict = dict(target_actor_params)
    target_critic_1_dict = dict(target_critic_1_params)
    target_critic_2_dict = dict(target_critic_2_params)

    for name in critic_1_dict:
        critic_1_dict[name] = tau * critic_1_dict[name].clone() + \
                         (1 - tau) * target_critic_1_dict[name].clone()
    for name in critic_2_dict:
        critic_2_dict[name] = tau * critic_2_dict[name].clone() + \
                         (1 - tau) * target_critic_2_dict[name].clone()
    for name in actor_dict:
        actor_dict[name] = tau * actor_dict[name].clone() + \
                      (1 - tau) * target_actor_dict[name].clone()

    target_critic_1.load_state_dict(critic_1_dict)
    target_critic_2.load_state_dict(critic_2_dict)
    target_actor.load_state_dict(actor_dict)

def hard_update_network_parameters(actor, target_actor, critic, target_critic):
    # hard update version
    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
        target_param.data.copy_(param.data)
    #for target_param, param in zip(target_critic.parameters(), critic.parameters()):
    #    target_param.data.copy_(param.data)
    #target_actor.load_state_dict(actor.state_dict())
    #target_critic.load_state_dict(critic.state_dict())

if __name__ == '__main__':
    TARGET_UPDATE = 10
    LR_DECAY = 500

    buffer = ReplayMemory(100)
    env = env(arm_size=6, n_actions=2, action_bounds= [.75,2*np.pi], num_d_vars=2)
    #noise = ActionNoise()

    noise = 0.05
    #for curr in range(0, len(env.state), env.mod_size):
    #    mod = env.state[curr:curr + env.mod_size]
    #    if mod[-2] == 1:
    #        env.l_cnt += 1
    env.l_cnt = 2
    n_actions = 2
    var_size = n_actions * env.l_cnt
    lr_actor = .0001
    lr_critic = .001
    train_episodes = 6000
    test_episodes = 50
    explore_episodes = 500
    env.explore_episodes = explore_episodes

    actor = Actor(input_size=len(env.state), goal_size=len(env.goal), var_size=var_size, n_actions=n_actions,fc1_dims=400,
                  fc2_dims=128, a_dims= 64, v_dims= 10, lr_actor=lr_actor,action_bound=[.75,2*np.pi])
    target_actor = Actor(input_size=len(env.state), goal_size=len(env.goal), var_size=var_size, n_actions=n_actions,fc1_dims=400,
                         fc2_dims=128, a_dims= 64, v_dims= 10, lr_actor=lr_actor,action_bound=[.75,2*np.pi])
    critic_1 = Critic(input_size=len(env.state), goal_size=len(env.goal), var_size=var_size, n_actions=n_actions,fc1_dims=400,
                    fc2_dims=128, a_dims= 64, v_dims= 10, lr_critic=lr_critic)
    target_critic_1 = Critic(input_size=len(env.state), goal_size=len(env.goal), var_size=var_size, n_actions=n_actions,fc1_dims=400,
                           fc2_dims=128,  a_dims= 64, v_dims= 10, lr_critic=lr_critic)
    critic_2 = Critic(input_size=len(env.state), goal_size=len(env.goal), var_size=var_size, n_actions=n_actions,
                      fc1_dims=400,
                      fc2_dims=128, a_dims=64, v_dims=10, lr_critic=lr_critic)
    target_critic_2 = Critic(input_size=len(env.state), goal_size=len(env.goal), var_size=var_size, n_actions=n_actions,
                             fc1_dims=400,
                             fc2_dims=128, a_dims=64, v_dims=10, lr_critic=lr_critic)

    # initialize networks to have the same features
    #hard_update_network_parameters(actor, target_actor, critic, target_critic)
    soft_update_network_parameters(actor, target_actor, critic_1, target_critic_1,
                                   critic_2, target_critic_2, tau=1)

    for ep in range(train_episodes):
        env.ep = ep
        env.reset()
        print('goal: ', env.goal)
        #print('actor lr: ', actor.lr)
        #print('critic lr: ', critic.lr)
        if ep % 25 == 0:
            print('episode: ', ep)
            validation(env)
            env.reset()
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
            #print('PUSHING TO BUFFER')
            #print('state: ', env.state)
            #print('variables: ', T.tensor(np.array(env.variables)).type(T.FloatTensor))
            #print('next variables: ', T.tensor(np.array(next_variables)).type(T.FloatTensor))
            #print('action: ', T.tensor(np.array([action])).type(T.FloatTensor))
            #print('rew: ', T.tensor(np.array([rew])).type(T.FloatTensor))
            #print('goal: ', env.goal)
            #print('1 - done: ', T.tensor(np.array([1 - done])).type(T.LongTensor))
            buffer.push(env.state, T.tensor(np.array(env.variables)).type(T.FloatTensor), T.tensor(np.array(next_variables)).type(T.FloatTensor), T.tensor(np.array([action])).type(T.FloatTensor),
                        T.tensor(np.array([rew])).type(T.FloatTensor), env.goal,
                        T.tensor(np.array([1 - done])).type(T.LongTensor))
            if done:
                print('variables: ', next_variables)
                #print('reward: ', rew)
                if env.failed:
                    #for i in range(0, env.num_d_vars * env.l_cnt, env.num_d_vars):
                        #couple = env.variables[i:i + env.num_d_vars]
                        #writer.add_scalar('Arrangements/fail', couple[0], couple[1])
                    #    not_worked_x.append(couple[0])
                    #    not_worked_y.append(couple[1])
                    print('failed')
                    print('goal point was: ' + str(env.goal) + " while end eff position was " + str(env.buffer_goal))
                    print('distance was: ', dist)
                    #print('using buffer goal of: ', env.buffer_goal)
                    env.sequence.append((env.state,  T.tensor(np.array(env.variables)).type(T.FloatTensor),
                                         T.tensor(np.array(next_variables)).type(T.FloatTensor),
                                         T.tensor(np.array([action])).type(T.FloatTensor),
                                         T.tensor(np.array([1.0])).type(T.FloatTensor),
                                         T.tensor(np.array([1 - done])).type(T.LongTensor)))
                    #print('PUSHING TO BUFFER FROM FAILED')
                    for part in env.sequence:
                        #print('state: ', part[0])
                        #print('variables: ', part[1])
                        #print('next variables: ', part[2])
                        #print('action: ', part[3])
                        #print('rew: ', part[4])
                        #print('goal: ', env.buffer_goal)
                        #print('1 - done: ', part[5])
                        buffer.push(part[0], part[1], part[2], part[3], part[4], env.buffer_goal, part[5])
                else:
                    #for i in range(0, env.num_d_vars * env.l_cnt, env.num_d_vars):
                    #    couple = env.variables[i:i + env.num_d_vars]
                        #writer.add_scalar('Arrangements/pass', couple[0], couple[1])
                    #    worked_x.append(couple[0])
                    #    worked_y.append(couple[1])
                    print('succeeded')
                    #print('goal point was: ' + str(env.goal))
                    print('distance: ', dist)
                writer.add_scalar('Distance/train', dist, ep)
                break
            else:
                env.sequence.append((env.state,  T.tensor(np.array(env.variables)).type(T.FloatTensor),
                                     T.tensor(np.array(next_variables)).type(T.FloatTensor),
                                     T.tensor(np.array([action])).type(T.FloatTensor),
                                     T.tensor(np.array([rew])).type(T.FloatTensor),
                                     T.tensor(np.array([1 - done])).type(T.LongTensor)))
            optimize_model(buffer,env, actor, target_actor, critic_1, target_critic_1, critic_2, target_critic_2, ep)
            env.variables = copy.deepcopy(next_variables)
            #print('env variables copied as: ', env.variables)
        #if ep % TARGET_UPDATE == 0:
        #    hard_update_network_parameters(actor,target_actor,critic,target_critic)
    #pyplot.scatter(not_worked_x, not_worked_y)
    #pyplot.scatter(worked_x, worked_y)
    results = []
    for test in range(test_episodes):
        env.reset()
        #print('goal: ', env.goal)
        for curr in range(0, len(env.state), env.mod_size):
            #print('state: ', env.state)
            mod = env.finished_state[curr:curr + env.mod_size]
            if mod[-2] != 1:
                continue
            env.state[curr - env.mod_size: curr + env.mod_size] = env.a_add[env.active_l_cnt]
            env.active_l_cnt += 1
            if env.active_l_cnt == env.l_cnt:
                env.state[curr + env.mod_size: curr + 3 * env.mod_size] = env.a_add[env.active_l_cnt]
            env.test_step(actor, env.state, env.variables, env.goal)
            if env.active_l_cnt == env.l_cnt:
                dist, rew, end_eff_pos = term_reward(env, curr, env.variables, env.goal)
                print('variables: ', env.variables)
                print('final distance: ', dist)
                writer.add_scalar('Distance/test', dist, test)
                break
