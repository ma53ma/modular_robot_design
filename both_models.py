import os
import torch as T
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 0
LOG_SIG_MIN = -20
epsilon = 1e-6

# initializes weights, don't know what it does
def weights_init_(m):
    if isinstance(m, nn.Linear):
        T.nn.init.xavier_uniform_(m.weight, gain=1)
        T.nn.init.constant_(m.bias,0)

class DQN(nn.Module):
    def __init__(self, state_size, action_size, lr, gamma, target_size, pre_process_dims, n_vars,l_cnt,c_size):
        super(DQN, self).__init__()
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.gamma = gamma
        self.pre_process_dims = pre_process_dims
        self.n_actions = n_vars

        self.var_pre_process = nn.Linear(self.n_actions, self.pre_process_dims)
        self.model = nn.Sequential(nn.Linear(64 + l_cnt * n_vars + target_size + c_size, 128),
                                   nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32),
                                   nn.ReLU(), nn.Linear(32, action_size))
        self.goal_layer = nn.Linear(target_size, 9)
        self.a_layer = nn.Linear(state_size, 64)
        self.latent_layer = nn.Linear(64 + l_cnt * n_vars + target_size + c_size, 32)

        self.apply(weights_init_)

    def forward(self, state, variables, goal, c, batch, env):
        #c = T.tensor(np.array([c])).type(T.FloatTensor)
        masked_batch_size = len(variables)
        if batch:
            #print('variables: ', variables)
            tensor_variables = variables
            '''
            pre_process_vars = T.from_numpy(np.zeros((masked_batch_size, env.l_cnt * self.pre_process_dims))).type(
                T.FloatTensor)
            for i in range(0, env.num_vars * env.l_cnt, env.num_vars):
                d_var_set = tensor_variables[:, 0:env.num_vars]
                mask = T.tensor(tuple(map(lambda s: s[0] > 0,
                                          d_var_set)), dtype=T.bool)
                output = self.var_pre_process(d_var_set)
                good_pre_process_vars = T.from_numpy(np.zeros((masked_batch_size, self.pre_process_dims))).type(
                    T.FloatTensor)
                #print('mask: ', mask)
                #print('output: ', output)
                #print('good pre proces vars: ', good_pre_process_vars)
                good_pre_process_vars[mask] = output[mask]
                pre_process_vars[:, i * int(self.pre_process_dims / self.n_actions):i * int(
                   self.pre_process_dims / self.n_actions) + self.pre_process_dims] = good_pre_process_vars
            '''
        else:
            tensor_variables = T.from_numpy(variables).type(T.FloatTensor)
            '''
            pre_process_vars = T.from_numpy(np.array([0.0] * (env.l_cnt * self.pre_process_dims))).type(T.FloatTensor)
            for i in range(0, self.n_actions * env.l_cnt, self.n_actions):
                d_var_set = tensor_variables[i:i+ self.n_actions]
                output = self.var_pre_process(d_var_set)
                if d_var_set[0] > 0:
                    pre_process_vars[i * int(self.pre_process_dims / self.n_actions):i * int(
                        self.pre_process_dims / self.n_actions) + self.pre_process_dims] = output
            '''

        a_res = self.a_layer(state)
        goal_res = self.goal_layer(goal)
        if batch:
            tot_res = T.cat((a_res, tensor_variables, goal, c),1)
            Z = self.latent_layer(tot_res)
        else:
            #print(env.c)
            tot_res = T.cat((a_res, tensor_variables, goal, c), 0)
            Z = self.latent_layer(tot_res)
        #print('tot_res', tot_res)
        return self.model(tot_res), Z

# critic network
class QNetwork(nn.Module):
    def __init__(self, n_actions, l_cnt, Z_size, a_dims, pre_process_dims, fc1_dims, fc2_dims):
        super(QNetwork, self).__init__()

        self.a_dims = a_dims
        self.pre_process_dims = pre_process_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.l_cnt = l_cnt
        self.Z_size = Z_size

        self.loss_fn = nn.MSELoss()

        # Q1 architecture
        self.var_pre_process1 = nn.Linear(self.n_actions, self.pre_process_dims)
        self.action_pre_process1 = nn.Linear(self.n_actions, self.pre_process_dims)
        #self.a1 = nn.Linear(self.input_size + self.goal_size, self.a_dims)
        #self.g1 = nn.Linear(self.goal_size, self.g_dims)
        #self.linear1 = nn.Linear(self.Z_size + self.pre_process_dims, self.fc1_dims)
        self.linear1 = nn.Linear(self.Z_size + self.n_actions + 1, self.fc1_dims)
        self.linear2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        # Q2 architecture
        self.var_pre_process2 = nn.Linear(self.n_actions, self.pre_process_dims)
        self.action_pre_process2 = nn.Linear(self.n_actions, self.pre_process_dims)
        #self.a2 = nn.Linear(self.input_size + self.goal_size, self.a_dims)
        #self.g2 = nn.Linear(self.goal_size, self.g_dims)
        #self.linear3 = nn.Linear(self.l_size + self.m_size + self.pre_process_dims + self.c_size, self.fc1_dims)
        #self.linear3 = nn.Linear(self.Z_size + self.pre_process_dims, self.fc1_dims)
        self.linear3 = nn.Linear(self.Z_size + self.n_actions + 1, self.fc1_dims)
        self.linear4 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q2 = nn.Linear(self.fc2_dims, 1)

        self.apply(weights_init_)

    def forward(self, variables, Z, action, batch, env):
        masked_batch_size = len(variables)
        '''
        if batch:
            tensor_variables = variables
            pre_process_action1 =  T.from_numpy(np.zeros((masked_batch_size, self.pre_process_dims))).type(
                T.FloatTensor)
            pre_process_action2 = T.from_numpy(np.zeros((masked_batch_size, self.pre_process_dims))).type(
                T.FloatTensor)
            #print('action: ', action)
            action_mask = T.tensor(tuple(map(lambda s: s[0] > 0,
                                      action)), dtype=T.bool)
            action_output1 = self.action_pre_process1(action)
            action_output2 = self.action_pre_process2(action)
            pre_process_action1[action_mask] = action_output1[action_mask]
            pre_process_action2[action_mask] = action_output2[action_mask]
            # pre_process_action2 = self.action_pre_process2(action) not sure why this is here
            pre_process_vars1 = T.from_numpy(np.zeros((masked_batch_size, env.l_cnt * self.pre_process_dims))).type(
                T.FloatTensor)
            pre_process_vars2 = T.from_numpy(np.zeros((masked_batch_size, env.l_cnt * self.pre_process_dims))).type(
                T.FloatTensor)
            for i in range(0, env.num_vars * env.l_cnt, env.num_vars):
                d_var_set = tensor_variables[:, 0:env.num_vars]
                mask = T.tensor(tuple(map(lambda s: s[0] > 0,
                                          d_var_set)), dtype=T.bool)
                output1 = self.var_pre_process1(d_var_set)
                output2 = self.var_pre_process2(d_var_set)
                good_pre_process_vars1 = T.from_numpy(np.zeros((masked_batch_size, self.pre_process_dims))).type(
                    T.FloatTensor)
                good_pre_process_vars2 = T.from_numpy(np.zeros((masked_batch_size, self.pre_process_dims))).type(
                    T.FloatTensor)
                good_pre_process_vars1[mask] = output1[mask]
                good_pre_process_vars2[mask] = output2[mask]
                pre_process_vars1[:, i * int(self.pre_process_dims / self.n_actions):i * int(
                    self.pre_process_dims / self.n_actions) + self.pre_process_dims] = good_pre_process_vars1
                pre_process_vars2[:, i * int(self.pre_process_dims / self.n_actions):i * int(
                    self.pre_process_dims / self.n_actions) + self.pre_process_dims] = good_pre_process_vars2
        else:
            pre_process_action1 = T.from_numpy(np.array([0.0] * self.pre_process_dims)).type(
                T.FloatTensor)
            pre_process_action2 = T.from_numpy(np.array([0.0] * self.pre_process_dims)).type(
                T.FloatTensor)
            if action != 0:
                pre_process_action1 = self.action_pre_process1(action)
                pre_process_action2 = self.action_pre_process2(action)
        '''
        #goal_output1 = self.g1(goal)
        #goal_output2 = self.g2(goal)

        #print('state output1: ', state_output1)
        #print('pre process vars: ', pre_process_vars1)
        #print('action: ', action)
        #goal_output1 = self.g1(goal)
        #goal_output2 = self.g2(goal)
        if batch:
            #print('Z shape: ', Z.shape)
            #print('action shape: ', action.shape)
            concat_state1 = T.cat((Z, action.view(Z.size()[0],-1)),1)
            concat_state2 = T.cat((Z, action.view(Z.size()[0],-1)),1)
        else:
            #variables = T.from_numpy(variables).type(T.FloatTensor)
            concat_state1 = T.cat((Z, action),0)
            concat_state2 = T.cat((Z, action),0)
        state_value1 = self.linear1(concat_state1)
        state_value2 = self.linear3(concat_state2)
        state_value1 = F.relu(state_value1)
        state_value2 = F.relu(state_value2)

        state_action_value1 = F.relu(self.linear2(state_value1))
        state_action_value2 = F.relu(self.linear4(state_value2))

        state_action_value1 = self.q1(state_action_value1)
        state_action_value2 = self.q2(state_action_value2)

        return state_action_value1, state_action_value2

class GaussianPolicy(nn.Module):
    def __init__(self, n_actions, l_cnt, Z_size, a_dims, pre_process_dims, fc1_dims, fc2_dims, action_space):
        super(GaussianPolicy, self).__init__()

        self.n_actions = n_actions
        self.a_dims = a_dims
        self.pre_process_dims = pre_process_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.l_cnt = l_cnt
        self.Z_size = Z_size

        #self.var_pre_process = nn.Linear(self.n_actions, self.pre_process_dims)
        #self.a1 = nn.Linear(self.n_actions + self.goal_size, self.a_dims)
        #self.g1 = nn.Linear(self.goal_size, self.g_dims)
        #self.linear1 = nn.Linear(self.pre_process_dims*l_cnt + self.goal_size, self.fc1_dims)
        self.linear1 = nn.Linear(self.Z_size, self.fc1_dims)

        self.linear2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # two outputs for action
        self.mean_linear = nn.Linear(self.fc2_dims, n_actions)
        self.log_std_linear = nn.Linear(self.fc2_dims, n_actions)

        self.apply(weights_init_)

        # action rescaling
        low = action_space[0:, 0]
        high = action_space[0:, -1]
        #print('action space: ', action_space)
        #print(low)
        #print(high)
        self.action_scale = T.FloatTensor(
            (high - low) / 2.)
        self.action_bias = T.FloatTensor(
            (high + low) / 2.)

    def forward(self, Z , batch, env):
        #print('variables: ', variables)
        '''
        if batch:
            tensor_variables = variables
            pre_process_vars = T.from_numpy(np.zeros((env.batch_size, env.l_cnt * self.pre_process_dims))).type(
                T.FloatTensor)
            for i in range(0, env.num_vars * env.l_cnt, env.num_vars):
                d_var_set = tensor_variables[:, 0:env.num_vars]
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
            #print('pre process vars first: ', pre_process_vars)
            for i in range(0, env.num_vars * env.l_cnt, env.num_vars):
                d_var_set = tensor_variables[i:i+ env.num_vars]
                #print('d var set: ', d_var_set)
                output = self.var_pre_process(d_var_set)
                if d_var_set[0] > 0:
                    pre_process_vars[i * int(self.pre_process_dims / self.n_actions):i * int(
                        self.pre_process_dims / self.n_actions) + self.pre_process_dims] = output
                    #print('pre process vars updated: ', pre_process_vars)
        #state_output = self.a1(state)
        #
        #print('pre process: ', pre_process_vars)
        '''
        #print('pre process vars: ', pre_process_vars)
        output1 = self.linear1(Z)
        output1 = F.relu(output1)
        output2 = self.linear2(output1)
        output2 = F.relu(output2) # just added this
        mean = self.mean_linear(output2)
        log_std = self.log_std_linear(output2)
        log_std = T.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, Z, batch, env):
        mean, log_std = self.forward(Z, batch, env)
        std = log_std.exp()
        #normal = Normal(0, 1)
        normal = Normal(0, 1) # what should mean and std be here?
        x_t = normal.rsample() # for reparameterization trick
        y_t = T.tanh(x_t*std + mean)
        #y_t = T.sigmoid(x_t) # actions (changing from tanh to sigmoid
        # action = y_t
        #print('un scaled action: ', y_t)
        action = y_t * self.action_scale + self.action_bias # other people do not use scale or bias
        #print('scaled and bias action: ', action)
        log_prob = normal.log_prob(x_t)

        # enforcing action bound
        #log_prob -= T.log(1 - y_t.pow(2) + epsilon)
        #print('self.action_scale: ', self.action_scale)
        #print('y_t: ', y_t)
        log_prob -= T.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        #print('log prob: ', log_prob)

        # make these negative?
        if batch:
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = log_prob.sum(0, keepdim=True)
        # mean = T.tanh(means)
        #print('mean: ', mean)
        #print('self.action_scale: ', self.action_scale)
        #print('self.action_bias: ', self.action_bias)
        mean = T.tanh(mean) * self.action_scale + self.action_bias
        #print('new mean: ', mean)
        return action, log_prob, mean
