import os
import torch as T
from torch import nn
import random
import math
import copy
from collections import namedtuple
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from sac_utils import soft_update, hard_update
from both_models import GaussianPolicy, QNetwork, DQN
from torch.distributions import Normal

# setting up syntax for buffer samples
Transition = namedtuple('Transition',
                        ('state', 'variables', 'next_state','next_variables', 'goal','c','Z','DQN_action',
                         'SAC_action', 'DQN_reward','SAC_reward', 'done','mask'))

epsilon = 1e-6

# initializing class for buffer replay
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

# initializing agent for RL
class Agent(object):
    def __init__(self, num_inputs, action_space, goal_size, args, n_vars, l_cnt, env):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        # network layer sizes
        # state size
        self.a_dims = 20
        # goal size
        self.g_dims = 15
        # size used for action/variable pre-processing
        self.pre_process_dims = 2
        # fully connected layer sizes
        self.fc1_dims = 64
        self.fc2_dims = 32
        # learning rates
        self.critic_lr = args.critic_lr
        self.actor_lr = args.actor_lr
        self.dqn_lr = args.dqn_lr
        # size for current point in arm
        self.c_size = 1
        # size for discrete module type chosen
        self.m_size = 1
        # size for latent encoding of state
        self.Z_size = 32

        #print('len(action space) ',action_space.shape)

        # initializing DQN
        self.dqn = DQN(len(env.state), env.mod_size, self.dqn_lr, self.gamma, len(env.goal), self.pre_process_dims, n_vars, l_cnt, self.c_size)
        self.dqn_target = DQN(len(env.state), env.mod_size, self.dqn_lr, self.gamma, len(env.goal), self.pre_process_dims, n_vars, l_cnt, self.c_size)
        self.dqn_optim = optim.RMSprop(self.dqn.parameters(), lr=args.dqn_lr)

        # initializing critic network for SAC
        self.critic = QNetwork(n_vars, l_cnt, self.Z_size, self.a_dims, self.pre_process_dims, self.fc1_dims, self.fc2_dims)
        self.critic_target = QNetwork(n_vars, l_cnt, self.Z_size, self.a_dims, self.pre_process_dims, self.fc1_dims, self.fc2_dims)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr)
        # hard update to set critic equal to target critic
        hard_update(self.critic_target, self.critic)

        # initializing actors for different module types
        self.link_policy = GaussianPolicy(n_vars, l_cnt, self.Z_size, self.a_dims, self.pre_process_dims, self.fc1_dims, self.fc2_dims, action_space)
        self.link_policy_optim = Adam(self.link_policy.parameters(), lr=args.actor_lr)
        self.bracket_policy = GaussianPolicy(n_vars, l_cnt, self.Z_size, self.a_dims, self.pre_process_dims, self.fc1_dims, self.fc2_dims, action_space)
        self.bracket_policy_optim = Adam(self.bracket_policy.parameters(), lr=args.actor_lr)
        self.actuator_policy = GaussianPolicy(n_vars, l_cnt, self.Z_size, self.a_dims, self.pre_process_dims, self.fc1_dims, self.fc2_dims, action_space)
        self.actuator_policy_optim = Adam(self.actuator_policy.parameters(), lr=args.actor_lr)
        self.gripper_policy = GaussianPolicy(n_vars, l_cnt, self.Z_size, self.a_dims, self.pre_process_dims, self.fc1_dims, self.fc2_dims, action_space)
        self.gripper_policy_optim = Adam(self.gripper_policy.parameters(), lr=args.actor_lr)

    def indiv_sample(self, max_dqn_val, Z, batch, env):
        '''        if max_dqn_val == 0:
            action, log_prob, mean = [0,0,0]
        elif max_dqn_val == 1:
            action, log_prob, mean = [0,0,0]
        elif max_dqn_val == 2:
            action, log_prob, mean = self.sample(self.link_policy, Z, batch, env)
        else:
            action, log_prob, mean = [0,0,0]
        return action, log_prob, mean'''
        if max_dqn_val == 2:
            action, log_prob, mean = self.sample(self.link_policy, Z, batch, env)
        else:
            action, log_prob, mean = T.FloatTensor([[0],[0],[0]])
        return action, log_prob, mean

    def update_full_sample(self, max_dqn_vals, Z, batch, env):
        max_dqn_vals = max_dqn_vals.flatten()
        #print('max dqn vals size: ', max_dqn_vals.size())
        #print('Z: ', Z)
        copy_Z = copy.deepcopy(Z)
        #print('copy Z: ', copy_Z)
        link_Z = copy.deepcopy(copy_Z)
        bracket_Z = copy.deepcopy(copy_Z)
        act_Z = copy.deepcopy(copy_Z)
        gripper_Z = copy.deepcopy(copy_Z)
        #print('act Z size: ', act_Z.size())
        #print('max dqn vals: ', max_dqn_vals.size())
        act_Z[max_dqn_vals != 0] = T.zeros(copy_Z[0].size())
        bracket_Z[max_dqn_vals != 1] = T.zeros(copy_Z[0].size())
        link_Z[max_dqn_vals != 2] = T.zeros(copy_Z[0].size())
        gripper_Z[max_dqn_vals != 3] = T.zeros(copy_Z[0].size())
        #print('Link Z: ', link_Z)
        #act_actions, act_log_prob, act_mean = self.sample(self.actuator_policy, act_Z, batch, env)
        act_actions = T.zeros((env.batch_size,1))
        act_log_prob = T.zeros((env.batch_size,1))
        act_mean = T.zeros((env.batch_size,1))
        #bracket_actions, bracket_log_prob, bracket_mean = self.sample(self.bracket_policy, bracket_Z, batch, env)
        bracket_actions = T.zeros((env.batch_size,1))
        bracket_log_prob = T.zeros((env.batch_size,1))
        bracket_mean = T.zeros((env.batch_size,1))
        link_actions, link_log_prob, link_mean = self.sample(self.link_policy,link_Z, batch, env)
        #gripper_actions, gripper_log_prob, gripper_mean = self.sample(self.gripper_policy, gripper_Z, batch, env)
        gripper_actions = T.zeros((env.batch_size,1))
        gripper_log_prob = T.zeros((env.batch_size,1))
        gripper_mean = T.zeros((env.batch_size,1))
        full_actions = T.zeros(link_actions.size())
        full_log_probs = T.zeros(link_log_prob.size())
        full_means = T.zeros(link_mean.size())
        #print('full actions before', full_actions)
        full_actions[max_dqn_vals == 0] = act_actions[max_dqn_vals == 0]
        #print('full actions after actuators', full_actions)
        full_actions[max_dqn_vals == env.mod_size - 1] = bracket_actions[max_dqn_vals == env.mod_size - 1]
        #print('full actions after brackets', full_actions)
        full_actions[max_dqn_vals == env.mod_size - 2] = link_actions[max_dqn_vals == env.mod_size - 2]
        #print('full actions after links', full_actions)
        full_actions[max_dqn_vals == env.mod_size - 3] = gripper_actions[max_dqn_vals == env.mod_size - 3]
        #print('full actions after grippers', full_actions)

        full_log_probs[max_dqn_vals == 0] = act_log_prob[max_dqn_vals == 0]
        full_log_probs[max_dqn_vals == env.mod_size - 1] = bracket_log_prob[max_dqn_vals == env.mod_size - 1]
        full_log_probs[max_dqn_vals == env.mod_size - 2] = link_log_prob[max_dqn_vals == env.mod_size - 2]
        full_log_probs[max_dqn_vals == env.mod_size - 3] = gripper_log_prob[max_dqn_vals == env.mod_size - 3]

        full_means[max_dqn_vals == 0] = act_mean[max_dqn_vals == 0]
        full_means[max_dqn_vals == env.mod_size - 1] = bracket_mean[max_dqn_vals == env.mod_size - 1]
        full_means[max_dqn_vals == env.mod_size - 2] = link_mean[max_dqn_vals == env.mod_size - 2]
        full_means[max_dqn_vals == env.mod_size - 3] = gripper_mean[max_dqn_vals == env.mod_size - 3]

        return full_actions, full_log_probs, full_means

    def sample(self, policy, Z, batch, env):
        if env.evaluate == True:
            with T.no_grad():
                mean, log_std = policy.forward(Z, batch, env)
        else:
            mean, log_std = policy.forward(Z, batch, env)
        std = log_std.exp()
        normal = Normal(mean, std) # what should mean and std be here?
        x_t = normal.rsample() # for reparameterization trick
        y_t = T.tanh(x_t)
        #y_t = T.sigmoid(x_t) # actions (changing from tanh to sigmoid
        # action = y_t
        #print('un scaled action: ', y_t)
        action = y_t*policy.action_scale + policy.action_bias # other people do not use scale or bias
        #print('scaled and bias action: ', action)
        log_prob = normal.log_prob(x_t)

        # enforcing action bound
        #log_prob -= T.log(1 - y_t.pow(2) + epsilon)
        #print('self.action_scale: ', self.action_scale)
        #print('y_t: ', y_t)
        log_prob -= T.log(policy.action_scale * (1 - y_t.pow(2)) + epsilon)
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
        mean = T.tanh(mean) * policy.action_scale + policy.action_bias
        #print('new mean: ', mean)
        return action, log_prob, mean

    def select_action(self, z, max_q_val, env, evaluate=False):
        if evaluate is False:
            action,_,_ = self.indiv_sample(max_q_val, z, 0, env)
        else:
            _,_,action = self.indiv_sample(max_q_val, z, 0, env)
        return action.detach().cpu().numpy()

    def dqn_optimize_model(self,buffer, env, ep):
        if len(buffer) < env.batch_size:
            return 0
        transitions = buffer.sample(env.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = T.cat(batch.state).view(env.batch_size, len(env.state))
        variables_batch = T.cat(batch.variables).view(env.batch_size, env.l_cnt * env.num_vars)
        next_variables_batch = T.cat(batch.next_variables).view(env.batch_size, env.l_cnt * env.num_vars)
        action_batch = T.cat(batch.DQN_action).view(env.batch_size, 1)
        goal_batch = T.cat(batch.goal).view(env.batch_size, len(env.goal))
        mask_batch = T.cat(batch.mask).view(env.batch_size, env.mod_size)
        c_batch = T.cat(batch.c).view(env.batch_size, 1)
        #print('variables: ', variables_batch)
        #print('action_batch: ')
        state_action_outputs = self.dqn(state_batch, variables_batch, goal_batch, c_batch, 1, env)
        state_action_values = state_action_outputs[0].gather(1, action_batch)

        next_state_batch = T.cat(batch.next_state).view(env.batch_size, len(env.state))
        next_state_vals = self.dqn_target(next_state_batch, next_variables_batch, goal_batch, c_batch, 1, env)[0].detach()
        next_state_vals[mask_batch] = -float('inf')

        max_next_state_vals = next_state_vals.max(1)[0]
        done_batch = T.cat(batch.done)
        reward_batch = T.cat(batch.DQN_reward)

        expected_state_action_values = reward_batch + self.dqn.gamma * max_next_state_vals * done_batch

        # Computing loss
        loss = self.dqn.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
        self.dqn_optim.zero_grad()

        # back-propagate loss
        loss.backward()
        for param in self.dqn.parameters():
            if param.grad is not None and param.grad.data is not None:
                param.grad.data.clamp_(-1, 1)
        # optimizer step
        self.dqn_optim.step()
        self.dqn_scheduler.step()

        if ep % env.target_update_interval == 0:
            soft_update(self.dqn_target, self.dqn, self.tau)

        return loss.item()

    def sac_optimize_model(self, memory, env, ep):
        # sample a batch from memory
        if memory.position < env.batch_size:
            return 0
        transitions = memory.sample(env.batch_size)

        batch = Transition(*zip(*transitions))

        state_batch = T.cat(batch.state).view(env.batch_size, len(env.state))
        next_state_batch = T.cat(batch.next_state).view(env.batch_size, len(env.state))
        variables_batch = T.cat(batch.variables).view(env.batch_size, env.l_cnt * env.num_vars)
        next_variables_batch = T.cat(batch.next_variables).view(env.batch_size, env.l_cnt * env.num_vars)
        #print('sac action batch: ', batch.SAC_action)
        #print('action batch: ', action_batch)
        reward_batch = T.cat(batch.SAC_reward).view(env.batch_size, 1)
        #print('reward batch: ', reward_batch)
        goal_batch = T.cat(batch.goal).view(env.batch_size, len(env.goal))
        done_batch = T.cat(batch.done).view(env.batch_size, 1)
        c_batch = T.cat(batch.c).view(env.batch_size, 1)
        Z_batch = T.cat(batch.Z).view(env.batch_size, env.Z_size)
        m_batch = T.cat(batch.DQN_action).type(T.FloatTensor).view(env.batch_size,1)
        #print('sac action batch: ', batch.SAC_action)
        sac_action_batch = T.cat(batch.SAC_action).view(env.batch_size, env.num_vars)
        d_c_action_batch = T.cat([m_batch,sac_action_batch], dim=1)
        #print('d_c_action_batch ', d_c_action_batch)

        with T.no_grad():
            # these may be suspect, check other code bases
            next_q_vals, next_Z = self.dqn(next_state_batch, next_variables_batch, goal_batch, c_batch, 1, env)
            masked_next_q_vals = masking(next_q_vals,env.mod_size,m_batch)
            #print('current q vals: ', m_batch)
            #print('masked next q vals: ', masked_next_q_vals)
            next_state_disc_action = T.argmax(masked_next_q_vals, dim=1).type(T.FloatTensor).view(env.batch_size,1)
            #print('next_state_disc_action: ', next_state_disc_action)
            next_state_cont_action, next_state_log_pi, _ = self.update_full_sample(next_state_disc_action, next_Z, 1, env)
            next_state_d_c_action = T.cat([next_state_disc_action,next_state_cont_action], dim=1).view(env.batch_size, env.num_vars + 1)
            #print('next combined action: ', next_state_d_c_action)
            qf1_next_target, qf2_next_target = self.critic_target(next_variables_batch,next_Z,next_state_d_c_action,1,env)
            #min_qf_next_target = T.min(qf1_next_target, qf2_next_target)
            #print('min qf next target: ', min_qf_next_target)
            #print('next state log pi: ', next_state_log_pi)
            min_qf_next_target = T.min(qf1_next_target, qf2_next_target) - self.alpha*next_state_log_pi
            next_q_value = reward_batch + done_batch * self.gamma* min_qf_next_target
        qf1, qf2 = self.critic(variables_batch, Z_batch, d_c_action_batch, 1, env)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.update_full_sample(T.flatten(m_batch), Z_batch, 1, env)
        d_pi_action_batch = T.cat((m_batch, pi))
        qf1_pi, qf2_pi = self.critic(variables_batch, Z_batch, d_pi_action_batch,1, env)
        min_qf_pi = T.min(qf1_pi, qf2_pi)
        #print('min qf pi: ', min_qf_pi)
        #print('log pi: ', log_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.bracket_policy_optim.zero_grad()
        self.actuator_policy_optim.zero_grad()
        self.link_policy_optim.zero_grad()
        self.gripper_policy_optim.zero_grad()
        policy_loss.backward()
        self.bracket_policy_optim.step()
        self.actuator_policy_optim.step()
        self.link_policy_optim.step()
        self.gripper_policy_optim.step()

        total_loss = qf_loss + policy_loss

        if ep % env.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        self.critic_scheduler.step()
        self.actuator_policy_scheduler.step()
        self.bracket_policy_scheduler.step()
        self.link_policy_scheduler.step()
        self.gripper_policy_scheduler.step()
        return total_loss

        #return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

def masking(probs, mod_size, prev_action):
    #print('probs: ', probs)
    comp_zeros = T.zeros(prev_action.size()).flatten()
    prev_action = prev_action.flatten()
    batch_size = probs.size()[0]
    non_act_infs = T.clone(probs)
    non_act_infs[:,1:4] = T.ones(batch_size,1)*-float('inf')
    act_infs = T.clone(probs)
    act_infs[:,0:1] = T.ones(batch_size,1)*-float('inf')

    probs[prev_action > comp_zeros] = non_act_infs[prev_action > comp_zeros]
    probs[prev_action == comp_zeros] = act_infs[prev_action == comp_zeros]
    return probs