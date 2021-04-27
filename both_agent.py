import os
import torch as T
import random
import math
from collections import namedtuple
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from sac_utils import soft_update, hard_update
from sac_models import GaussianPolicy, QNetwork, DeterministicPolicy


Transition = namedtuple('Transition',
                        ('state', 'variables', 'next_state','next_variables', 'goal','action', 'reward', 'done'))
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

class SAC(object):
    def __init__(self, num_inputs, action_space, goal_size, args, n_actions, l_cnt):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.a_dims = 20
        self.g_dims = 15
        self.pre_process_dims = 2
        self.fc1_dims = 64
        self.fc2_dims = 32
        self.critic_lr = args.critic_lr
        self.actor_lr = args.actor_lr

        print('len(action space) ',action_space.shape)

        self.critic = QNetwork(num_inputs, n_actions, l_cnt, goal_size, self.a_dims, self.g_dims, self.pre_process_dims, self.fc1_dims, self.fc2_dims)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr)

        self.critic_target = QNetwork(num_inputs, n_actions, l_cnt, goal_size, self.a_dims, self.g_dims, self.pre_process_dims, self.fc1_dims, self.fc2_dims)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == 'Gaussian':
            # target entropy = -dim(A) as discussed in paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -T.prod(T.Tensor(action_space.shape)).item()
                self.log_alpha = T.zeros(1, requires_grad=True)
                self.alpha_optim = Adam([self.log_alpha], lr=args.actor_lr)
            print('policy action space: ', action_space)
            self.policy = GaussianPolicy(num_inputs, n_actions, l_cnt, goal_size, self.a_dims, self.g_dims, self.pre_process_dims, self.fc1_dims, self.fc2_dims, action_space)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.actor_lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], goal_size, self.a_dims, self.g_dims, self.pre_process_dims, self.fc1_dims, self.fc2_dims, action_space)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.actor_lr)
        #LR_DECAY = 2000

        #policy_optimizer = optim.RMSprop(self.policy.parameters(), lr=self.actor_lr)
        #critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=self.critic_lr)
        #lambda1 = lambda ep: min(math.exp(-ep / LR_DECAY), 3e-4)
        #self.policy_scheduler = optim.lr_scheduler.LambdaLR(policy_optimizer, lambda1)
        #self.critic_scheduler = optim.lr_scheduler.LambdaLR(critic_optimizer, lambda1)

    def select_action(self, state, variables, goal, env, evaluate=False):
        if evaluate is False:
            action,_,_ = self.policy.sample(state, variables, goal, 0, env)
        else:
            _,_,action = self.policy.sample(state, variables, goal, 0, env)
        return action.detach().cpu().numpy()

    def optimize_model(self, memory, env, ep):
        # sample a batch from memory
        if memory.position < env.batch_size:
            return
        transitions = memory.sample(env.batch_size)

        batch = Transition(*zip(*transitions))

        state_batch = T.cat(batch.state).view(env.batch_size, len(env.state))
        next_state_batch = T.cat(batch.next_state).view(env.batch_size, len(env.state))
        variables_batch = T.cat(batch.variables).view(env.batch_size, env.l_cnt * env.num_d_vars)
        next_variables_batch = T.cat(batch.next_variables).view(env.batch_size, env.l_cnt * env.num_d_vars)
        action_batch = T.cat(batch.action).view(env.batch_size, env.num_d_vars)
        #print('action batch: ', action_batch)
        reward_batch = T.cat(batch.reward).view(env.batch_size, 1)
        goal_batch = T.cat(batch.goal).view(env.batch_size, len(env.goal))
        done_batch = T.cat(batch.done).view(env.batch_size, 1)
        #print('reward batch: ', reward_batch)
        print('next state batch: ', next_state_batch)
        print('next vars batch: ', next_variables_batch)
        with T.no_grad():
            # these may be suspect, check other code bases
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, next_variables_batch, goal_batch, 1, env)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_variables_batch, goal_batch, next_state_action,1,env)
            #min_qf_next_target = T.min(qf1_next_target, qf2_next_target)
            #print('min qf next target: ', min_qf_next_target)
            #print('next state log pi: ', next_state_log_pi)
            min_qf_next_target = T.min(qf1_next_target, qf2_next_target) - self.alpha*next_state_log_pi
            next_q_value = reward_batch + done_batch * self.gamma* min_qf_next_target
        qf1, qf2 = self.critic(state_batch, variables_batch, goal_batch, action_batch, 1, env)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch, variables_batch, goal_batch, 1, env)

        qf1_pi, qf2_pi = self.critic(state_batch, variables_batch, goal_batch, pi, 1, env)
        min_qf_pi = T.min(qf1_pi, qf2_pi)
        #print('min qf pi: ', min_qf_pi)
        #print('log pi: ', log_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            print('alpha: ', self.alpha)
            alpha_tlogs = self.alpha.clone() # for tensorboard logs
        else:
            alpha_loss = T.tensor(0.)
            alpha_tlogs = T.tensor(self.alpha)

        if ep % env.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        #if ep > env.explore_episodes:
        self.critic_scheduler.step()
        self.policy_scheduler.step()

        #return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
