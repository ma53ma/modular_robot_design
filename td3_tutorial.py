import numpy as np
import torch as T
import torch.nn.functional as F
import torch.optim as optim
import os


# have overestimation bias and variance and bias since we are using NN to approximate
# policy and value

# can bolt these changes onto other algorithms

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

class Critic(nn.Module):
    #typical to extend functionality
    def __init__(self, lr_critic, input_dims, fc1_dims, fc2_dims, n_actions,
                 name, chkpt_dir='tmp/td3'):
        super(Critic,self).__init__()
        self.lr_critic = lr_critic
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir,name+'_td3')
        #training of model takes a while, td3 way faster than DDPG?

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_critic)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        q1_action_value = self.fc1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)

        q1 = self.q1(q1_action_value)

        return q1

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class Actor(nn.Module):
    def __init__(self, lr_actor, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chkpt_dir='tmp/td3'):
        super(Actor, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir,name+'_td3')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_actor)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = T.tanh(self.mu(prob))

        return prob

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent():
    def __init__(self, lr_actor, lr_critic, input_dims, tau, env,
                 gamma=1,update_actor_interval=25,warmup=500, n_actions=2,
                 max_size=1000, layer1_size=400, layer2_size=300, batch_size=25,
                 noise=0.1):
        # this is the noise for exploration
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.input_dims = input_dims
        self.tau = tau
        self.env = env
        self.gamma = gamma
        self.update_actor_interval = update_actor_interval
        self.warmup = warmup
        self.n_actions = n_actions
        self.max_size = max_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.batch_size = batch_size
        self.noise = noise

        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayMemory(max_size)
        # we are delaying updates for actor network, keep track with this
        self.learn_step_cntr = 0
        # to know when warm up has expired
        self.time_step = 0

        self.actor = Actor(lr_actor, input_dims, layer1_size,
                           layer2_size, n_actions, name='actor')
        self.critic_1 = Critic(lr_critic, input_dims, layer1_size,
                             layer2_size, n_actions, name='critic_1')
        self.critic_2 = Critic(lr_critic, input_dims, layer1_size,
                               layer2_size, n_actions, name='critic_2')
        self.target_actor = Actor(lr_actor, input_dims, layer1_size,
                                  layer2_size, n_actions, name='target_actor')
        self.target_critic_1 = Critic(lr_critic, input_dims, layer1_size,
                             layer2_size, n_actions, name='target_critic_1')
        self.target_critic_2 = Critic(lr_critic, input_dims, layer1_size,
                             layer2_size, n_actions, name='target_critic_2')
        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale=self.noise,
                                           size = (self.n_actions,)))
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        # debatable whether or not to add noise to an already noisy thing
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise),
                                 dtype=T.float).to(self.actor.device)
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1

        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.push(state, action, reward, new_state, done) # need to fix

    def learn(self):
        if self.memory.position < self.batch_size:
            return
        # sample memory
        # convert to proper tensors

        target_actions = self.target_actor.forward(new_state)
        # performing smoothing of chosen actions from target actor, clamp between -0.5 and 0.5
        # smoothing means adding noise
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=0.2)),-0.5,0.5)
        target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])
        q1_ = self.target_critic_1.forward(new_state, target_actions)
        q2_ = self.target_critic_2.forward(new_state, target_actions)

        # do same for online critic networks
        # values for actions actually taken and states actually in
        q1 = self.critic_1.forward(state,action)
        q2 = self.critic_2.forward(state,action)

        q1_[done] = 0.0
        q2_[done] = 0.0 # will set to 0.0 if done is true

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_,q2_)

        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size,1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target,q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_interval != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self,tau=None):
        # at start, want to init. target networks with param's from online networks
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() +  \
                             (1-tau)*target_critic_1[name].clone()
        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() +  \
                             (1-tau)*target_critic_2[name].clone()
        for name in actor:
            actor[name] = tau*actor[name].clone() + \
                          (1-tau)*target_actor[name].clone()

        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor)

    def save_models(self):
        self.acterf










