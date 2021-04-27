import torch
from torch import nn
import torch.optim as optim

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
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

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
