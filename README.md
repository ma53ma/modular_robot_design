# modular_robot_design
## both_main.py  
This file runs the main training loop where the networks make actions, receive updates, and are subsequently updated using a replay buffer.  
##### class env(): the environment that the agent will act in  
def init: initialize environment  
def reset: reset the environment after each episode  
def step: once the actions have been chosen, receive the rewards and update the state  
def test_step: step function used when networks are being evaluated, not trained  

##### additional functions
def validation: run validation episodes  
def make_arrangement: format arrangement for turning it into an xacro file  
def main: running training loop  

## both_actions.py
def masking: masking action choices based on previous action taken (i.e., if actuator taken previously, cannot take actuator again)  
def sac_choose_action: choosing continuous action using soft actor-critic  
def dqn_choose_action: choosing discrete action using DQN  
def dqn_reward: calculating reward for discrete action from DQN  
def sac_reward: calculating reward for continuous action from SAC  
def sac_term_reward: calculating terminal reward if action is terminal  
def pos_neg_soft: soft reward function including positive rewards if within distance threshold and negative if not  
def pos_soft_rew: reward signal for within distance threshold  
def neg_soft_rew: reward signal for out of distance threshold  
def soft_rew: soft reward function with all positive rewards (using exp)
def binary_rew: hard reward function including positive rewards if within distance threshold and 0 if not  
def binary_orient_rew: hard reward function for when orientation is included in goal, reward if within distance and orientation threshold, 0 if not  
def tiered_binary_orient_rew: reward function including positive rewards for passing both thresholds or either threshold, 0 if not passing any 

## both_agent.py
This file includes files relevant to the agent within this RL environment  
##### class ReplayMemory()
def init: initialize buffer  
def push: push a sample into the buffer  
def sample: sample random batch from buffer  
def len: obtain current size of buffer  

##### class Agent()
def init: initialize the agent  
def indiv_sample: sample continuous action for a single arrangement (not batch, not necessarily using network)  
def update_fall_sample: sample continuous action for a batch for network updates  
def sample: sample a continuous action using actor network  
def select_action:  select a continuous action based on if network is in training or evaluation mode  
def dqn_optimize_model: optimize model for DQN  
def sac_optimize_model: optimize model for SAC  

##### additional functions  
def masking: masking action choices based on previous action taken (i.e., if actuator taken previously, cannot take actuator again)  
