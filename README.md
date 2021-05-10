# modular_robot_design
# both_main.py  
This file runs the main training loop where the networks make actions, receive updates, and are subsequently updated using a replay buffer.  
class env(): the environment that the agent will act in  
  def init: initialize environment  
  def reset: reset the environment after each episode  
  def step: once the actions have been chosen, receive the rewards and update the state  
  def test_step: step function used when networks are being evaluated, not trained  

