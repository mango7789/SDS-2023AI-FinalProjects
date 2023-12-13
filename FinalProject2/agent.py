import numpy as np

decay_rate = 0.97
epis_steps = 2000
bandit_count = 100
act_timeout = 0.25

def record_last_action(action, choices, last_action, step):
    action[step - 1] = last_action
    choices[last_action] += 1
    return action, choices

def find_next_bandit(step):
    pass

def agent(observation, configuration):
    global my_action, opponent_action, my_choices, opponent_choices, wins, losses, last_step
    global my_last_action, total_reward, last_reward

    curr_step = observation.step
    if curr_step == 0:
        my_action = np.zeros(epis_steps, 'int')
        opponent_action = np.zeros(epis_steps, 'int')
        my_choices = np.zeros(bandit_count, 'int')
        opponent_choices = np.zeros(bandit_count, 'int')
        wins = np.zeros(bandit_count, 'int')
        losses = np.zeros(bandit_count, 'int')
        last_step = np.zeros(bandit_count, 'int')
        
        my_last_action = np.random.randint(0, bandit_count - 1)
        total_reward = 0
        last_reward = 0
        return my_last_action

    # find the last actions taken our opponent's agent
    my_idx = observation.agentIndex
    opponent_idx = abs(1 - my_idx)
    opponent_last_action = observation.lastActions[opponent_idx]

    # record our and our opponent's last action
    my_action, my_choices = record_last_action(my_action, my_choices, my_last_action, curr_step)
    opponent_action, opponent_choices = record_last_action(opponent_action, opponent_choices
                                                           , opponent_last_action, curr_step)
    last_step[my_last_action] = last_step[opponent_last_action] = curr_step - 1

    # update our total reward and last reward
    if observation.reward > total_reward:
        total_reward += 1
        wins[my_last_action] += 1
        last_reward = 1
    else:
        losses[my_last_action] += 1
        last_reward = 0

    my_last_action = find_next_bandit(curr_step)
    return my_last_action

