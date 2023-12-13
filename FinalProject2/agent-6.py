import numpy as np

# all constants related to the code are defined before the functions
decay_rate = 0.97
epis_rounds = 2000
bandit_count = 100
lookback = 60  # the number of rounds for looking back
num_my_best = 70
num_my_best_best = 10
num_probabs = 1001
decay_matrix_1 = np.zeros([epis_rounds, num_probabs], 'float64')
decay_matrix_2 = np.zeros([epis_rounds, num_probabs], 'float64')
probability = np.linspace(0, 1, num_probabs)
for i in range(decay_matrix_1.shape[0]):
    decay_matrix_1[i] = probability * decay_rate ** i
decay_matrix_2 = 1 - decay_matrix_1
def policy_1(my, opponent, last_action):
    if opponent[last_action] > 1 and my[last_action] <= 1:
        return True
    return False
def policy_2(my, last_action, num_oppo_last, flag):
    if not flag:
        if num_oppo_last > 1 and my[last_action] <= 4:
            return True
    return False
def policy_3(my_choices, my_bests, last_action):
    if last_action in my_bests and my_choices[last_action] > 1:
        return True
    return False
def policy_4(not_chosen, opponent, flag):
    if flag:
        not_chosen_available = (not_chosen) & (opponent == 0)
        if not_chosen_available.any():
            return np.random.choice(np.where(not_chosen_available)[0])
        return np.random.choice(np.where(not_chosen)[0])
    return None
def policy_5(num_best_best, my_best_best):
    steps = np.zeros(num_best_best)
    for i in range(num_best_best):
        steps[i] = bandit_last_step[my_best_best[i]]
    return my_best_best[np.argmin(steps)]
def update_threshold(seq):
    raw_probab_1 = decay_matrix_1[:seq.shape[0], :][seq == 1, :]
    raw_probab_0 = decay_matrix_2[:seq.shape[0], :][seq == 0, :]
    each_probab = raw_probab_0.prod(axis=0) * raw_probab_1.prod(axis=0)
    half_area = each_probab.sum() / 2
    cum_sum = np.cumsum(each_probab)
    best_estimation = np.argmin(np.abs(cum_sum - half_area))
    return (probability[best_estimation] * 0.97 ** seq.shape[0])
def find_new_bandit(step):
    start = step - lookback if step >= lookback else 0
    opponent_last_moves = opponent_round_record[start:step]
    opponent_last_bandit = opponent_last_moves[-1]
    num_opponent_last_move = (opponent_last_moves == opponent_last_bandit).sum()
    my_bests = np.argsort(thresholds)[-num_my_best:]
    my_best_bests = my_bests[-num_my_best_best:]

    # estimate the next threshold based on decay rate
    thresholds[last_bandit] = update_threshold(bandits_record[last_bandit, :record_index[last_bandit]])
    thresholds[opponent_last_bandit] = update_threshold(bandits_record[opponent_last_bandit, :record_index[opponent_last_bandit]])

    exploit = ((losses[my_best_bests] == 0) & (my_choices[my_best_bests] > 0)).all() \
               or (thresholds[my_best_bests] > .63).all()
    not_chosen = (my_choices == 0)
    not_chosen_any = not_chosen.any()
    # # epsilon-greedy method
    # now_epsilon = 0.1 / (step + 1)
    # if np.random.rand() < now_epsilon and step <= 1500:
    #     return np.random.randint(bandit_count)
    # print the choices of imitation, exploration and exploitation
    if step == 1999:
        print(choice_path)
    # several fixed empirical policy to be chosen from

    # imitation
    if policy_1(my_choices, opponent_choices, opponent_last_bandit):
        choice_path[step, 0] = 1
        return opponent_last_bandit

    if policy_2(my_choices, opponent_last_bandit, num_opponent_last_move, not_chosen_any):
        choice_path[step, 1] = 1
        return opponent_last_bandit

    # exploitation
    if not exploit:
        if policy_3(my_choices, my_bests, opponent_last_bandit):
            choice_path[step, 2] = 1
            return opponent_last_bandit
        # exploration
        choice = policy_4(not_chosen, opponent_choices, not_chosen_any)
        if choice != None:
            choice_path[step, 3] = 1
            return choice

    # final exploitation
    choice_path[step, 4] = 1
    return policy_5(num_my_best_best, my_best_bests)

def agent(observation, configuration):
    global my_round_record, my_choices, opponent_round_record, opponent_choices,\
            thresholds, bandits_record, record_index, wins, losses, bandit_last_step
    global last_bandit, last_reward, total_reward
    global choice_path
    # If the current step is 0, initialize these global variables
    if observation.step == 0:
        my_round_record = np.zeros(epis_rounds, 'int')
        my_choices = np.zeros(bandit_count, 'int')
        opponent_round_record = np.zeros(epis_rounds, 'int')
        opponent_choices = np.zeros(bandit_count, 'int')
        # We can choose different kinds of initial threshold
        # (i.e. Gaussian distribution/uniform distribution/almost at 0.5 with some noise)
        # thresholds = np.random.rand(bandit_count) * 0.01 + 0.495
        thresholds = np.random.rand(bandit_count) * 0.001 + 0.5

        record_index = np.zeros(bandit_count, 'int')
        bandits_record = np.zeros([bandit_count, 2000], 'int')

        wins = np.zeros(bandit_count, 'int')
        losses = np.zeros(bandit_count, 'int')
        bandit_last_step = np.zeros(bandit_count, 'int')

        total_reward = 0
        last_bandit = np.random.randint(bandit_count)

        choice_path = np.zeros((epis_rounds, 5),'int')
        return last_bandit

    # find the last action taken by opponent's agent
    opponent_action = observation.lastActions[abs(1 - observation.agentIndex)]

    # record the information of the last step
    my_round_record[observation.step - 1] = last_bandit
    my_choices[last_bandit] += 1
    opponent_round_record[observation.step - 1] = opponent_action
    opponent_choices[opponent_action] += 1
    bandits_record[opponent_action, record_index[opponent_action]] = -1
    record_index[opponent_action] += 1

    # record whether our agent get reward in the last round or not and update the total reward
    if observation.reward > total_reward:
        total_reward += 1
        bandits_record[last_bandit, record_index[last_bandit]] = 1
        wins[last_bandit] += 1
        last_reward = 1
    else:
        bandits_record[last_bandit, record_index[last_bandit]] = 0
        losses[last_bandit] += 1
        last_reward = 0
    record_index[last_bandit] += 1

    # find the next "best" bandit based on the previous information
    last_bandit = int(find_new_bandit(observation.step))
    bandit_last_step[last_bandit] = observation.step
    return last_bandit

from kaggle_environments import make, evaluate
from agent_7 import agent as agent_test

env = make("mab", debug=True)

env.run(["agent-6.py", "agent-6.py"])
def mean_reward(rewards):
    return sum(r[0] for r in rewards) / float(len(rewards))

# Run multiple episodes to estimate its performance.
print("My Agent vs test Agent:", mean_reward(evaluate("mab", [agent, agent_test], num_episodes=10)))
# print("My Agent vs Negamax Agent:", mean_reward(evaluate("mab", [agent, "negamax"], num_episodes=10)))