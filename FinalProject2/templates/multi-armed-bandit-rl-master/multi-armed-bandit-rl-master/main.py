from constants import *
from arm import Arm
from agent import Agent

import matplotlib.pyplot as plt


def get_settings():
    """ Reads and saves the settings.

    Returns:
        dict: Dictionary containing all the settings
    """

    settings = dict()
    settings['k'] = int(input("Number of bandit's arms (k): "))
    settings['N'] = int(input("Number of problems to generate (N): "))
    settings['T'] = int(input("Number of training steps (T): "))
    settings['strategy'] = Strategy(int(input(
        "0 - Greedy\n1 - e-greedy\n2 - Optimistic Initial Values\n3 - Upper-Confidence Bound\n4 - Action Preferences\nAgent's strategy: ")))
    settings['diagnostics'] = input("Do you want to see diagnostics (y/n)? ")
    return settings


def get_best_action(arms):
    """ Finds the best action based on the expected reward.

    Args:
        arms (list): list containing bandit's arms

    Returns:
        int: index of the best action
    """

    best_arm = 0
    for i, a in enumerate(arms):
        if a.p > arms[best_arm].p:
            best_arm = i
    return best_arm


def show_diagnostics(t, n, strategy, gaussian, bernoulli):
    """ Calculates and displays experiment diagnostics

    Args:
        t (int):                Number of steps
        n (int):                Number of problems
        strategy (Strategy):    Strategy used
        gaussian (list):        List of dictionaries containg experiment results
        bernoulli (list):       List of dictionaries containg experiment results
    """
    gauss_rewards = []
    gauss_actions = []
    bern_rewards = []
    bern_actions = []

    for i in range(t):
        gauss_rewards.append(0)
        gauss_actions.append(0)
        bern_rewards.append(0)
        bern_actions.append(0)

        for g_res, b_res in zip(gaussian, bernoulli):
            gauss_rewards[-1] += g_res['rewards'][i]
            bern_rewards[-1] += b_res['rewards'][i]

            if g_res['actions'][i] == g_res['best-action']:
                gauss_actions[-1] += 1
            if b_res['actions'][i] == b_res['best-action']:
                bern_actions[-1] += 1

        gauss_rewards[-1] /= n
        gauss_actions[-1] /= n
        gauss_actions[-1] *= 100

        bern_rewards[-1] /= n
        bern_actions[-1] /= n
        bern_actions[-1] *= 100

    strategy_name = strategy.name.title()
    strategy_name = strategy_name.replace('_', ' ')

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'Diagnostics for {strategy_name} and Gaussian Bandits')

    ax1.set_title('Average rewards')
    ax1.set_xlabel('Learning step')
    ax1.set_ylabel('Reward')
    ax1.plot(gauss_rewards)

    ax2.set_title("Choosing best action")
    ax2.set_xlabel('Learning step')
    ax2.set_ylabel('% of correctly chosen actions')
    ax2.plot(gauss_actions)

    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.suptitle(f'Diagnostics for {strategy_name} and Bernoulli Bandits')

    ax1.set_title('Average rewards')
    ax1.set_xlabel('Learning step')
    ax1.set_ylabel('Reward')
    ax1.plot(bern_rewards)

    ax2.set_title("Choosing best action")
    ax2.set_xlabel('Learning step')
    ax2.set_ylabel('% of correctly chosen actions')
    ax2.plot(bern_actions)

    plt.show()


def run_problem(k, t, strategy):
    """ Runs a problem and trains the agent according to the provided settings.

    Args:
        k (int):                Number of arms
        t (int):                Number of training steps
        strategy (Strategy):    Agent's strategy

    Returns:
        dict: results for the Gaussian bandits
        dict: results for the Bernoulli bandits
    """

    gaussian_arms = [Arm(Distribution.GAUSSIAN)
                     for _ in range(k)]

    bernoulli_arms = [Arm(Distribution.BERNOULLI)
                      for _ in range(k)]

    gaussian_agent = Agent(strategy, gaussian_arms)
    bernoulli_agent = Agent(strategy, bernoulli_arms)

    gaussian_results = dict()
    bernoulli_results = dict()

    gaussian_results['actions'], gaussian_results['rewards'] = [], []
    bernoulli_results['actions'], bernoulli_results['rewards'] = [], []

    for _ in range(t):
        reward, action = gaussian_agent.choose_action()
        gaussian_results['rewards'] .append(reward)
        gaussian_results['actions'].append(action)
    gaussian_results['best-action'] = get_best_action(gaussian_arms)

    for _ in range(t):
        reward, action = bernoulli_agent.choose_action()
        bernoulli_results['rewards'].append(reward)
        bernoulli_results['actions'].append(action)
    bernoulli_results['best-action'] = get_best_action(bernoulli_arms)

    return gaussian_results, bernoulli_results


if __name__ == "__main__":
    settings = get_settings()

    total_gaussian_results = []
    total_bernoulli_results = []

    for _ in range(settings['N']):
        gaussian_results, bernoulli_results = run_problem(
            settings['k'], settings['T'], settings['strategy'])

        total_gaussian_results.append(gaussian_results)
        total_bernoulli_results.append(bernoulli_results)

    if settings['diagnostics'] == 'y':
        show_diagnostics(
            settings['T'], settings['N'], settings['strategy'], total_gaussian_results, total_bernoulli_results)
