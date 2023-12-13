import math
import random

from constants import *


class Agent():
    def __init__(self, strategy, arms, eps=EPSILON, optimistic_sum=OPTIMISTIC_SUM, optimistic_n=OPTIMISTIC_N, c=C, alpha=ALPHA):
        """ Constructs an agent.

        Args:
            strategy (Strategy):    Agent's exploration strategy
            bandits (list):         List containing all arms (population)
            eps:                    Used in e-greedy. Defines the probability of choosing a random action. Default: 0.2
            optimistic_sum:         Used in Optimistic Initial Values. Specifies the initial summed reward. 
                                        The bigger, the longer it takes to converge. Default: 10
            optimistic_n:           Used in Optimistic Initial Values. Specifies the initial count of actions. 
                                        The bigger, the longer it takes to converge. Default: 1
            c:                      Used in Upper-Confidence Bound. Regulates the amount of exploration. Default: 1
            alpha:                  Used in Action Preferences. Default: 0.1
        """

        self.strategy = strategy
        self.arms = arms

        # Stores past actions data
        self.actions_histogram = [0 for _ in arms] if strategy != Strategy.OPTIMISTIC_INITIAL_VALUES else [
            optimistic_n for _ in arms]

        # Stores expected rewards using Sample-Average Method.
        # Each tuple (implemented as a lsit) contains: (sum_of_rewards, number_of_times_a_was_taken)
        self.rewards = [0 for _ in arms] if strategy != Strategy.OPTIMISTIC_INITIAL_VALUES else [
            optimistic_sum for _ in arms]

        # Used for e-greedy
        self.eps = eps

        # Used for Upper-Confidence Bound
        self.c = c

        # Used for Action Preferences
        self.preferences = [0 for _ in arms]
        self.alpha = alpha

    def choose_action(self):
        """ Chooses the best action according to applied strategy.

        Returns:
            float:  reward obtained from the chosen action
            int:    index of the chosen arm
        """

        if self.strategy == Strategy.GREEDY:
            return self.__choose_greedy()
        elif self.strategy == Strategy.EGREEDY:
            return self.__choose_e_greedy()
        elif self.strategy == Strategy.OPTIMISTIC_INITIAL_VALUES:
            return self.__choose_greedy()
        elif self.strategy == Strategy.UPPER_CONFIDENCE_BOUND:
            return self.__choose_upper_confidence_bound()
        else:
            return self.__choose_action_preferences()

    def __choose_greedy(self):
        """ Chooses the action according to the greedy strategy

        Returns:
            float:  reward obtained from the chosen action
            int:    index of the chosen arm
        """
        best_arm_index = 0
        best_expected_reward = -math.inf

        for i, r in enumerate(self.rewards):
            # We set expected_reward to infinity if we have never tried something
            expected_reward = math.inf if self.actions_histogram[i] == 0 else r / \
                self.actions_histogram[i]

            if expected_reward > best_expected_reward:
                best_arm_index = i
                best_expected_reward = expected_reward

        best_arm = self.arms[best_arm_index]
        actual_reward = best_arm.get_reward()

        self.rewards[best_arm_index] += actual_reward
        self.actions_histogram[best_arm_index] += 1

        return actual_reward, best_arm_index

    def __choose_e_greedy(self):
        """ Chooses the action using e-greedy strategy.
            Behaves randomly at a defined probability. 

        Returns:
            float:  reward obtained from the chosen action
            int:    index of the chosen arm
        """
        if random.uniform(0, 1) < 1 - self.eps:
            # Agent behaves greedily
            return self.__choose_greedy()
        else:
            action = random.randint(0, len(self.arms) - 1)

            reward = self.arms[action].get_reward()
            self.rewards[action] += reward
            self.actions_histogram[action] += 1

            return reward, action

    def __choose_upper_confidence_bound(self):
        """ Chooses the action using Upper-Confidence-Bound strategy

        Returns:
            float:  reward obtained from the chosen action
            int:    index of the chosen arm
        """
        best_arm_index = 0
        best_expected_reward = -math.inf

        for i, r in enumerate(self.rewards):
            # We set expected_reward to infinity if we have never tried something
            expected_reward = math.inf
            uncertainty = 0

            if self.actions_histogram[i] != 0:
                timestep = sum(self.actions_histogram)
                uncertainty = self.c * \
                    math.sqrt(math.log(timestep) / self.actions_histogram[i])

                expected_reward = self.rewards[i] / \
                    self.actions_histogram[i] + uncertainty

            if expected_reward > best_expected_reward:
                best_arm_index = i
                best_expected_reward = expected_reward

        best_arm = self.arms[best_arm_index]
        actual_reward = best_arm.get_reward()

        self.rewards[best_arm_index] += actual_reward
        self.actions_histogram[best_arm_index] += 1

        return actual_reward, best_arm_index

    def __choose_action_preferences(self):
        """ Chooses the best action based on Action Preference

        Returns:
            float:  reward obtained from the chosen action
            int:    index of the chosen arm
        """
        policy = self.__policy_from_preferences()

        random_value = random.uniform(0, 1)
        value = policy[0]

        best_action = 0
        for p in policy[1:]:
            if value > random_value:
                break
            best_action += 1
            value += p

        reward = self.arms[best_action].get_reward()

        self.__update_preferences(policy, best_action, reward)

        self.actions_histogram[best_action] += 1
        self.rewards[best_action] += reward

        return reward, best_action

    def __policy_from_preferences(self):
        """ Calculates the policy using the preferences.
            Based on Boltzman distribution.

        Returns:
            list: agent's policy 
        """
        policy = []
        exponentials = []
        for p in self.preferences:
            exponentials.append(math.exp(p))
        denumerator = sum(exponentials)

        for e in exponentials:
            policy.append(e / denumerator)

        return policy

    def __update_preferences(self, policy, action, reward):
        """ Updates agent's preferences.

        Args:
            policy (list):      current agent's policy
            action (int):       index of a chosen action
            reward (float):     received reward
        """

        average_reward = 0 if sum(self.actions_histogram) == 0 else sum(self.rewards) / \
            sum(self.actions_histogram)

        for i, p in enumerate(self.preferences):
            if i == action:
                self.preferences[i] = p + self.alpha * \
                    (reward - average_reward) * (1 - policy[i])
            else:
                self.preferences[i] = p - \
                    self.alpha * (reward - average_reward) * policy[i]
