import random
from constants import *


class Arm():
    def __init__(self, distribution):
        """ Constructor for the arm of a bandit.
            We precalculate all the required values for all the strategies
            in order to allow strategy switching during exectution. 
            Fun stuff can be observed this way.

        Args:
            distribution (Distribution): specifies the type of distribution
        """
        self.distribution = distribution

        # Used for both Gaussian and Bernoulli
        self.p = random.uniform(0, 1)
        # Used for Gaussian
        self.std = random.uniform(0, 1)

    def get_reward(self):
        """ Calculates the reward from an agent

        Returns:
            float: reward
        """
        if self.distribution == Distribution.GAUSSIAN:
            return self.__sample_from_gaussian()
        else:
            return self.__sample_from_bernoulli()

    def __sample_from_gaussian(self):
        """ Sample the reward from gaussian distribution

        Returns:
            float: reward
        """
        return random.gauss(self.p, self.std)

    def __sample_from_bernoulli(self):
        """ Samples the reward using the Bernoulli distribution

        Returns:
            int: reward
        """
        return 1 if random.uniform(0, 1) < self.p else 0
