from enum import Enum

EPSILON = 0.2
OPTIMISTIC_SUM = 10
OPTIMISTIC_N = 1
C = 1
ALPHA = 0.1


class Distribution(Enum):
    GAUSSIAN = 0
    BERNOULLI = 1


class Strategy(Enum):
    GREEDY = 0
    EGREEDY = 1
    OPTIMISTIC_INITIAL_VALUES = 2
    UPPER_CONFIDENCE_BOUND = 3
    ACTION_PREFERENCES = 4
