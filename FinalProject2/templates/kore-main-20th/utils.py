import numpy as np


PAR = {'max_lookahead': 30, 'board_size': 21, 'base_build_bias': 1.0, 'capture_bias': 0.0, 'base_max_dist': 8,
       'needed_yards': 2, 'discount_rate': 1.01, 'starvation_discount': 1.02, 'min_base_value': 0.0,
       'phantom_income': True, 'risk_adjustment': True, 'avg_flight_time': 10, 'build_delay': 6, 'turn_penalty': False}  # Discount val hardcoded in cygraph for speed reasons

SIZE = PAR['board_size']
LOOK = PAR['max_lookahead']
FACES = 5
BAD_VAL = -999999.0

NONE = 0
KING = 1
QUEEN = 2
JACK = 3
JOKER = 4


dist_matrix_initialized = False

class DistanceManager:
    def __init__(self):
        self.dist_matrix = np.zeros((21, 21, 21, 21), dtype=np.intc)
        for x_1 in range(0, 21):
            for y_1 in range(0, 21):
                for x_2 in range(0, 21):
                    for y_2 in range(0, 21):
                        self.dist_matrix[x_1, y_1, x_2, y_2] = min(abs(x_1 - x_2), abs(21 + x_1 - x_2),
                                                              abs(21 + x_2 - x_1)) + min(abs(y_1 - y_2),
                                                                                         abs(21 + y_1 - y_2),
                                                                                         abs(21 + y_2 - y_1))

    def calc_distance(self, x1, y1, x2, y2):
        return self.dist_matrix[x1, y1, x2, y2]



# A simple implementation of Priority Queue
# using Queue.
class PriorityQueue(object):
    def __init__(self):
        self.queue = []
        self.value = []
        self.extra = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0

    # for inserting an element in the queue
    def insert(self, data, value, extra=None):
        self.queue.append(data)
        self.value.append(value)
        self.extra.append(extra)

    # for popping an element based on Priority
    def pop(self):
        max_val = 0
        for i in range(len(self.queue)):
            if self.value[i] > self.value[max_val]:
                max_val = i
        item = self.queue[max_val]
        value = self.value[max_val]
        extra = self.extra[max_val]
        del self.queue[max_val]
        del self.value[max_val]
        del self.extra[max_val]
        if extra is not None:
            return item, value, extra
        else:
            return item