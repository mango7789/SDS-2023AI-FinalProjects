import numpy as np
import random
import time
from test_agent import act as act_2

COL = 7
ROW = 6
INAROW = 4
def terminal(board, mark):
    ## iterate from bottom to top to reduce time complexity
    # horizontal
    for row in range(ROW - 1, -1, -1):
        for col in range(COL - INAROW + 1):
            window = list(board[row, col:col + INAROW])
            if window.count(mark) == 4:
                return True
    # vertical
    for row in range(ROW - INAROW, -1, -1):
        for col in range(COL):
            window = list(board[row:row + INAROW, col])
            if window.count(mark) == 4:
                return True
    # positive-diagonal
    for row in range(ROW - INAROW, -1, -1):
        for col in range(COL - INAROW + 1):
            window = list(board[range(row, row + INAROW), range(col, col + INAROW)])
            if window.count(mark) == 4:
                return True
    # negative-diagonal
    for row in range(ROW - 1, INAROW - 2, -1):
        for col in range(COL - INAROW + 1):
            window = list(board[range(row, row - INAROW, -1), range(col, col + INAROW)])
            if window.count(mark) == 4:
                return True
    return False
def put_new_piece(state, action, mark, rows):
    next_state = state.copy()
    # find the legal row from bottom to top
    for row in range(rows - 1, -1, -1):
        if not next_state[row][action]:
            next_state[row][action] = mark
            return next_state

class MCTS():
    def __init__(self, obs, config):
        self.board = np.asarray(obs.board).reshape(config.rows, config.columns)
        self.player = obs.mark
        self.time_limit = config.actTimeout - 0.1
        self.rows = config.rows
        self.columns = config.columns
        self.root_node = (0,)
        self.tree = {self.root_node: {
            'board': self.board, 'player': self.player, 'child': [], 'visits': 0, 'reward': 0
        }}
        self.total_visits = 0

    def ucb(self, node):
        total_visits = self.total_visits
        node_visits = self.tree[node]['visits']
        if not total_visits or not node_visits:
            return np.inf
        else:
            value_estimates = self.tree[node]['reward'] / node_visits
            exploration = np.sqrt(2 * np.log(total_visits) / node_visits)
            # set the tunable number to 1
            ucb_score = value_estimates + exploration
            return ucb_score

    def selection(self):
        leaf_node = (0,)
        # repeated until a node that has at least one unsimulated child is reached
        while True:
            children = self.tree[leaf_node]['child']
            number_of_child = len(children)
            if not number_of_child:
                return leaf_node
            max_ucb = -np.inf
            best_action = leaf_node
            for child in children:
                child_node = leaf_node + (child,)
                current_ucb = self.ucb(child_node)
                if current_ucb > max_ucb:
                    max_ucb = current_ucb
                    best_action = child
            leaf_node = leaf_node + (best_action,)

    def expansion(self, leaf_node):
        current_board = self.tree[leaf_node]['board']
        player_mark = self.tree[leaf_node]['player']
        check_board = current_board[0, 0:self.columns]  # check if there are leagal actions
        actions = [c for c in range(self.columns) if not check_board[c]]
        is_terminal = terminal(current_board, player_mark)  # if current state is terminal,skip to the backpropagation
        child_node = leaf_node
        flag = 0

        if len(actions) and not is_terminal:
            # expand the current node and add new child nodes to MC tree
            for action in actions:
                child = leaf_node + (action,)
                new_board = put_new_piece(current_board, action, player_mark, self.rows)
                self.tree[child] = {'board': new_board, 'player': player_mark,
                                    'child': [], 'visits': 0, 'reward': 0
                                    }
                if terminal(new_board, player_mark):
                    best_action = action
                    flag = 1
            self.tree[leaf_node]['child'] = actions
            # if a child node reaches terminal state,return the child node.
            # else return a random choice of the child nodes
            if flag:
                child_node = best_action
            else:
                child_node = random.choice(actions)

        return leaf_node + (child_node,)

    def simulation(self, child_node):
        previous_board = self.tree[child_node]['board']
        previous_player = self.tree[child_node]['player']

        is_terminal = terminal(previous_board, previous_player)
        winning_player = previous_player
        count = 0
        # simulate until a terminal state or the max-depth(set to 3) is reached
        while not is_terminal:
            check_board = previous_board[0, 0:self.columns]
            actions = [c for c in range(self.columns) if not check_board[c]]
            if not len(actions) or count == 3:
                return None
            else:
                count += 1
                current_player = int(3 - previous_player)
                for action in actions:
                    previous_board = put_new_piece(previous_board, action, current_player, self.rows)
                    result = terminal(previous_board, current_player)
                    if result:
                        return current_player
            previous_player = current_player

        return winning_player

    def backpropagation(self, child_node, winner):
        self.total_visits += 1
        player = self.tree[(0,)]['player']
        # add the reward and visit times to the nodes along the action path
        if winner == None:
            reward = 1
        elif winner == player:
            reward = 2
        else:
            reward = -10
        for i in range(len(child_node)):
            node = child_node[0:i + 1]
            self.tree[node]['visits'] += 1
            self.tree[node]['reward'] += reward

    def game(self):
        self.initial_time = time.time()
        is_expanded = False
        while time.time() - self.initial_time < self.time_limit:
            node = self.selection()
            if not is_expanded:
                node = self.expansion(node)
                is_expanded = True
            winner = self.simulation(node)
            self.backpropagation(node, winner)
        leaf_node = (0,)
        actions = self.tree[leaf_node]['child']
        random.shuffle(actions)
        max_reward = -np.inf
        for action in actions:
            action = leaf_node + (action,)
            reward = self.tree[action]['reward']
            if reward > max_reward:
                max_reward = reward
                best_action = action
        return best_action[1]
def act(obs, config):
    agent = MCTS(obs, config)
    return agent.game()


# from kaggle_environments import evaluate, make
#
# env = make("connectx", debug=True)
# env.reset()
# # Play as the first agent against default "random" agent.
# env.run([act, "random"])

def mean_reward(rewards):
    return sum(r[0] for r in rewards) / float(len(rewards))
from kaggle_environments import evaluate, make

# Run multiple episodes to estimate its performance.
print("My Agent vs Agent2:", mean_reward(evaluate("connectx", [act, act_2], num_episodes=10)))
