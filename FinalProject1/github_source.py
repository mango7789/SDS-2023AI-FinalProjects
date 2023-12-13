import numpy as np
import math
import random
import time

def act(obs, config):

    def terminal(board, mark):
        COL=7
        ROW=6
        INAROW=4
        for row in range(ROW-1,-1,-1):
            for col in range(COL - INAROW + 1):
                window = list(board[row, col:col + INAROW])
                if window.count(mark)==4:
                    return True
        for row in range(ROW - INAROW,-1,-1):
            for col in range(COL):
                window = list(board[row:row + INAROW, col])
                if window.count(mark)==4:
                    return True
        for row in range(ROW - INAROW,-1,-1):
            for col in range(COL - INAROW + 1):
                window = list(board[range(row, row + INAROW), range(col, col + INAROW)])
                if window.count(mark)==4:
                    return True
        for row in range(INAROW - 1, ROW):
            for col in range(COL - INAROW + 1):
                window = list(board[range(row, row - INAROW, -1), range(col, col + INAROW)])
                if window.count(mark)==4:
                    return True
        return False

    def put_new_piece(grid, col, mark, config):
        next_state = grid.copy()
        for row in range(config.rows - 1, -1, -1):
            if not next_state[row][col]:
                next_state[row][col] = mark
                return next_state

    class MCTS():
        def __init__(self, obs, config):
            self.board = np.asarray(obs.board).reshape(config.rows, config.columns)
            self.config = config
            self.player = obs.mark
            self.time_limit = self.config.actTimeout - 0.2
            self.root_node = (0,)
            self.tree = {self.root_node: {
                            'board': self.board, 'player': self.player,'child': [], 'visits': 0,'reward': 0
                        }}
            self.total_visits = 0

        def ucb(self, node):
            total_visits=self.total_visits
            node_visits=self.tree[node]['visits']
            if not total_visits:
                return math.inf
            else:
                quality = self.tree[node]['reward'] / (node_visits+1)
                exploration = math.sqrt( 2 * math.log(total_visits) / (node_visits+1) )
                ucb_score = quality + 2 * exploration
                return ucb_score

        def selection(self):
            is_terminal = False
            leaf_node = (0,)
            while not is_terminal:
                node = leaf_node
                children=self.tree[node]['child']
                number_of_child = len(children)
                if not number_of_child:
                    leaf_node = node
                    is_terminal = True
                else:
                    max_ucb_score = -math.inf
                    best_action = leaf_node
                    for child in children:
                        child_node = leaf_node + (child,)
                        current_ucb = self.ucb(child_node)
                        if current_ucb > max_ucb_score:
                            max_ucb_score = current_ucb
                            best_action = child
                    leaf_node = leaf_node + (best_action,)
            return leaf_node

        def expansion(self, leaf_node):
            current_state = self.tree[leaf_node]['board']
            player_mark = self.tree[leaf_node]['player']
            current_board = np.asarray(current_state).reshape(config.rows * config.columns)
            actions = [c for c in range(self.config.columns) if not current_board[c]]
            is_terminal = terminal(current_state, player_mark)
            child_node = leaf_node
            is_availaible = False

            if len(actions) and not is_terminal:
                childs = []
                for action in actions:
                    child = leaf_node + (action,)
                    childs.append(action)
                    new_board = put_new_piece(current_state, action, player_mark, self.config)
                    self.tree[child] = {'board': new_board, 'player': player_mark,
                                           'child': [],'visits': 0, 'reward': 0
                                       }

                    if terminal(new_board, player_mark):
                        best_action = action
                        is_availaible = True

                self.tree[leaf_node]['child'] = childs

                if is_availaible:
                    child_node = best_action
                else:
                    child_node = random.choice(childs)

            return leaf_node + (child_node,)

        def simulation(self, child_node):
            self.total_visits += 1
            state = self.tree[child_node]['board']
            previous_player = self.tree[child_node]['player']

            is_terminal = terminal(state, previous_player)
            winning_player = previous_player
            count = 0

            while not is_terminal:
                current_board = np.asarray(state).reshape(config.rows * config.columns)
                self.actions_available = [c for c in range(self.config.columns) if not current_board[c]]
                if not len(self.actions_available) or count == 3:
                    return None
                else:
                    count += 1
                    current_player=int(3-previous_player)
                    for actions in self.actions_available:
                        state = put_new_piece(state, actions, current_player, self.config)
                        result = terminal(state, current_player)
                        if result:
                            return current_player

                previous_player = current_player

            return winning_player

        def backpropagation(self, child_node, winner):
            player = self.tree[(0,)]['player']
            if winner == None:
                reward = 0
            elif winner == player:
                reward = 2
            else:
                reward = -5
            node = child_node
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
            max_visits = -math.inf
            for action in actions:
                action = leaf_node + (action,)
                visit = self.tree[action]['visits']
                if visit > max_visits:
                    max_visits = visit
                    best_action = action
            return best_action[1]

    my_agent = MCTS(obs, config)
    return my_agent.game()


from kaggle_environments import evaluate, make

env = make("connectx", debug=True)
env.reset()
# Play as the first agent against default "random" agent.
env.run([act, "random"])
#
# def mean_reward(rewards):
#     return sum(r[0] for r in rewards) / float(len(rewards))
# from kaggle_environments import evaluate, make
#
# # Run multiple episodes to estimate its performance.
# print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [act, "random"], num_episodes=10)))
# print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [act, "negamax"], num_episodes=10)))