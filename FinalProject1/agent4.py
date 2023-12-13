import random
import math
import time
from test_agent4 import act as act_test4
def act(observation, configuration):
    initial_time = time.time()
    remainOvertime=observation.remainingOverageTime
    ROW=configuration.rows
    COL=configuration.columns
    INAROW=configuration.inarow
    time_limit = configuration.actTimeout
    tunable_number=1

    def opponent_mark(mark):
        '''
        return the opponent mark of the current mark
        '''
        return 3 - mark

    def opponent_score(score):
        '''
        return the oppnent score of the current score
        '''
        return 2 - score

    def get_put_row(board,column):
        '''
        find the feasible value of row to put a new piece based on the given column
        '''
        for r in range(ROW-1,-1,-1):
            if board[column+r*COL] == 0:
                return r

    def put_new_piece(board, column, mark):
        '''
        put a new piece based on the given column
        '''
        row = get_put_row(board,column)
        board[column + row * COL] = mark

    def get_win_row(board,mark,column):
        '''
        find the highest row(minimum row) of our mark based on the given column
        '''
        for r in range(ROW):
            if board[column+r*COL] == mark:
                return r

    def is_win(board, column, mark):
        '''
        check if our agent wins the game
        '''
        inarow = INAROW - 1
        row = get_win_row(board,mark,column)

        def count(offset_row, offset_column):
            for i in range(1, inarow + 1):
                r = row + offset_row * i
                c = column + offset_column * i
                if (
                        r < 0
                        or r >= ROW
                        or c < 0
                        or c >= COL
                        or board[c + (r * COL)] != mark
                ):
                    return i - 1
            return inarow

        return (
                count(1, 0) >= inarow  # vertical
                or (count(0, 1) + count(0, -1)) >= inarow  # horizontal
                or (count(-1, -1) + count(1, 1)) >= inarow  # positive-diagonal
                or (count(-1, 1) + count(1, -1)) >= inarow  # negative-diagonal
        )

    def is_tie(board):
        '''
        check if the game is in tie state
        '''
        return not (any(mark==0 for mark in board))

    def get_score(board, column, mark):
        '''
        get the state and score of current board
        '''
        if is_win(board, column, mark):
            return (True, 2)
        if is_tie(board):
            return (True, 1)
        else:
            return (False, None)

    def get_ucb(node_score, node_visits, total_visits):
        '''
        compute the UCB of the given child node
        '''
        if node_visits == 0:
            return math.inf
        return (node_score / node_visits + tunable_number *
                math.sqrt( 2 * math.log(total_visits) / node_visits))

    def random_action(board):
        '''
        randomly choose a legal action
        '''
        return random.choice([c for c in range(COL) if board[c] == 0])

    def default_policy_simulation(board, mark):
        '''
        the simulation policy used in MCTS.simulate()
        '''
        original_mark = mark
        board = board.copy()
        column = random_action(board)
        put_new_piece(board, column, mark)
        is_terminal, score = get_score(board, column, mark)
        while not is_terminal:
            mark = opponent_mark(mark)
            column = random_action(board)
            put_new_piece(board, column, mark)
            is_terminal, score = get_score(board, column, mark)
        if mark == original_mark:
            return score
        return opponent_score(score)

    class MCTS():
        def __init__(self, board, mark, parent=None, is_terminal=False, terminal_score=None, action=None):
            self.board = board.copy()
            self.mark = mark
            self.children = []
            self.parent = parent
            self.score = 0
            self.visits = 0
            self.available_moves = [c for c in range(COL) if board[c] == 0]
            self.expandable_moves = self.available_moves.copy()
            self.is_terminal = is_terminal
            self.terminal_score = terminal_score
            self.action = action

        def choose_best_child(self):
            '''
            choose a child with the max UCB score
            '''
            children_scores = [get_ucb(child.score,child.visits,self.visits) for child in self.children]
            max_score = max(children_scores)
            index = children_scores.index(max_score)
            return self.children[index]

        def is_expandable(self):
            '''
            check if current node is expandable
            '''
            return len(self.expandable_moves) > 0

        def simulate(self):
            if self.is_terminal:
                return self.terminal_score
            return opponent_score(default_policy_simulation(self.board, self.mark))

        def backpropagation(self, simulation_score):
            self.score += simulation_score
            self.visits += 1
            if self.parent:
                self.parent.backpropagation(opponent_score(simulation_score))

        def expand_and_simulate(self):
            column = random.choice(self.expandable_moves)
            child_board = self.board.copy()
            put_new_piece(child_board, column, self.mark)
            is_terminal, terminal_score = get_score(child_board, column, self.mark)
            self.children.append(
                MCTS(child_board, opponent_mark(self.mark),parent=self,is_terminal=is_terminal,terminal_score=terminal_score,action=column)
            )
            simulation_score = self.children[-1].simulate()
            self.children[-1].backpropagation(simulation_score)
            self.expandable_moves.remove(column)

        def play_game(self):
            if self.is_terminal:
                self.backpropagation(self.terminal_score)
                return
            if self.is_expandable():
                self.expand_and_simulate()
                return
            self.choose_best_child().play_game()

        def choose_child_action(self):
            '''
            choose the best action for current state
            '''
            children_scores = [child.score for child in self.children]
            max_score = max(children_scores)
            index = children_scores.index(max_score)
            return self.children[index]

    board = observation.board
    mark = observation.mark
    global current_state

    current_state = MCTS(board, mark, parent=None, is_terminal=False, terminal_score=None, action=None)
    count=0

    while time.time() - initial_time < time_limit + (0.3 if remainOvertime > 0.8 else 0):
        count += 1
        current_state.play_game()

    current_state = current_state.choose_child_action()
    return current_state.action

from kaggle_environments import evaluate, make


def mean_reward(rewards):
    return sum(r[0] for r in rewards) / float(len(rewards))
from kaggle_environments import evaluate, make

# Run multiple episodes to estimate its performance.
print("agent4 vs agent4_visits:", mean_reward(evaluate("connectx", [act, act_test4], num_episodes=10)))