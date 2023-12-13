import math
import random
import time

def act(obs,config):
    initial_time=time.time()
    timelimit=config.timeout-0.34
    tunable_constant=1
    ROW=6
    COLUMN=7
    INAROW=4

    def put_new_piece(board,column,mark):
        row=max([r for r in range(ROW) if board[column+r*COLUMN] == 0 ])
        board[column+row*COLUMN] = mark

    def is_win(board, column, mark):
        """ Checks for a win. Taken from the Kaggle environment. """
        row = min([r for r in range(ROW) if board[column + r * COLUMN] == mark])

        def count(offset_row, offset_column):
            for i in range(1, INAROW):
                r = row + offset_row * i
                c = column + offset_column * i
                if (
                        r < 0
                        or r >= ROW
                        or c < 0
                        or c >= COLUMN
                        or board[c + r * COLUMN] != mark
                ):
                    return i - 1
            return INAROW-1

        return (
                count(1, 0) >= INAROW-1  # vertical
                or (count(0, 1) + count(0, -1)) >= INAROW-1  # horizontal
                or (count(-1, -1) + count(1, 1)) >= INAROW-1  # positive-diagonal
                or (count(-1, 1) + count(1, -1)) >= INAROW-1  # negative-diagonal
        )

    def is_tie(board):
        return not (any(mark == 0) for mark in board)

    def get_score(board,column,mark):
        if is_win(board,column,mark):
            return (True,1)
        if is_tie(board):
            return (True,0.5)
        return (False,None)

    def get_ucb(score,visits,total_visits):
        if total_visits==0:
            return math.inf
        return (score/visits + tunable_constant*math.sqrt(2*math.log(total_visits)/visits))

    def random_action(board):
        return random.choice([c for c in range(COLUMN) if board[c] == 0])

    def defult_policy_simulation(board,mark):
        new_mark = mark
        board = board.copy()
        column = random_action(board)
        put_new_piece(board, column, new_mark)
        is_terminal, score = get_score(board, column, new_mark)
        while not is_terminal:
            new_mark = 3 - new_mark
            column = random_action(board)
            put_new_piece(board, column, new_mark)
            is_terminal, score = get_score(board, column, new_mark)
        if new_mark == mark:
            return score
        return 1 - score

    def action_by_opponent(new_board,old_board):
        for i,piece in enumerate(new_board):
            if piece!=old_board[i]:
                return i%COLUMN
        return -1

    class MCTS():
        def __init__(self,board,mark,is_terminal=False,terminal_score=None,action=None,parent=None):
            self.board=board.copy()
            self.mark=mark
            self.score=0
            self.visits=0
            self.available_actions=[c for c in range(COLUMN) if board[c] == 0]
            self.expandable_actions=self.available_actions.copy()
            self.is_terminal=is_terminal
            self.terminal_score=terminal_score
            self.action=action
            self.children=[]
            self.parent=parent

        def selection(self,action):
            for child in self.children:
                if child.action==action:
                    return child
            return None

        def choose_best_child(self):
            max_ucb=-math.inf
            best_child=None
            for child in self.children:
                child_ucb=get_ucb(child.score,child.visits,self.visits)
                if child_ucb>max_ucb:
                    max_ucb=child_ucb
                    best_child=child
            return best_child

        def simulate(self):
            if self.is_terminal:
                return self.terminal_score
            return 1-defult_policy_simulation(self.board,self.mark)

        def backpropagation(self,simulation_score):
            self.score+=simulation_score
            self.visits+=1
            if self.parent is not None:
                self.parent.backpropagation(1-simulation_score)

        def expand_simulate(self):
            column=random.choice(self.expandable_actions)
            child_board=self.board.copy()
            put_new_piece(child_board,column,self.mark)
            is_terminal,terminal_score=get_score(child_board,column,self.mark)
            self.children.append(
                MCTS(child_board,3-self.mark,is_terminal=is_terminal,terminal_score=terminal_score,action=column,parent=self)
            )
            simulation_score=self.children[-1].simulate()
            self.children[-1].backpropagation(simulation_score)
            self.expandable_actions.remove(column)

        def play_game(self):
            if self.is_terminal:
                self.backpropagation(self.terminal_score)
                return
            if len(self.expandable_actions)>0:
                self.expand_simulate()
                return
            self.choose_best_child().play_game()

        def choose_action(self):
            max_score = -math.inf
            best_child = None
            for child in self.children:
                child_score = child.score
                if child_score > max_score:
                    max_score = child_score
                    best_child = child
            return best_child.action

    board=obs.board
    mark=obs.mark

    global current_state

    try:
        current_state=current_state.selection(action_by_opponent(board,current_state.board))
        current_state.parent=None
    except:
        current_state=MCTS(board,mark,is_terminal=False,terminal_score=None,action=None,parent=None)

    while time.time()-initial_time < timelimit:
        current_state.play_game()

    return current_state.choose_action()

# def mean_reward(rewards):
#     return sum(r[0] for r in rewards) / float(len(rewards))
# from kaggle_environments import evaluate, make
#
# # Run multiple episodes to estimate its performance.
# print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [act, "random"], num_episodes=10)))
# print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [act, "negamax"], num_episodes=10)))

from kaggle_environments import evaluate, make

env = make("connectx", debug=True)
env.reset()
# Play as the first agent against default "random" agent.
env.run([act, "random"])