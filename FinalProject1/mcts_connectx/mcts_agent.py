import time
import random
import math

def act(observation, configuration):
    '''
    1.The parameters used frequently in following functions like rows,columns,inarow are recorded by "constant" variables
    to reduce time complexity;
    2.The functions which include manipulation of board are defined beyond the scope of class MCTS to prevent bugs/errors
    (i.e. put_new_piece/default_policy_simulation);
    3.Class MCTS are composed of selection,expansion,simulation,backpropagation plus some condition check functions
    play_game is the control part of the entire class;choose_action_child is also included in the class since we need to
    return a global current_state.
    '''
    # record the current time first to increase the accuracy on the check of overagetime in "while" iteration
    initial_time = time.time()
    ROW = configuration.rows
    COL = configuration.columns
    INAROW = configuration.inarow
    tunable_number = 1 # the tunable parameter in calculating the value of UCB

    def opponent_mark(mark):
        '''
        return the opponent mark of the current mark,change between 1 and 2
        '''
        return 3 - mark

    def opponent_score(score):
        '''
        return the oppnent score of the current score,change between 0,0.5,1
        '''
        return 2 - score

    def get_put_row(board,column):
        '''
        find the feasible value of row(maximum row) to put a new piece based on the given column
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
        since the action(column) is recorded,we only need O(INAROW) time to check if our agent wins the game
        '''
        inarow = INAROW - 1 # only need INAROW-1 entries in one direction
        row = get_win_row(board,mark,column)

        def count(offset_row, offset_column):
            '''
            the input parameter can be thought as our check direction
            offset_row:
            0:horizontal  1:vertical-to-bottom   -1:vertical-to-top
            offset_col:
            0:vertical    1:horizontal-to-right  -1:horizontal-to-left
            '''
            for i in range(1, inarow + 1):
                # start from the action to margin
                r = row + offset_row * i
                c = column + offset_column * i
                # if the coordinates(r,c) reach out of the board or the mark in the current position is not our check mark
                # return the counted number of consecutive marks
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
                # we only need INAROW-1 same marks except for the current mark to confirm a win state
                count(1, 0) >= inarow  # vertical
                or (count(0, 1) + count(0, -1)) >= inarow  # horizontal
                or (count(-1, -1) + count(1, 1)) >= inarow  # positive-diagonal
                or (count(-1, 1) + count(1, -1)) >= inarow  # negative-diagonal
        )

    def is_tie(board):
        '''
        check if the game is in tie state
        '''
        return not (any(mark == 0 for mark in board))

    def get_score(board, column, mark):
        '''
        return the state and score of current board
        '''
        if is_win(board, column, mark):
            return (True, 2)
        if is_tie(board):
            return (True, 1)
        else:
            return (False, 0)

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

    def opponent_action(new_board, old_board):
        '''
        find the last action taken by our opponent agent
        '''
        for i, piece in enumerate(new_board):
            if piece != old_board[i]:
                return i % COL

    class MCTS():
        def __init__(self, board, mark, parent=None, is_terminal=False, terminal_score=None, action=None):
            self.board = board.copy()
            self.mark = mark
            self.children = []
            self.parent = parent
            self.available_moves = [c for c in range(COL) if board[c] == 0]
            self.expandable_moves = self.available_moves.copy()
            self.score = 0
            self.visits = 0
            self.is_terminal = is_terminal
            self.terminal_score = terminal_score
            self.action = action

        def selection(self, action):
            '''
            select the child node taken by our opponent
            '''
            for child in self.children:
                if child.action == action:
                    return child
            return None

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
            '''
            simulate until the terminal state is reached
            '''
            if self.is_terminal:
                return self.terminal_score
            return opponent_score(default_policy_simulation(self.board, self.mark))

        def backpropagation(self, simulation_score):
            '''
            backpropagate until the root Node is reached
            '''
            self.score += simulation_score
            self.visits += 1
            if self.parent:
                self.parent.backpropagation(opponent_score(simulation_score))

        def expand_and_simulate(self):
            '''
            combine the expansion and simulation process
            '''
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
            '''
            the main function in class MCTS
            '''
            if self.is_terminal:
                # if a terminal state is reached,skip to the backpropagation
                self.backpropagation(self.terminal_score)
                return
            if self.is_expandable():
                # if the current node is expandable,choose a child to expand
                self.expand_and_simulate()
                return
            # if the current node is not terminal and not expandable,choose the child with
            # highest UCB score to play the game
            self.choose_best_child().play_game()

        def choose_action_child(self):
            '''
            choose the best action for current state
            the return parameter is a MCTS class for the sake of recording the current Monte Carlo Tree
            '''
            children_scores = [child.score for child in self.children]
            max_score = max(children_scores)
            index = children_scores.index(max_score)
            return self.children[index]

    board = observation.board
    mark = observation.mark

    global current_state # use global variable to store the former MCTS information

    try:
        # if current_state already exists,let it be the root node
        current_state = current_state.selection(opponent_action(board, current_state.board))
        current_state.parent = None
    except:
        # if current_state==None,initialize it based on the observed board and mark
        current_state = MCTS(board, mark, parent=None, is_terminal=False, terminal_score=None, action=None)

    time_limit = configuration.actTimeout # the time limit for each action
    # remainOvertime = observation.remainingOverageTime
    # step = observation.step
    # # there is no need to leave overagetime for step 1 and step 2
    # # allocate 3s for each actiontime(approximate 20 steps)
    # time_available = (3*(step!=1 and step!=2) if remainOvertime>3.3 else 0)
    time_available = 0.3

    # iterates until the given time runs out
    while time.time() - initial_time < time_limit + time_available:
        current_state.play_game()

    # use the current_state to store the information of the best child node taken by our agent
    # and the global property of current_state can help us reuse the Monte Carlo Tree we built
    # in the current "act" for next "act" to reduce time complexity
    current_state = current_state.choose_action_child()
    # return the action taken from the root to the best child node
    return current_state.action
