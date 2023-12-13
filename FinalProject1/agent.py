import numpy as np
import time
import random
import math
from test_agent import act as act_test

class mcts():

    def __init__(self,obs,config):
        self.columns=config.columns
        self.rows=config.rows
        self.inarow=config.inarow
        self.timelimit=config.actTimeout-0.3
        self.mark=obs.mark
        self.board=np.asarray(obs.board).reshape(self.rows,self.columns)
        # construct the Monte Carlo Tree based on a dict
        self.root=(0,)
        self.tree={self.root:{
            "board": self.board,"mark":self.mark,"score":0,"visits":0,"children":[]
        }}
        self.total_visits=0

    def is_terminal(self,board,mark):
        ROW=self.rows
        COL=self.columns
        INAROW=self.inarow
        # Iterate from bottom to top to reduce the time complexity.
        # horizontal
        for row in range(ROW-1,-1,-1):
            for col in range(COL-INAROW+1):
                check=list(board[row,col:col+INAROW])
                if check.count(mark)==INAROW:
                    return True
        # vertical
        for row in range(ROW-1,INAROW-2,-1):
            for col in range(COL-INAROW+1):
                check=list(board[row-INAROW+1:row+1,col])
                if check.count(mark)==INAROW:
                    return True
        # positive-diagonal
        for row in range(ROW-INAROW,-1,-1):
            for col in range(COL-INAROW+1):
                check=[board[row+c,col+c] for c in range(INAROW)]
                if check.count(mark)==INAROW:
                    return True
        # negatice-diagonal
        for row in range(ROW-INAROW,-1,-1):
            for col in range(COL-INAROW+1):
                check=[board[row-c,col+c] for c in range(INAROW)]
                if check.count(mark)==INAROW:
                    return True

        return False

    def put_new_pieces(self,board,action,mark):
        new_board=board.copy()
        for row in range(self.rows-1,-1,-1):
            if new_board[row][action]==0:
                new_board[row][action]=mark
                return new_board

    def ucb(self,node):
        node_visited_times=self.tree[node]["visits"]
        total_visited_times=self.total_visits
        if node_visited_times==0 or total_visited_times==0:
            return math.inf
        # the tunable constant can be chosen differently for different games
        tunable_constant=2
        return (self.tree[node]["score"]/node_visited_times+
                tunable_constant*math.sqrt(2*math.log(total_visited_times)/node_visited_times))

    def selection(self):
        node=(0,)
        while True:
            children=self.tree[node]["children"]
            number_of_children=len(children)
            if number_of_children==0:
                return node
            max_ucb=-math.inf
            best_child=node
            for child in children:
                new_node=node+(child,)
                child_ucb=self.ucb(new_node)
                if child_ucb>=max_ucb:
                    max_ucb=child_ucb
                    best_child=new_node
            node=best_child

    def expansion(self,node):
        current_board=self.tree[node]["board"]
        current_mark=self.tree[node]["mark"]
        check_board=current_board[0,0:self.columns]
        actions=[col for col in range(self.columns) if check_board[col]==0]
        if self.is_terminal(current_board,current_mark) or len(actions)==0:
            return node+(node,)
        flag=0
        best_child=node
        for action in actions:
            child_node=node+(action,)
            new_board=self.put_new_pieces(current_board,action,current_mark)
            # only expand one root node,so we don't need to change the mark
            self.tree[child_node]={
                "board":new_board,"mark":current_mark,"score":0,"visits":0,"children":[]
            }
            if self.is_terminal(new_board,current_mark):
                flag=1
                best_child=child_node
        self.tree[node]["children"] = actions

        if flag==1:
            return best_child
        return node+(random.choice(actions),)

    def simulation(self,node):
        current_board=self.tree[node]["board"]
        current_mark=self.tree[node]["mark"]
        is_terminal=self.is_terminal(current_board,current_mark)
        winner=current_mark
        count=0
        while not is_terminal:
            check_board = current_board[0, 0:self.columns]
            actions=[col for col in range(self.columns) if check_board[col] == 0]
            if len(actions)==0 or count==3:
                return None
            count += 1
            new_mark=int(3-current_mark)
            for action in actions:
                current_board = self.put_new_pieces(current_board, action, new_mark)
                is_terminal = self.is_terminal(current_board, new_mark)
                if is_terminal:
                    winner=new_mark
                    return winner
            current_mark=new_mark

        return winner

    def backpropagation(self,node,winner):
        self.total_visits+=1
        mark=self.tree[(0,)]["mark"]
        # the score can be changed by the results of the evaluation
        if winner==mark:
            score=1
        elif winner==None:
            score=0
        else:
            score=-10
        self.tree[node]["score"]+=score
        self.tree[node]["visits"]+=1

    def game(self):
        initial_time=time.time()
        is_expanded=False
        while time.time()-initial_time<self.timelimit:
            node=self.selection()
            # just expand the root node considering the time limit
            if not is_expanded:
                node=self.expansion(node)
                is_expanded=True
            winner=self.simulation(node)
            self.backpropagation(node,winner)

        leaf_node=(0,)
        children=self.tree[leaf_node]["children"]
        max_score=-math.inf
        for child in children:
            next_node=leaf_node+(child,)
            score=self.tree[next_node]["visits"]
            if score>max_score:
                max_score=score
                best_children=next_node
        return best_children[1]

def act(obs,config):
    agent=mcts(obs,config)
    return agent.game()
#
from kaggle_environments import evaluate, make
def mean_reward(rewards):
    return sum(r[0] for r in rewards) / float(len(rewards))

# Run multiple episodes to estimate its performance.
print("My Agent vs test Agent:", mean_reward(evaluate("connectx", [act, act_test], num_episodes=10)))
