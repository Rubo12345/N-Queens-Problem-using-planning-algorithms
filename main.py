from mimetypes import init
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
from collections import deque
import math
import Attacking_Queens

class Queen:
    def __init__(self):
        self.row = 0                    #row coordinate of queen
        self.col = 0                    #col coordinate of queen
        self.board_size = 0             #board size
        self.positions = []             #initial positions of queens in the board                    

    def setcoords(self, m):
        self.row = m[0]
        self.col = m[1]

    def getpositions(self, current_board):
        '''Given a board, find current position of n queens'''
        
        self.board_size = len(current_board)
        for x in range(self.board_size):
            for y in range(self.board_size):
                if current_board[x][y] != 0:
                    self.positions.append((x,y))

    def is_valid(self, position):
        '''Check if the new position is within the chessboard'''

        if position[0] in range(self.board_size) and position[1] in range(self.board_size):
            return True

    def movequeen(self, i, board):
        '''Move one queen along the column. 
        If the position is valid, create a board with the new configuration'''

        new_position = (self.row + i, self.col)
        if new_position[0] != self.row:
            if self.is_valid(new_position):
                new_board = board.copy()
                new_board[new_position[0], new_position[1]] = board[self.row, self.col]
                new_board[self.row, self.col] = 0
            else:
                new_board = board
            return new_board

def plot(board):

    #Show chess board
    chessboard = np.array([[(i+j)%2 for i in range(len(board))] for j in range(5)])
    plt.imshow(chessboard,cmap='ocean')
    plt.show()

def generate_configuration(n, weight_range):
    
    #Generate a nxn board with a random queen with random weight in each column.
    #0 is an empty space
    board = np.zeros([n,n])

    for i in range(0, n):
        row_index = random.randrange(0, n, 1)
        board[row_index, i] = random.randrange(1, weight_range, 1)

    return board

def bfs(init_board_state, board_size):

    queens = Queen()
    #Initialize queue and visited board configuration
    visited = []
    queue = []

    #append initial configuration
    visited.append(init_board_state)
    queue.append(init_board_state)
    
    while queue:

        #pop the first element from the queue and mark it as visited
        current_state = queue.pop(0)

        #get positions of queens in current board
        queens.getpositions(current_state)
    
        for i in range(len(queens.positions)):
            for queen in queens.positions:
                for j in range(-board_size+1, board_size):
                    queens.setcoords(queen)
                    new_state = queens.movequeen(j, current_state)
                    
                    is_in_visited = any(np.array_equal(new_state, x) for x in visited)
                    if not is_in_visited:
                        if new_state is not None:
                            queue.append(new_state)
                            visited.append(new_state)


# class Node:
#     def __init__(self, row, col, g, h, parent):
#         self.row = row
#         self.col = col
#         self.g = g         # cost to come (previous g + moving cost)
#         self.h = h          # heuristic
#         self.cost = self.g + self.h      # total cost (depend on the algorithm)
#         self.parent = parent    # previous node             

class State:
    def __init__(self, state):
        self.state = state
        self.g = 0
        self.h = math.inf
        self.f = self.g + self.h
        self.previous = []

def g_cost(current_state, previous_state):
    g = 0
    queens = Queen()
    if previous_state == None:
        return 0
    else:
        queens.getpositions(current_state)
        cs = queens.positions
        queens.positions = []
        queens.getpositions(previous_state)
        ps = queens.positions
        for i in range(len(cs)):
            if cs[i] != ps[i]:
                g += math.sqrt((cs[i][0] - ps[i][0])**2 + (cs[i][1] - ps[i][1])**2)
    return g

def min_f_cost(List):
    Cost = math.inf
    index = 0
    for i in range(len(List)):
        if Cost > List[i].cost:
            Cost = List[i].cost
            index = i 
    q = List[index]
    return q

def Astar(init_board_state, board_size):
    queens = Queen()
    Open_List = []
    Closed_List = []
    Open_List.append(init_board_state)

    while(Open_List):
        current_state = min_f_cost(Open_List)
        Open_List.pop(Open_List.index(current_state))
        
        queens.getpositions(current_state)
        for i in range(len(queens.positions)):
            for queen in queens.positions:
                cost = current_state.g + 1
                for j in range(-board_size+1, board_size):
                    queens.setcoords(queen)
                    new_state = queens.movequeen(j, current_state)

                    if new_state in Open_List:
                        if new_state.g <= cost:
                            continue
                    if new_state in Closed_List:
                        if new_state.g <= cost:
                            continue
                    else:
                        Open_List.append(new_state)
                        new_state.h = Attacking_Queens.attackingpairs(new_state)
                    new_state.g = cost
                    new_state.cost = new_state.h + new_state.g

        Closed_List.append(current_state)
    return Closed_List

if __name__ == "__main__":

    board_size = 5
    weight_range = 8

    #Generate the initial random configuration of the board
    init_board_state = generate_configuration(board_size, weight_range)

    #plot(board)

    bfs(init_board_state, board_size)
    
#g(x) will the steps moved
