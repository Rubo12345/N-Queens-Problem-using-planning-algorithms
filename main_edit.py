# from multiprocessing.reduction import duplicate
from mimetypes import init
from re import I
from sqlite3 import Row
from tkinter.filedialog import Open
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
from collections import deque
import Attacking_Queens
import math

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
    dup_check = []
    board = np.zeros([n,n])
    for i in range(0, n):
        row_index = random.randrange(0,n,1)
        weight = random.randrange(1, weight_range, 1)
        if weight in dup_check:
            while(weight in dup_check):
                weight = random.randrange(1, weight_range, 1)
        board[row_index,i] = weight
        dup_check.append(board[row_index,i])
    # print(board)
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
                    # is_in_visited = True or False
                    if not is_in_visited:             # if is_in_visited = False
                        if new_state is not None:
                            queue.append(new_state)
                            visited.append(new_state)
    print(new_state)

# -----------------------------------------------------------------------------------------------

Board = [[0,0,0,7,0],
        [5,0,0,0,0],
        [0,0,1,0,0],
        [0,2,0,0,0],
        [0,0,0,0,8]]

Queens = [[2,2],[3,1],[0,1],[0,3],[4,4]] # 1,2,5,7,8
Weights = [1, 2, 5, 7, 8]

class Node:
    def __init__(self, row, col, g, h, parent):
        self.row = row
        self.col = col
        self.g = g         # cost to come (previous g + moving cost)
        self.h = h          # heuristic
        self.cost = self.g + self.h      # total cost (depend on the algorithm)
        self.parent = parent    # previous node             

def min_f_cost(List):
    Cost = math.inf
    index = 0
    for i in range(len(List)):
        if Cost > List[i].cost:
            Cost = List[i].cost
            index = i 
    q = List[index]
    return q

# Attacking_Queens.checkattackers(Board,q.row,q.col) * weight[i] = Heuristics

def Astar(Board):
    for i in range(len(Queens)):
        current_queen = Node(2,2,0,math.inf,None)
        Open_List = [];Closed_List = []
        Open_List.append(current_queen)

        if current_queen.h == 0:                              # Think on checking only the number of attacking pairs
            Board[current_queen[0],current_queen[1]] == Weights[i]
            Closed_List.append(current_queen)
            break

        while(Open_List):
            q = min_f_cost(Open_List) 
            Open_List.pop(Open_List.index(q))
            q.cost = q.g + Attacking_Queens.checkattackers(Board,q.row,q.col)
            Successors = [Node(q.row - 1,q.col,0,math.inf,q),Node(q.row+1,q.col,0,math.inf,q)]
            for successor in Successors:
                successor_current_cost = q.g + 1 # Add the weight 
                
                # if successor.h == 0:                              # Think on checking only the number of attacking pairs
                #     Board[successor[0],successor[1]] == Weights[i]
                #     Closed_List.append(successor)
                #     break

                # if successor.h != 0:
                #     successor.g = q.g + 1
                #     successor.h = Attacking_Queens.checkattackers(Board,successor.row,successor.col)
                #     successor.cost = successor.g + successor.h
                
                if successor in Open_List:
                    if successor.g <= successor_current_cost:
                        continue
                if successor in Closed_List:
                    if successor.g <= successor_current_cost:
                        continue
                else:
                    Open_List.append(successor)
                    successor.h = Attacking_Queens.checkattackers(Board,successor.row,successor.col) * Weights[i]
                successor.g = successor_current_cost
                successor.cost = successor.h + successor.g

                # If you don't get any position for successor in the column, just move on to the next iteration
        Closed_List.append(q)
    return Closed_List
    
Astar(Board)         
            





# if __name__ == "__main__":

#     board_size = 5
#     weight_range = 8

#     #Generate the initial random configuration of the board
#     init_board_state = generate_configuration(board_size, weight_range)

#     # plot(init_board_state)

#     bfs(init_board_state, board_size)
    
