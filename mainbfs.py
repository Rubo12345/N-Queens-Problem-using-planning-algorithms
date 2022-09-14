from queue import Empty
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
from collections import deque
from attacking_queens import *

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
        self.positions = []
        self.board_size = len(current_board)
        for x in range(self.board_size):
            for y in range(self.board_size):
                if current_board[x][y] != 0:
                    self.positions.append((x,y))

    def is_valid(self, position):
        '''Check if the new position is within the chessboard'''

        if position[0] in range(0, self.board_size) and position[1] in range(0, self.board_size):
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

def plot(board, pos_queens, title):

    #Show chess board
    chessboard = np.array([[(i+j)%2 for i in range(len(board))] for j in range(len(board))])
    plt.imshow(chessboard,cmap='ocean')
    plt.title(title, fontweight="bold")
    for queen in pos_queens:
        plt.text(queen[0], queen[1], '♕', fontsize=20, ha='center', va='center', color='black')

def generate_configuration(n, weight_range):
    
    #Generate a nxn board with a random queen with random weight in each column.
    #0 is an empty space
    board = np.zeros([n,n])
    init_pos = []
    for i in range(0, n):
        row_index = random.randrange(0, n, 1)
        board[row_index, i] = random.randrange(1, weight_range, 1)
        init_pos.append((i, row_index))
    
    return board, init_pos

def takeSecond(elem):
    return elem[1]

def bfs(init_board_state, board_size):

    flag = 0
    queens = Queen()
    #Initialize queue and visited board configuration
    visited = []
    queue = []

    #append initial configuration
    visited.append(init_board_state)
    queue.append((init_board_state, np.inf))
    
    while queue:
        queue.sort(key=takeSecond)
        #pop the first element from the queue and mark it as visited
        m = queue.pop(0)
        current_state = m[0]
        n = m[1]

        #get positions of queens in current board
        queens.getpositions(current_state)

        if n == 0:
            flag = 1
            break
    
        for i in range(len(queens.positions)):
            for queen in queens.positions:
                for j in range(-board_size+1, board_size):
                    queens.setcoords(queen)
                    new_state = queens.movequeen(j, current_state)
                    
                    is_in_visited = any(np.array_equal(new_state, x) for x in visited)
                    if not is_in_visited:
                        if new_state is not None:
                            att_queens = attackingpairs(new_state)
                            queue.append((new_state, att_queens))
                            visited.append(new_state)

    if flag == 1:
        return current_state, queens.positions
    else:
        print("No solution was found.")

if __name__ == "__main__":

    board_size = 5
    weight_range = 8

    #Generate the initial random configuration of the board
    init_board_state, init_pos = generate_configuration(board_size, weight_range)

    solution, queens_pos = bfs(init_board_state, board_size)
    plt.figure(1)
    plot(init_board_state, init_pos, 'Initial Configuration')
    plt.figure(2)
    plot(solution, queens_pos, 'Solution')
    plt.show()


    
