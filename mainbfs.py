from queue import Empty
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
from collections import deque
from Attacking_Queens import *

class Queen:
    def __init__(self):
        self.row = 0                    #row coordinate of queen
        self.col = 0                    #col coordinate of queen
        self.board_size = 0             #board size
        self.positions = []             #initial positions of queens in the board                    
        self.weights = []

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
                    self.weights.append((current_board[x][y]))

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

def Astar(init_board_state, board_size):
    flag = 0
    new_state_g=0
    queens = Queen()
    Open_List = []
    Open_List_States = []
    Closed_List = []
    Open_List.append((init_board_state,np.inf,0,np.inf))
    Open_List_States.append((init_board_state))

    while Open_List:
        Open_List.sort(key = takeSecond)
        m = Open_List.pop(0)
        current_state = m[0]
        f_cost = m[1]
        g_cost = m[2]
        h_cost = m[3]

        queens.getpositions(current_state)
        # print(h_cost)
        if h_cost == 0:
            flag = 1
            break

        for i in range(len(queens.positions)):
            for queen in queens.positions:
                cost = g_cost + queens.weights[queens.positions.index(queen)]**2
                for j in range(-board_size+1, board_size):
                    queens.setcoords(queen)
                    queen_weight = queens.weights[queens.positions.index(queen)]
                    new_state = queens.movequeen(j, current_state)
                    new_state_g += queen_weight**2

                    # new_state_h = attackingpairs(new_state)
                    # new_state_cost = new_state_g + new_state_h
                    # a = (new_state,new_state_cost,new_state_g,new_state_h)

                    # for i in range(len(Open_List)):
                    #     if new_state == Open_List[i][0]:
                    #         if new_state_g <= cost:
                    #             continue

                    # for j in range(len(Closed_List)):
                    #     if new_state == Closed_List[i][0]:
                    #         if new_state_g <= cost:
                    #             continue

                    if (new_state,new_state_cost,new_state_g,new_state_h) in Open_List: #This is wrong
                        if new_state_g <= cost:
                            continue
                    if (new_state,new_state_cost,new_state_g,new_state_h) in Closed_List:
                        if new_state_g <= cost:
                            continue
                    else:
                        new_state_h = attackingpairs(new_state)
                        new_state_cost = new_state_g + new_state_h
                        Open_List.append((new_state,new_state_cost,new_state_g,new_state_h))
                    new_state_g = cost
                    new_state_cost = new_state_h + new_state_g

        Closed_List.append(current_state)
    return current_state, queens.positions

if __name__ == "__main__":

    board_size = 5
    weight_range = 8

    # Generate the initial random configuration of the board
    init_board_state, init_pos = generate_configuration(board_size, weight_range)
    # print(init_board_state)
    # solution, queens_pos = bfs(init_board_state, board_size)
    solution, queens_pos = Astar(init_board_state,board_size )
    plt.figure(1)
    plot(init_board_state, init_pos, 'Initial Configuration')
    plt.figure(2)
    plot(solution, queens_pos, 'Solution')
    plt.show()


    
