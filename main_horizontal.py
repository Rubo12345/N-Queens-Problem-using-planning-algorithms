from lib2to3.pgen2.token import OP
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
import csv
import pandas as pd
from itertools import chain
import time
import Local_Search

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
        dir_row = [0, 1, 0, -1]
        dir_col = [1, 0, -1, 0]
        new_position = (self.row + dir_row[i], self.col + dir_col[i])
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
        plt.text(queen[1], queen[0], '♕', fontsize=20, ha='center', va='center', color='black')

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

def takeThird(elem):
    return elem[2]

def finalmoves(init_board, solution):
    column = 0
    costs = []
    for row_int, row_sol in zip(init_board.T, solution.T):
        column = column + 1
        for i in range(len(init_board)):
            for j in range(len(init_board)):
                if row_int[i] == row_sol[j] and row_int[i] != 0:
                    move = i-j
                    if move < 0:
                        if move == 1:
                            costs.append(row_int[i]**2)
                            print("Move column " + str(column) + " down " + str(abs(move)) + " square.")
                            
                        else:
                            costs.append((row_int[i]**2)*abs(move))
                            print("Move column " + str(column) + " down " + str(abs(move)) + " squares.")

                    elif move > 0:
                        if move == 1:
                            costs.append(row_int[i] ** 2)
                            print("Move column " + str(column) + " up " + str(abs(move)) + " square.")
                            
                        else:
                            # print("Move column " + str(column) + " up " + str(abs(move)) + " squares.")
                            costs.append((row_int[i]**2) * abs(move))
                            print("Move column " + str(column) + " up " + str(abs(move)) + " squares.")
    totalcost = sum(costs)
    print("The solution cost is: " + str(totalcost))

def bfs(init_board_state, board_size):
    print(" ")
    print("Running BFS...")

    start = time.time()

    flag = 0
    queens = Queen()
    #Initialize queue and visited board configuration
    visited = []
    queue = []
    search_depth = 0
    expanded_nodes = []
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

        search_depth += 1
        if n == 0:
            flag = 1
            break

        for queen in queens.positions:
            # for j in range(-board_size, board_size+1):
            for j in range(4):
                queens.setcoords(queen)
                new_state = queens.movequeen(j, current_state)

                expanded_nodes.append(new_state)
                is_in_visited = any(np.array_equal(new_state, x) for x in visited)
                if not is_in_visited:
                    if new_state is not None:
                        att_queens = attackingpairs(new_state)
                        queue.append((new_state, att_queens))
                        visited.append(new_state)
    
    if flag == 1:
        end = time.time()
        print("Elapsed time: " + str(round(end-start, 2)) + " s")
        print("Nodes expanded: " + str(len(expanded_nodes)))
        print("Search depth: " + str(search_depth))
        finalmoves(init_board_state, current_state)
        
        branching_factor = len(visited)**(1/search_depth)
        print("The branching factor is " + str(round(branching_factor, 2)))

        return current_state, queens.positions
    else:
        print("No solution was found.")

def Astar(init_board_state, board_size):
    print(" ")
    print("Running A*...")
    
    start = time.time()
    
    flag = 0
    queens = Queen()
    Open_List = []
    Closed_List = []
    expanded_nodes = []
    visited = []
    search_depth = 0
    init_board_state_h = attackingpairs(init_board_state) * 10
    
    #Store board configuration, total cost travelled so far, f (cost+move+heuristic), heuristic
    Open_List.append((init_board_state, 0, init_board_state_h, init_board_state_h))
    
    while Open_List:
        Open_List.sort(key = takeThird)
        m = Open_List.pop(0)
        current_state = m[0]
        g_cost = m[1] #total cost ravelled so far
        f_cost = m[2] #cost+move+heuristic
        h_cost = m[3] #heuristic
        
        Closed_List.append(current_state)
        search_depth += 1
        queens.getpositions(current_state)
        
        if h_cost == 0:
            flag = 1
            break

        for queen in queens.positions:
            for j in range(4):
                queens.setcoords(queen)
                queen_weight = queens.weights[queens.positions.index(queen)]
                new_state = queens.movequeen(j, current_state)

                expanded_nodes.append(new_state)

                is_in_Closed_List = any(np.array_equal(new_state, x) for x in Closed_List)
                
                is_in_Open_List = False
                for node in range(len(Open_List)):
                    if (new_state == Open_List[node][0]).all():
                        is_in_Open_List = True
                        index = node
                        current_cost = Open_List[node][2]

                if new_state is not None:
                    if not is_in_Closed_List:
                        visited.append(new_state)

                        new_state_h = attackingpairs(new_state) * 10
                        new_state_g = g_cost + queen_weight**2
                        new_state_cost = new_state_g + new_state_h

                        if is_in_Open_List:
                            if new_state_cost < current_cost:
                                Open_List.pop(index)
                            else:
                                continue

                        Open_List.append((new_state,new_state_g, new_state_cost, new_state_h))

    if flag == 1:
        end = time.time()
        print("Elapsed time: " + str(round(end-start, 2)) + " s")
        print("Nodes expanded: " + str(len(expanded_nodes)))
        print("Search depth: " + str(search_depth))
        finalmoves(init_board_state, current_state)

        branching_factor = len(visited)**(1/search_depth)
        print("The branching factor is " + str(round(branching_factor, 2)))

        return current_state, queens.positions
    else:
        print("No solution was found.")

def hill_climbing(init_board_state):
    print(" ")
    print("Running Hill Climbing...")

    start = time.time()

    current_state,No_of_Attacking_Pair, iteration, moves, solution_cost = Local_Search.restarts(init_board_state)
    queens_pos_hc = []
    board = np.zeros([len(current_state),len(current_state)])

    for i in range(len(current_state)):
        for j in range(len(current_state[i])):
            if current_state[i][j] != 0:
                queens_pos_hc.append([i,j])
            board[i,j] = current_state[i][j]
    current_state = board

    end = time.time()

    print("Elapsed time: " + str(round(end-start, 2)) + " s")
    print("Nodes expanded: " + str(len(moves)))
    
    return current_state, queens_pos_hc

if __name__ == "__main__":
    init_pos = []
    r = 0
    reader = csv.reader(open('board.csv', encoding='utf-8-sig'))
    init_board_state = list(reader)
    for row in init_board_state:
        for i in range(len(row)):
            if row[i] == "":
                row[i] = 0
            else:
                row[i] = int(row[i])
                init_pos.append((r,i))
        r = r+1
    
    init_board_state = np.array(init_board_state)
    board_size = len(init_board_state)

    solution_bfs, queens_pos_bfs = bfs(init_board_state, board_size)
    plt.figure(1)
    plot(init_board_state, init_pos, 'Initial Configuration')
    plt.figure(2)
    plot(solution_bfs, queens_pos_bfs, 'Solution BFS')
    solution_Astar, queens_pos_Astar = Astar(init_board_state, board_size)
    plt.figure(3)
    plot(solution_Astar, queens_pos_Astar, 'Solution A*')
    solution_hc, queens_pos_hc = hill_climbing(init_board_state)
    plt.figure(4)
    plot(solution_hc, queens_pos_hc, "Solution Hill Climbing with Simulated Annealing")
    plt.show()

