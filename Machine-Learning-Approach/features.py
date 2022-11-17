
'''
All Possible Features:
1) Heaviest Queen
2) Lightest Queen
3) Total Weights
4) Ratio of Heaviest Queen to Lightest Queen
5) Mean Weight of the board
6) Median Weight
7) Heaviest Queen Attacks
8) Lightest Queen Attacks
9) Horizontal Attacks
10) Vertical Attacks
11) Diagonal Attacks
12) Pair of Attacking Queens 
13) Highest number of attacks by queen
'''

import numpy as np
import math
import pandas as pd

dataframe1 = pd.read_excel('Data.xlsx')
df = dataframe1[:1]
init_board = df.iloc[0][1]
print(init_board)
board = [[[0, 7, 0, 0, 0], [7, 0, 3, 0, 0], [0, 0, 0, 4, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 6]]]

class Board:
    def __init__(self,board):
        self.board = board[0]
        self.board_size = len(board[0])
        self.heaviest_queen = 0
        self.lightest_queen = 0
        self.total_weight = 0
        self.ratio_H_to_L = 0
        self.mean_weight = self.total_weight / self.board_size
        self.median_weight = 0
        self.Horizontal_Attacks = 0
        self.Vertical_Attacks = 0
        self.Diagonal_Attacks = 0
        self.pair_of_attacking_queens = 0
        self.highest_attacks_by_queen = 0
        self.queens = []

    def get_all_queens(self):
        for i in range(0, self.board_size):
            for j in range(0,self.board_size):
                if self.board[i][j] != 0:
                    self.queens.append(self.board[i][j])
        return self.queens

    def Heaviest_Queen(self):
        self.queens = Board.get_all_queens(self)
        self.heaviest_queen = max(self.queens)
        return self.heaviest_queen

    def Lightest_Queen(self):
        self.queens = Board.get_all_queens(self)
        self.Lightest_queen = min(self.queens)
        return self.Lightest_queen

    def Total_Weight(self):
        self.queens = Board.get_all_queens(self)
        self.total_weight = sum(self.queens)
        return self.total_weight

    def Ratio_Heavy_to_Light(self):
        self.queens = Board.get_all_queens(self)
        self.ratio = Board.Heaviest_Queen(self) / Board.Lightest_Queen(self)
        return self.ratio

    def Mean_weight(self):
        self.queens = Board.get_all_queens(self)
        self.mean_weight = np.mean(self.queens)
        return self.mean_weight

    def Median_weight(self):
        self.queens = Board.get_all_queens(self)
        self.median_weight = np.median(self.queens)
        return self.median_weight

    
board = Board(board)
a = board.Median_weight()
print(a)
