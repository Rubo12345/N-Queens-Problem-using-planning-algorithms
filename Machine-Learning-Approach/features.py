
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
import Attacking_Queens

# dataframe1 = pd.read_excel('Data.xlsx')
# df = dataframe1[:1]
# init_board = df.iloc[0][1]
# print(init_board)
board = [[0, 7, 0, 0, 0], [7, 0, 3, 0, 0], [0, 0, 0, 4, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 6]]

class Features:
    def __init__(self,board):
        self.board = board
        self.board_size = len(board)
        self.heaviest_queen = 0
        self.lightest_queen = 0
        self.total_weight = 0
        self.ratio_H_to_L = 0
        self.mean_weight = self.total_weight / self.board_size
        self.median_weight = 0
        self.horizontal_attacks = 0
        self.vertical_attacks = 0
        self.diagonal_attacks = 0
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
        self.queens = Features.get_all_queens(self)
        self.heaviest_queen = max(self.queens)
        return self.heaviest_queen

    def Lightest_Queen(self):
        self.queens = Features.get_all_queens(self)
        self.Lightest_queen = min(self.queens)
        return self.Lightest_queen

    def Total_Weight(self):
        self.queens = Features.get_all_queens(self)
        self.total_weight = sum(self.queens)
        return self.total_weight

    def Ratio_Heavy_to_Light(self):
        self.queens = Features.get_all_queens(self)
        self.ratio = Features.Heaviest_Queen(self) / Features.Lightest_Queen(self)
        return self.ratio

    def Mean_weight(self):
        self.queens = Features.get_all_queens(self)
        self.mean_weight = np.mean(self.queens)
        return self.mean_weight

    def Median_weight(self):
        self.queens = Features.get_all_queens(self)
        self.median_weight = np.median(self.queens)
        return self.median_weight

    def Horizontal_Attacks(self):
        pass

    def Vertical_Attacks(self):
        pass

    def Diagonal_Attacks(self):
        pass

    def Pairs_Attacking_Queens(self):
        pairs = Attacking_Queens.attackingpairs(self.board)
        return pairs

    def heuristic_1(self):
        h1 = Attacking_Queens.attackingpairs(self.board) * Features.Total_Weight(self)
        return h1

    def heuristic_2(self):
        h2 = Attacking_Queens.attackingpairs(self.board) * Features.Mean_weight(self)
        return h2

    def heuristic_3(self):
        pass

    def heuristic_4(self):
        pass

board = Features(board)
a = board.Pairs_Attacking_Queens()
print(a)
