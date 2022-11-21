
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
        return attacking_pairs(self.board)[1]
        #pass

    def Vertical_Attacks(self):
        return attacking_pairs(self.board)[3]

    def Diagonal_Attacks(self):
        return attacking_pairs(self.board)[2]

    def avang(self): #average value of the angle of the line joining two adjacent queens and the horizontal
        sum=0
        coord=gencoord(self.board)
        q=len(coord)
        for i in range(q-1):
            sum += np.arctan2((coord[i][0]-coord[i+1][0]),(coord[i][1]-coord[i+1][1]))        
        return sum

    def avdist(self):
        sum=0
        coord = gencoord(self.board)
        q = len(coord)
        for i in range(q-1):
            for j in range(i+1,q):
                sum += ((coord[i][0]-coord[j][0])**2 + (coord[i][1]-coord[j][1])**2)**0.5
        return sum
    
    def avweightedcoord(self):
        sum = 0
        coord = gencoord(self.board)
        q = len(coord)
        for i in range(q):
            sum+=coord[i][0]*coord[i][2]
        return sum
        
    def avcoord(self):
        sum = 0
        coord = gencoord(self.board)
        q = len(coord)
        for i in range(q):
            sum += coord[i][0]
        return sum

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
        return Features.avdist(self)

    def heuristic_4(self):
        return Features.avweightedcoord(self)

    def heuristic_5(self):
        return Features.Pairs_Attacking_Queens(self)

    def heuristic_6(self):
        sumsquare=0
        attset = attacking_pairs(self.board)[0]
        for i in attset:
            for j in i:
                sumsquare+=self.board[j[0]][j[1]]**2
        return len(attset)*sumsquare*10

def gencoord(brd):
    coord = []
    for i in range(len(brd)):
        for j in range(len(brd)):
            temp = []
            if brd[i][j] > 0:
                temp.append(i)
                temp.append(j)
                temp.append(brd[i][j])
                coord.append(temp)
    return coord

def attpairs(board, x, y): #x is row number and y is column
    count = 0
    dcount=0
    vcount=0
    hcount=0
    attackers = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] > 0 and (i != x or j != y):
                xdiff = i-x
                ydiff = j-y
                # print(i,j)
                if xdiff == 0 or ydiff == 0:
                    count += 1
                    if xdiff == 0:
                        hcount+=1
                        if ydiff > 0:
                            attackers.append([[x, y], [i, j]])
                        else:
                            attackers.append([[i, j], [x, y]])
                    elif ydiff == 0:
                        vcount+=1
                        if xdiff > 0:
                            attackers.append([[x, y], [i, j]])
                        else:
                            attackers.append([[i, j], [x, y]])
                elif (xdiff/ydiff) == 1 or (xdiff/ydiff) == -1:
                    count += 1
                    dcount+=1
                    if x < i:
                        attackers.append([[x, y], [i, j]])
                    else:
                        attackers.append([[i, j], [x, y]])
    return count, attackers, hcount, dcount, vcount

def attacking_pairs(board):  # Returns the exact set of queens that are attacking each other and the number of hor, ver and diag attacking pairs 
    count = 0
    hcount=0
    vcount=0
    dcount=0
    l = len(board)
    att_set = []
    for i in range(l):
        for j in range(l):
            if board[i][j] > 0:
                temp = attpairs(board, i, j)
                count = count+temp[0]  # count+checkattackers(board,i,j)
                hcount+=temp[2]
                dcount+=temp[3]
                vcount+=temp[4]
                for k in temp[1]:
                    if k not in att_set:
                        att_set.append(k)
    count = count/2
    return att_set, hcount, dcount, vcount

board = Features(board)
a = board.Pairs_Attacking_Queens()
print(a)