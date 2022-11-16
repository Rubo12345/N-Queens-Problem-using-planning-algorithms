from lib2to3.pgen2.token import OP
from queue import Empty
import matplotlib.pyplot as plt
import time as t
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
import copy

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

    def movequeen(self, iter, board):
        '''Move one queen along the column. 
        If the position is valid, create a board with the new configuration'''
        dir_row = [0, 1, 0, -1]
        dir_col = [1, 0, -1, 0]
        new_position = (self.row + dir_row[iter], self.col + dir_col[iter])
        if self.is_valid(new_position):
            if board[new_position] == 0:
                new_board = board.copy()
                new_board[new_position[0], new_position[1]] = board[self.row, self.col]
                new_board[self.row, self.col] = 0
                board = new_board
        
        return board

def plot(board, pos_queens, title):
    #Show chess board
    chessboard = np.array([[(i+j)%2 for i in range(len(board))] for j in range(len(board))])
    plt.imshow(chessboard,cmap='ocean')
    plt.title(title, fontweight="bold")
    for queen in pos_queens:
        plt.text(queen[1], queen[0], board[queen[0],queen[1]], fontsize=20, ha='center', va='center', color='black')

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

def attpairs(board,x,y):
  count=0
  attackers=[]
  for i in range(len(board)):
    for j in range(len(board[i])):
      if board[i][j]>0 and (i!=x or j!=y):
        xdiff=i-x
        ydiff=j-y
        #print(i,j)
        if xdiff==0 or ydiff==0:
          count+=1
          if xdiff==0:
            if ydiff>0:
              attackers.append([[x,y],[i,j]])
            else:
              attackers.append([[i,j],[x,y]])
          elif ydiff==0:
            if xdiff>0:
              attackers.append([[x,y],[i,j]])
            else:
              attackers.append([[i,j],[x,y]])
        elif (xdiff/ydiff)==1 or (xdiff/ydiff)==-1:
          count+=1
          if x<i:
            attackers.append([[x,y],[i,j]])
          else:
            attackers.append([[i,j],[x,y]])
  return count, attackers

def rboard(N):
    import random, copy
    board=[]
    queencoords=[]
    for i in range(N):
        temp=[]
        a=random.randint(0,N-1)
        for j in range(N):
            if j==a:
                r=random.randint(1,9)
                temp.append(r)
                queencoords.append([j,i,r])
            else:
                temp.append(0)
        board.append(temp)
    t=copy.deepcopy(board)
    for i in range(N):
        for j in range(N):
            board[i][j]=t[j][i]
    return board

def genboard(coords):
  brd=[]
  for i in range(len(coords)):
    temp=[]
    for j in range(len(coords)):
      temp.append(0)
    brd.append(temp)

  for i in coords:
    brd[i[0]][i[1]]=i[2]
  return brd

def gencoord(brd):
    coord=[]
    for i in range(len(brd)):
        for j in range(len(brd)):
            temp=[]
            if brd[i][j]>0:
                temp.append(i)
                temp.append(j)
                temp.append(brd[i][j])
                coord.append(temp)
    return coord

def print2D(m):
    for i in range(len(m)):
        for j in range(len(m[i])):
            print(m[i][j], end="\t")
        print()

def movesa(old,new,coord):
    coords=copy.deepcopy(coord)
    for i in coords:
        if i[0]==old[0] and i[1]==old[1]:
            i[0]=new[0]
            i[1]=new[1]
    return coords

def findmoves(r,c,brd):
    hmoves=[]
    vmoves=[]
    dmoves=[]
    coords=gencoord(brd)
    N=len(brd)
    for i in range(N):
        if brd[r][i]<=0:
            hmoves.append([r,i])
    for i in range(N):
        if brd[i][c]<=0:
            vmoves.append([i,c])
    iter1=r
    iter2=c
    while iter1>=0 and iter2>=0: #check for other queens northwest of the current queen
        if brd[iter1][iter2]==0:
            dmoves.append([iter1,iter2])
        iter1-=1
        iter2-=1
    iter1=r
    iter2=c
    while iter1<N and iter2<N: #check for other queens Southeast of the current queen
        if brd[iter1][iter2]==0:
            dmoves.append([iter1,iter2])
        iter1+=1
        iter2+=1
    iter1=r
    iter2=c
    while iter1>=0 and iter2<N: #check for other queens Northeast of the current queen
        if brd[iter1][iter2]==0:
            dmoves.append([iter1,iter2])
        iter1-=1
        iter2+=1
    iter1=r
    iter2=c
    while iter1<N and iter2>=0: #check for other queens Southwest of the current queen
        if brd[iter1][iter2]==0:
            dmoves.append([iter1,iter2])
        iter1+=1
        iter2-=1
    mvset=[]
    mvset=vmoves
    for i in hmoves:
        mvset.append(i)
    return mvset

def costfinder(brd):
    N=len(brd)
    cost=0
    costarray=[]
    coords=gencoord(brd)
    for i in range(N):
        for j in range(N):
            if brd[i][j]>0:
                mvset=findmoves(i,j,brd)
                for k in mvset:
                    if (abs(i-k[0]))!=0:
                        cost=(brd[i][j]**2)*(abs(i-k[0]))
                    elif (abs(j-k[1]))!=0:
                        cost=(brd[i][j]**2)*(abs(j-k[1]))
                    att= attackingpairs(genboard(movesa([i,j],k,coords)))
                    costarray.append([[i,j],k,[cost,att]])
    return costarray

def sa(brd,T,cr,k,lim):
    endflag=0
    N=len(brd)
    awt=2*25*N/((N-1)*1.1)
    pwt=1
    metric=0
    coord=gencoord(brd)
    start=t.time()
    ctime=start
    n=0
    tcost=0
    moves=[]
    while T>lim and endflag==0 and (ctime-start)<10:
        n+=1
        ctime=t.time()
        cmetric=awt*attackingpairs(brd)
        if cmetric==0:
            endflag=1
        costs=costfinder(brd)
        if endflag==0:
            for i in costs:
                metric=awt*i[2][1]+pwt*i[2][0]
                if metric<=cmetric:
                    coord=copy.deepcopy(movesa(i[0],i[1],coord))
                    brd=copy.deepcopy(genboard(coord))
                    if attackingpairs(brd)==0:
                            endflag=1
                    moves.append([i[0],i[1]])
                    tcost=tcost+i[2][0]
                    break
                else :
                    prob=2.7183**(-(metric-cmetric)/(k*T))
                    r=np.random.random()
                    if r<prob:
                        coord=copy.deepcopy(movesa(i[0],i[1],coord))
                        brd=copy.deepcopy(genboard(coord))
                        if attackingpairs(brd)==0:
                            endflag=1
                        moves.append([i[0],i[1]])
                        tcost=tcost+i[2][0]
                        break
                
        T=T*cr
    return brd, attackingpairs(brd), moves, tcost

def restarts(rb):
    i=0
    flag=0
    start=t.time()
    ctime=start
    min=100
    mboard=[]
    while (ctime-start)<30 and flag==0:
        res=sa(rb,350,0.96,1,10)
        a=attackingpairs(res[0])
        if a<min:
            min=a
            mboard=res[0]
            matt=res[1]
            moves=res[2]
            tcost=res[3]
        if a==0:
            flag=1
        ctime=t.time()
        i+=1
    return mboard, matt, i, moves, tcost

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
                        att_queens = heuristicapno10(new_state)
                        queue.append((new_state, att_queens))
                        visited.append(new_state)
    
    if flag == 1:
        end = time.time()
        print("Elapsed time: " + str(round(end-start, 2)) + " s")
        print("Nodes expanded: " + str(len(expanded_nodes)))
        print("Search depth: " + str(search_depth))
        branching_factor = len(expanded_nodes)**(1/search_depth)
        print("The branching factor is " + str(round(branching_factor, 2)))

        return current_state, queens.positions
    else:
        print("No solution was found.")

def attacking_pairs(board): #Returns the exact set of queens that are attacking each other
  count=0
  l=len(board)
  att_set=[]
  for i in range(l):
    for j in range(l):
      if board[i][j]>0:
        temp=attpairs(board,i,j)
        count=count+temp[0] #count+checkattackers(board,i,j)
        for k in temp[1]:
          if k not in att_set:
            att_set.append(k)
  count=count/2
  return att_set

def heuristicap(new_state):
    return attackingpairs(new_state) * 10

def heuristicapno10(new_state):
    return attackingpairs(new_state) * 10

def heuristicweightedap(new_state):
    sumsquare=0
    attset=attacking_pairs(new_state)
    for i in attset:
        for j in i:
            sumsquare+=new_state[j[0],j[1]]**2
    return len(attset)*sumsquare

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
    init_board_state_h = heuristicapno10(init_board_state)
    
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

                        new_state_h = heuristicapno10(new_state)
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

        branching_factor = len(expanded_nodes)**(1/search_depth)
        print("The branching factor is " + str(round(branching_factor, 2)))

        return current_state, queens.positions
    else:
        print("No solution was found.")

def hill_climbing(init_board_state):
    print(" ")
    print("Running Hill Climbing...")

    start = time.time()

    current_state,No_of_Attacking_Pair, iteration, moves, solution_cost = restarts(init_board_state)
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

    #Read board from csv file
    r = 0
    reader = csv.reader(open('board.csv', encoding='utf-8-sig'))
    init_board_state = list(reader)
    #init_board_state = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 3, 0, 0, 2], [7, 7, 2, 0, 0, 0, 9], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 0, 0]])

    for row in init_board_state:
        for i in range(len(row)):
            if row[i] == "":
                row[i] = 0
            else:
                row[i] = int(row[i])
                init_pos.append((r,i))
        r = r+1
    
    init_board_state = np.array(init_board_state)
    #init_board_state = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 3, 0, 0, 0], [7, 7, 2, 0, 0, 2, 9], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 0, 0]])
    board_size = len(init_board_state)
    #Uncomment to generate a random board configuration
    '''board_size = 5
    weight_range = 8
    init_board_state, init_pos = generate_configuration(board_size, weight_range)'''
     
    solution_bfs, queens_pos_bfs = bfs(init_board_state, board_size)
    plt.figure(1)
    plot(init_board_state, init_pos, 'Initial Configuration')
    plt.figure(2)
    plot(solution_bfs, queens_pos_bfs, 'Solution Greedy Search')
    solution_Astar, queens_pos_Astar = Astar(init_board_state, board_size)
    plt.figure(3)
    plot(solution_Astar, queens_pos_Astar, 'Solution A*')
    solution_hc, queens_pos_hc = hill_climbing(init_board_state)
    plt.figure(4)
    plot(solution_hc, queens_pos_hc, "Solution Hill Climbing with Simulated Annealing")
    plt.show()

    '''print2D(init_board_state)
    print()
    print2D(solution_bfs)
    print()
    print2D(solution_Astar)
    print()
    print2D(solution_hc)
    print()
    print(init_board_state.tolist())
    print()
    print(solution_Astar.tolist())'''
