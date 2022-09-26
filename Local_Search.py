import copy
import numpy as np
from  Attacking_Queens import attackingpairs
import time as t

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

