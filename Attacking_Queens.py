def checkattackers(board,x,y):
  count=0
  iter=y
  flag=0
  while flag==0 and iter>=0: #check for other queens on the left of the current queen
    if board[x][iter]>0 and iter!=y:
      flag=1
    iter-=1
  count=count+flag
  #print(count)
  flag=0
  iter=y
  while flag==0 and iter<len(board):#check for other queens on the right of the current queen
    if board[x][iter]>0 and iter!=y:
      flag=1
    iter+=1
  count=count+flag
  #print(count)
  flag=0
  iter=x
  while flag==0 and iter>=0:#check for other queens above the current queen
    if board[iter][y]>0 and iter!=x:
      flag=1
    iter-=1
  count=count+flag
  #print(count)
  flag=0
  iter=x
  while flag==0 and iter<len(board[0]):#check for other queens below the current queen
    if board[iter][y]>0 and iter!=x:
      flag=1
    iter+=1
  count=count+flag
  #print(count)
  flag=0 
  #diagonals:
  iter1=x
  iter2=y
  while flag==0 and iter1>=0 and iter2>=0: #check for other queens northwest of the current queen
    if board[iter1][iter2]>0 and iter1!=x and iter2!=y:
      flag=1
    iter1-=1
    iter2-=1
  count=count+flag
  #print(count)
  flag=0 
  iter1=x
  iter2=y
  while flag==0 and iter2<len(board) and iter1>=0: #check for other queens northeast of the current queen
    if board[iter1][iter2]>0 and iter1!=x and iter2!=y:
      flag=1
    iter1-=1
    iter2+=1
  count=count+flag
  # if flag==1: print('northeast')
  # print(count)
  flag=0 
  iter1=x
  iter2=y
  while flag==0 and iter2<len(board) and iter1<len(board): #check for other queens southeast of the current queen
    if board[iter1][iter2]>0 and iter1!=x and iter2!=y:
      flag=1
    iter1+=1
    iter2+=1
    #print(iter1,iter2)
  count=count+flag
  #print(count)
  flag=0 
  iter1=x
  iter2=y
  while flag==0 and iter1<len(board) and iter2>=0: #check for other queens southwest of the current queen
    if board[iter1][iter2]>0 and iter1!=x and iter2!=y:
      flag=1
    iter1+=1
    iter2-=1
  count=count+flag
  #print(count)
  flag=0
  iter1=x
  iter2=y
  return count

# print(checkattackers([[1,1,1],[1,1,1],[1,1,1]],1,0))

def attackingpairs(board):
  count=0
  l=len(board)
  for i in range(l):
    for j in range(l):
      if board[i][j]>0:
        count=count+checkattackers(board,i,j)
  count=count/2
  return count
# print(attackingpairs([[0,1,0],[1,1,1],[0,1,0]]))