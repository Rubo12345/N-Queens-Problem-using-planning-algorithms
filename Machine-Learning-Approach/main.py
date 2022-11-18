from statistics import mean
import numpy as np
from math import factorial
import csv
import copy
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from numpy import mean
from sklearn.metrics import r2_score
import pickle

class CSV_:
    def __init__(self, path):
        self.CSVfilePath = path
    def readCSV_board(self, file_address=''):
        if len(file_address) == 0:
            file_address = self.CSVfilePath
        board_list = []
        cost_list = []
        with open(file_address, mode='r')as file:
            csvFile = csv.reader(file)
            board_size = int(next(csvFile)[0])  # first line is the length of board
            index = 0
            for lines in csvFile:
                # only when there are blank lines
                if index == 0:
                    board = []  # new board will start after cost
                    # continue
                if index < board_size:
                    board.append(list(lines))
                    index += 1
                    continue
                if index == board_size:
                    for i in range(0, len(board)):
                        for j in range(0, len(board)):
                            board[j][i] = int(board[j][i])
                    board_list.append(board)
                    cost_list.append(int(lines[0]))
                    index = 0
                    # print(board)
                    continue
        file.close()
        return board_list, cost_list

def get_data(path):
    csv_ = CSV_("Data/Data_5_new.txt")
    board_list,  cost_list = csv_.readCSV_board("Data/Data_5_new.txt")
    update_list, update_cost = csv_.readCSV_board("Data/Data_5_new.txt")
    board_list += update_list
    cost_list += update_cost
    return board_list, cost_list

board_list, cost_list = get_data("Data/Data_5_new.txt")
n = len(board_list[0])

def attacking_queens(grid):
    totalhcost = 0
    totaldcost = 0
    for i in range(0, n):
        for j in range(0, n):
            # if this node is a queen, calculate all violations
            if grid[i][j] != 0:
                # subtract 2 so don't count self
                # sideways and vertical
                totalhcost -= 2
                for k in range(0, n):
                    if grid[i][k] != 0:
                        totalhcost += 1
                    if grid[k][j] != 0:
                        totalhcost += 1
                # calculate diagonal violations
                k, l = i + 1, j + 1
                while k < n and l < n:
                    if grid[k][l] != 0:
                        totaldcost += 1
                    k += 1
                    l += 1
                k, l = i + 1, j - 1
                while k < n and l >= 0:
                    if grid[k][l] != 0:
                        totaldcost += 1
                    k += 1
                    l -= 1
                k, l = i - 1, j + 1
                while k >= 0 and l < n:
                    if grid[k][l] != 0:
                        totaldcost += 1
                    k -= 1
                    l += 1
                k, l = i - 1, j - 1
                while k >= 0 and l >= 0:
                    if grid[k][l] != 0:
                        totaldcost += 1
                    k -= 1
                    l -= 1
    return ((totaldcost + totalhcost) / 2)

def getQueenPos(board):
    queen_pos = []
    queen_weight = []
    for j in range(len(board[0])):
        for i in range(len(board)):
            if int(board[i][j]) != 0:
#                 print(board[i][j])
                queen_pos.append([i, j])
                queen_weight.append(board[i][j])

    return queen_pos, queen_weight

def is_attacking(board, pos):
    attacks = 0
    attackers = []
#     print("pos",pos)

    if len(board) == 0:
        return None

    for i in range(len(board[0])):
#         print("index i" ,i)
        if int(board[pos[0]][i]) != 0 and i != pos[1]:
#             print(pos[0], i)
            attackers.append([pos[0], i])
            attacks += 1

    i = pos[0] + 1
    j = pos[1] + 1
    while i < len(board) and j < len(board[0]):
        if int(board[i][j]) != 0:
#             print(j, i)
            attackers.append([i, j])
            attacks += 1
        i += 1
        j += 1

    i = pos[0] - 1
    j = pos[1] - 1
    while i >= 0 and j >= 0:
        if int(board[i][j]) != 0:
#             print(i, j)
            attackers.append([i, j])
            attacks += 1
        i -= 1
        j -= 1

    i = pos[0] - 1
    j = pos[1] + 1
    while i >= 0 and j < len(board[0]):
        if int(board[i][j]) != 0:
#             print(i, j)
            attackers.append([i, j])
            attacks += 1
        i -= 1
        j += 1

    i = pos[0] + 1
    j = pos[1] - 1
    while i > len(board) and j >= 0:
        if int(board[i][j]) != 0:
#             print(i, j)
            attackers.append([i, j])
            attacks += 1
        i += 1
        j -= 1

#     print("done")
#     print(attacks)
#     print("attackers", attackers)

    return attacks, attackers

def total_attacks(board, queen_pos):
    board_copy = copy.deepcopy(board)
    attacks = 0
    attackers = []
    iter = 0
    for queen in queen_pos:
        new_attacks, new_attackers = is_attacking(board_copy, [queen[0], queen[1]-iter])
        attacks += new_attacks
        attackers = attackers + new_attackers
        [j.pop(0) for j in board_copy]
        iter += 1

def heuristic1(grid):
    queenPos, queenWeight = getQueenPos(grid)
    he = 0
    board_copy = copy.deepcopy(grid)
    #     print(queenPos)
    for i in range(len(queenPos)):
        attacks, attacker = is_attacking(grid, queenPos[i])
        he += queenWeight[i]  **2* attacks
        [j.pop(0) for j in board_copy]
    #         print(he)
    return he

def heuristic2(grid):
    queenPos, queenWeight = getQueenPos(grid)
    he = []
#     print(queenPos)
    for i in range(len(queenPos)):
        attacks, attacker = is_attacking(grid, queenPos[i])
        he.append(queenWeight[i]**2*attacks)
#         print(he)
    return he

def queen_positions(grid):
    queen_pos = []
    for i in range(0, len(grid)):
        for j in range(0, len(grid)):
            if grid[j][i] != 0:
                queen_pos.append(j)
                continue
    return queen_pos

def pos_difference(grid):
    queen_pos = queen_positions(grid)
    pos_dif = []
    i = 1
    while i < len(queen_pos):
        a = abs(queen_pos[i] - queen_pos[i - 1])
        pos_dif.append(a)
        i = i + 1

    return pos_dif

def queen_weights(grid):
    queen_weight = []
    for i in range(0, len(grid)):
        for j in range(0, len(grid)):
            if grid[j][i] != 0:
                queen_weight.append(grid[j][i])
                continue
    return queen_weight

def heaviest_Q(grid):
    Qboard = queen_weights(grid)
    HeavyQ = max(Qboard)
    return HeavyQ

def lightest_Q(grid):
    Qboard = queen_weights(grid)
    HeavyQ = min(Qboard)
    return HeavyQ

def avg_weight(grid):
    Qboard = queen_weights(grid)
    avg = mean(Qboard)
    return avg

class training_sample:
    # define attributes of the training samples we need i.e. initial pattern, solved pattern and solved pattern cost by astar search algorithm.
    def __init__(self, pattern, cost):
        self.pattern = pattern
        self.cost = cost
        self.size = len(self.pattern)
        qPos, Qweight = getQueenPos(self.pattern)
        self.attacking_pairs = attacking_queens(self.pattern) # feature 1
        self.total_weight = sum(Qweight)
        self.queen_weight = queen_weights(self.pattern)  # feature 2
        self.average_weight = avg_weight(self.pattern) # feature 3
        self.average_move = len(self.pattern) / 2
        self.h = heuristic1(self.pattern)

def data_processing(board_list, cost_list):
    features = 4  # no. of features
    X = np.empty(shape=(1, features))  # features matrix for ML model
    Y = np.reshape(cost_list, (len(cost_list), 1))  # target matrix for ML model
    sample_node_list = []  # nodes of training samples
    for i in range(0, len(board_list)):
        current = training_sample(board_list[i], cost_list[i])
        sample_node_list.append(current)
        new_row = [current.h, current.attacking_pairs, current.attacking_pairs*current.total_weight, current.attacking_pairs*current.average_weight]
        X = np.vstack([X, new_row])
    X = X[1:, :]
    Xnew = np.hstack([X, Y])
    print(len(Xnew))
    data = pd.DataFrame(Xnew, columns=['h1', 'h2', 'h3', 'h4', 'cost']) #, 'h2', 'h3', 'h4', 'h5', 'h6', 'cost'])
    X_inputs = data[['h1', 'h2', 'h3', 'h4']] #, 'h2', 'h3', 'h4', 'h5', 'h6']]
    Y_targets = data['cost']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_inputs, Y_targets, test_size=0.1)
    return Xtrain, Xtest, Ytrain, Ytest

Xtrain, Xtest, Ytrain, Ytest = data_processing(board_list,cost_list)
print(Xtrain)
print(" ")
print(Ytrain)

# def rmse(targets, predictions):
#     return np.sqrt(np.mean(np.square(targets - predictions)))

# def train_and_evaluate(X_train, train_targets, X_val, val_targets):
#     model = LinearRegression()
#     poly = sklearn.preprocessing.PolynomialFeatures(degree=1)
#     polyfeat = poly.fit_transform(X_train)
#     polyfeat_val = poly.fit_transform(X_val)
#     model.fit(polyfeat, train_targets)
#     train_rmse = rmse(model.predict(polyfeat), train_targets)
#     val_rmse = rmse(model.predict(polyfeat_val), val_targets)
#     return model, train_rmse, val_rmse

# kfold = KFold(n_splits=10)
# models = []

# for train_idxs, val_idxs in kfold.split(Xtrain):
#     X_train = Xtrain.iloc[train_idxs]
#     train_targets = Ytrain.iloc[train_idxs]
#     X_val, val_targets = Xtrain.iloc[val_idxs], Ytrain.iloc[val_idxs]
#     model, train_rmse, val_rmse = train_and_evaluate(X_train,
#                                                      train_targets,
#                                                      X_val,
#                                                      val_targets,
#                                                      )
#     models.append(model)
#     print('Train RMSE: {}, Validation RMSE: {}'.format(train_rmse, val_rmse))

def rmse(targets, predictions):
        return np.sqrt(np.mean(np.square(targets - predictions)))

def Training(X_train,Y_train):
    for train_index, val_index in kfold.split(X_train):
        Xtrain = X_train.iloc[train_index]
        Ytrain = Y_train.iloc[train_index]
        Xval = X_train.iloc[val_index]
        Yval = Y_train.iloc[val_index]
        
        model = LinearRegression()
        poly = sklearn.preprocessing.PolynomialFeatures(degree = 1)
        polyfeatures = poly.fit_transform(Xtrain)
        polyfeatures_val = poly.fit_transform(Xval)
        model.fit(polyfeatures,Ytrain)
        training_rmse = rmse(model.predict(polyfeatures), Ytrain)
        validation_rmse = rmse(model.predict(polyfeatures_val),Yval)
        models.append(model)

        print('Train RMSE: {}, Validation RMSE: {}'.format(training_rmse, validation_rmse))
    return models, model

kfold = KFold(n_splits=10)
models = []
models, model = Training(Xtrain, Ytrain)

def predict_avg(models, inputs):
    return np.mean([model.predict(inputs) for model in models], axis=0)

poly2 = sklearn.preprocessing.PolynomialFeatures(degree= 1)
polyfeat = poly2.fit_transform(Xtest)
preds = predict_avg(models, polyfeat)
loss = rmse(Ytest, preds)

print("Error", loss)
print(" ")

accuracy = r2_score(Ytest, preds)

print("Accuracy", accuracy)
print(" ")

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

print(model.coef_)
print(poly2.get_feature_names(Xtrain.columns))