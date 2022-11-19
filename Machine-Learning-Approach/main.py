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
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from numpy import mean
from sklearn.metrics import r2_score
import pickle
import features

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

class Board_Features:
    def __init__(self, pattern, cost):
        self.features = features.Features(pattern)
        self.pattern = pattern
        self.cost = cost
        self.size = len(self.pattern)
        self.heaviest_queen = self.features.Heaviest_Queen()
        self.lightest_queen = self.features.Lightest_Queen()
        self.total_weight = self.features.Total_Weight()
        self.ratio_heavy_to_light = self.features.Ratio_Heavy_to_Light()
        self.mean_weight = self.features.Mean_weight()
        self.median_weight = self.features.Median_weight()
        self.attacking_pairs = self.features.Pairs_Attacking_Queens()
        self.heuristic_1 = self.features.heuristic_1()
        self.heuristic_2 = self.features.heuristic_2()

def get_data(path):
    csv_ = CSV_("Data/Data_5_new.txt")
    board_list,  cost_list = csv_.readCSV_board("Data/Data_5_new.txt")
    return board_list, cost_list

board_list, cost_list = get_data("Data/Data_5_new.txt")

def data_processing(board_list, cost_list):
    features = 5 
    X = np.empty(shape=(1, features))  # features matrix for ML model
    Y = np.reshape(cost_list, (len(cost_list), 1))  # target matrix for ML model
    sample_node_list = []  # nodes of training samples
    for i in range(0, len(board_list)):
        current = Board_Features(board_list[i], cost_list[i])
        sample_node_list.append(current)
        new_row = [current.heuristic_1, current.heuristic_2, current.attacking_pairs,current.total_weight, current.ratio_heavy_to_light]
        X = np.vstack([X, new_row])
    X = X[1:, :]
    Xnew = np.hstack([X, Y])
    data = pd.DataFrame(Xnew, columns=['h1', 'h2', 'h3', 'h4','h5', 'cost']) 
    X_inputs = data[['h1', 'h2', 'h3', 'h4','h5']] 
    Y_targets = data['cost']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_inputs, Y_targets, test_size=0.1)
    return Xtrain, Xtest, Ytrain, Ytest

Xtrain, Xtest, Ytrain, Ytest = data_processing(board_list,cost_list)

def rmse(targets, predictions):
        return np.sqrt(np.mean(np.square(targets - predictions)))

def performance_metrics(targets, predictions):
    absolute_error = metrics.mean_absolute_error(targets,predictions)
    mean_squared_error = metrics.mean_squared_error(targets,predictions)
    root_mean_squared_error = np.sqrt(metrics.mean_squared_error(targets, predictions))
    return absolute_error, mean_squared_error, root_mean_squared_error

def Training(X_train,Y_train):
    kfold = KFold(n_splits=10)
    models = []
    for train_index, val_index in kfold.split(X_train):
        Xtrain = X_train.iloc[train_index]
        Ytrain = Y_train.iloc[train_index]
        Xval = X_train.iloc[val_index]
        Yval = Y_train.iloc[val_index]
        model = LinearRegression()
        poly = PolynomialFeatures(degree = 1)
        polyfeatures = poly.fit_transform(Xtrain)
        polyfeatures_val = poly.fit_transform(Xval)
        model.fit(polyfeatures,Ytrain)
        training_rmse = rmse(model.predict(polyfeatures), Ytrain)
        validation_rmse = rmse(model.predict(polyfeatures_val),Yval)
        models.append(model)
        print('Train RMSE: {}, Validation RMSE: {}'.format(training_rmse, validation_rmse))
    return models, model

models, model = Training(Xtrain, Ytrain)

def predict_avg(models, inputs):
    return np.mean([model.predict(inputs) for model in models], axis=0)

poly2 = PolynomialFeatures(degree= 1)
polyfeat = poly2.fit_transform(Xtest)
preds = predict_avg(models, polyfeat)
loss = rmse(Ytest, preds)

print("Testing Error", loss)
print(" ")

accuracy = r2_score(Ytest, preds)

print("Testing Accuracy", accuracy)
print(" ")

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

print(model.coef_)
# print(poly2.get_feature_names(Xtrain.columns))