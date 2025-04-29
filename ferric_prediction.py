# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 07:54:05 2022

@author: Urmi Ghosh, modified after Sany He
"""

# import xlrd
# import xlwt

import numpy as np
import pandas as pd
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import random

np.random.seed(22)
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import sklearn.pipeline as pl
import sklearn.linear_model as lm
import sklearn.preprocessing as sp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold


# Train the Model
def model_train(Data_Train_X, Data_Train_Y, Data_Test_X, Data_Test_Y, Regress_Mode, data_apply_global,
                If_Predict_GlobalData):
    # Polynomial_Regression
    if Regress_Mode == 0:
        polynomial_features = PolynomialFeatures(degree=2)
        Data_Train_X = polynomial_features.fit_transform(Data_Train_X)

        polynomial_features = PolynomialFeatures(degree=2)
        Data_Test_X = polynomial_features.fit_transform(Data_Test_X)

        model = LinearRegression()

    # Extra Tree
    elif Regress_Mode == 1:
        model = ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse',
                                    max_depth=20, max_features='auto', max_leaf_nodes=None,
                                    max_samples=None, min_impurity_decrease=0.0,
                                    min_samples_leaf=1,
                                    min_samples_split=2, min_weight_fraction_leaf=0.0,
                                    n_estimators=15, n_jobs=None, oob_score=False,
                                    random_state=42, verbose=0, warm_start=False)
    elif Regress_Mode == 2:
        # Random Forest
        model = RandomForestRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse',
                                      max_depth=100, max_features='sqrt', max_leaf_nodes=80,
                                      max_samples=None, min_impurity_decrease=0.0,
                                      min_samples_leaf=1,
                                      min_samples_split=5, min_weight_fraction_leaf=0.0,
                                      n_estimators=800, n_jobs=-1, oob_score=False,
                                      random_state=20, verbose=0, warm_start=False)
        # elif Regress_Mode == 2 :
        # #Random Forest
        #     model = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
        #                             max_depth=50, max_features=None, max_leaf_nodes=80,
        #                             max_samples=None, min_impurity_decrease=0.0,
        #                             min_samples_leaf=1,
        #                             min_samples_split=2, min_weight_fraction_leaf=0.0,
        #                             n_estimators=300, n_jobs=-1, oob_score=False,
        #                             random_state=20, verbose=0, warm_start=False)

    elif Regress_Mode == 3:
        # XGBoost
        model = XGBRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                             max_depth=50, max_features=None, max_leaf_nodes=80,
                             max_samples=None, min_impurity_decrease=0.0,
                             min_impurity_split=None, min_samples_leaf=1,
                             min_samples_split=2, min_weight_fraction_leaf=0.0,
                             n_estimators=300, n_jobs=-1, oob_score=False,
                             random_state=20, verbose=0, warm_start=False)
    elif Regress_Mode == 4:
        # lightGBM
        model = LGBMRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                              max_depth=50, max_features=None, max_leaf_nodes=80,
                              max_samples=None, min_impurity_decrease=0.0,
                              min_impurity_split=None, min_samples_leaf=1,
                              min_samples_split=2, min_weight_fraction_leaf=0.0,
                              n_estimators=300, n_jobs=-1, oob_score=False,
                              random_state=20, verbose=0, warm_start=False)

    model.fit(Data_Train_X, Data_Train_Y)
    pred_y = model.predict(Data_Train_X)  # 训练集预测值 Train_Predict
    pred_yy = model.predict(Data_Test_X)  # 测试集预测值 Test_Predict

    # impute the error parameter
    # R2
    r2_model = sm.r2_score(Data_Train_Y, pred_y)  #impute TrainData's R2
    r2_test = sm.r2_score(Data_Test_Y, pred_yy)  #impute TestData's R2

    # RMSE
    rmse_model = np.sqrt(sm.mean_squared_error(Data_Train_Y, pred_y))  #RMSE impute TrainData's RMESE
    rmse_test = np.sqrt(sm.mean_squared_error(Data_Test_Y, pred_yy))  #RMSE impute TestData's RMSE

    MAE_model = mean_absolute_error(Data_Train_Y, pred_y)
    MAE_test = mean_absolute_error(Data_Test_Y, pred_yy)

    #Keep three decimals
    r2_model = round(r2_model, 3)
    r2_test = round(r2_test, 3)
    rmse_model = round(rmse_model, 3)
    rmse_test = round(rmse_test, 3)
    MAE_model = round(MAE_model, 3)
    MAE_test = round(MAE_test, 3)

    # temp_r
    # tempr = Error Parameter Set
    temp_r = []
    temp_r.append(r2_model)
    temp_r.append(r2_test)
    temp_r.append(rmse_model)
    temp_r.append(rmse_test)
    temp_r.append(MAE_model)
    temp_r.append(MAE_test)

    # plot
    binary_plot(y_train=Data_Train_Y,
                y_train_label=pred_y,
                y_test=Data_Test_Y,
                y_test_label=pred_yy,
                train_rmse=rmse_model,
                test_rmse=rmse_test,
                train_r2=r2_model,
                test_r2=r2_test,
                train_MAE=MAE_model,
                test_MAE=MAE_test)
    save_fig("Result_plot")

    #save the result of predict
    if If_Predict_GlobalData == 1:
        pred_global = model.predict(
            data_apply_global)  # 用我们训练好的model去预测全球数据 using trained model to predict the global garnet data
        np.savetxt('./pred_global.txt', pred_global)
        pred_global = pd.DataFrame(pred_global)
        pred_global.to_csv(r'out_test.txt', sep='\t', index=True, header=True)
    return temp_r



# make plot by sany He
def binary_plot(y_train, y_train_label, y_test, y_test_label,
                train_rmse, test_rmse, train_r2, test_r2, train_MAE, test_MAE,
                text_position=[0.5, -0.075]):
    """plot the binary diagram

    :param y_train: the label of the training data set
    :param y_train_label: the prediction of the training the data set
    :param y_test: the label of the testing data set
    :param y_test_label: the prediction of the testing data set
    :param train_rmse: the RMSE score of the training data set
    :param test_rmse: the RMSE score of the testing data set
    :param train_r2: the R2 score of the training data set
    :param test_r2: the R2 score of the testing data set
    :param test_position: the coordinates of R2 text for
    """

    plt.figure(figsize=(6, 6))
    plt.scatter(y_train, y_train_label, marker="s", c="green",
                label="Training set-RMSE={}".format(train_rmse))
    plt.scatter(y_test, y_test_label, marker="o", c="red",
                label="Test set-RMSE={}".format(test_rmse))
    plt.legend(loc="upper left", fontsize=14)
    plt.xlabel("Reference value", fontsize=20)
    plt.ylabel("Predicted value", fontsize=20)
    a = [0, 1];
    b = [0, 1]
    plt.plot(a, b, c="green")
    plt.text(text_position[0], text_position[1] + 0.275,
             r'$R^2(train)=${}'.format(train_r2),
             fontdict={'size': 16, 'color': '#000000'})
    plt.text(text_position[0], text_position[1] + 0.2,
             r'$R^2(test)=${}'.format(test_r2),
             fontdict={'size': 16, 'color': '#000000'})
    plt.text(text_position[0], text_position[1] + 0.1,
             r'$MAE(train)=${}'.format(train_MAE),
             fontdict={'size': 16, 'color': '#000000'})
    plt.text(text_position[0], text_position[1] + 0.025,
             r'$MAE(test)=${}'.format(test_MAE),
             fontdict={'size': 16, 'color': '#000000'})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim((-0.2, 1.2))


# save figure by sany He
def save_fig(fig_id, tight_layout=True):
    '''
    Run to save automatic pictures

    :param fig_id: image name
    '''
    path = "./Image/" + fig_id + ".png"
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)



