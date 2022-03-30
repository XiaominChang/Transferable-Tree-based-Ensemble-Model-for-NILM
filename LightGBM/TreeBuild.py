import numpy as np
import lightgbm as lgb
import os
import pandas as pd
import sklearn
from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, zero_one_loss
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
# import graphviz
# import shap
import lightgbm as lgb
import time
from math import sqrt
import re
import os
import math
import pickle
from sklearn.metrics import mean_squared_error, zero_one_loss,mean_absolute_error,r2_score
from random import sample
class Try_LighGBM():
    def __init__(self, num_round=10, eta=0.23179, gamma=0, Lambda=0.7145, scoring="mse"):
        self.scoring = scoring
        self.num_round = num_round
        self.eta = eta
        self.gamma = gamma
        self.Lambda = Lambda
        self.ensemble = []
        self.g = None
        self.h = None
        self.haty = None
        self.f = None
        self.oldTrees=pd.read_csv("/treeStructure/ukdale/treeStructure_fridge.csv").groupby(['tree_index'])
        self.treeIndex=self.oldTrees.size().values.tolist()
        self.treenum=len(self.treeIndex)
        self.oldtree=None
        self.bins= 500
        self.shaplist= [9, 8, 0, 18, 10, 7, 3, 15, 1]

    def _G(self, y_train):
        return -2 * (y_train - self.haty)

    def _Gain(self, listL, listR):
        GL = self.g[listL].sum()
        GR = self.g[listR].sum()
        HL = self.h[listL].sum()
        HR = self.h[listR].sum()

        return (GL ** 2 / (HL + self.Lambda) + GR ** 2 / (HR + self.Lambda) - (GR + GL) ** 2 / (
                    HL + HR + self.Lambda)) / 2 - self.gamma

    def _w(self, indexlist):
        return -np.sum(self.g[indexlist]) / (np.sum(self.h[indexlist]) + self.Lambda)

    def fit(self, X_train, y_train):

        def BestSplit(X_train, y_train, indexlist, bestSplitFeature):
            bestGain = 0
            bestSplitValue = -1
            print(len(X_train))
            if bestSplitFeature in self.shaplist:
                print("yes")
                AllValue = sorted(set(X_train[:, bestSplitFeature]))
                if len(AllValue) > self.bins:
                    try:
                        ValueSet = sample(AllValue, self.bins)
                    except:
                        print("length of Allvalue is :", len(AllValue))
                        print("length of self.bins is :", self.bins)
                else:
                    ValueSet = AllValue
                for Val in ValueSet:
                    boolindexLeft = X_train[:, bestSplitFeature] <= Val
                    boolindexRight = ~boolindexLeft
                    indexLeft = indexlist[boolindexLeft]
                    indexRight = indexlist[boolindexRight]
                    gain = self._Gain(indexLeft, indexRight)

                    if gain > bestGain:
                        bestGain = gain
                        bestSplitValue = Val
            else:
                for feature in range(X_train.shape[1]):
                    AllValue = sorted(set(X_train[:, feature]))
                    if len(AllValue) > self.bins:
                        try:
                            ValueSet = sample(AllValue, self.bins)
                        except:
                            print("length of Allvalue is :", len(AllValue))
                            print("length of self.bins is :", self.bins)
                    else:
                        ValueSet = AllValue
                    for Val in ValueSet:
                        boolindexLeft = X_train[:, feature] <= Val
                        boolindexRight = ~boolindexLeft
                        indexLeft = indexlist[boolindexLeft]
                        indexRight = indexlist[boolindexRight]
                        gain = self._Gain(indexLeft, indexRight)

                        if gain > bestGain:
                            bestGain = gain
                            bestSplitFeature=feature
                            bestSplitValue = Val
            return bestSplitFeature, bestSplitValue
            # for Feature in range(X_train.shape[1]):
            #     ValueSet = set(X_train[:, Feature])
            #     for Val in ValueSet:
            #         boolindexLeft = X_train[:, Feature] <= Val
            #         boolindexRight = ~boolindexLeft
            #         indexLeft = indexlist[boolindexLeft]
            #         indexRight = indexlist[boolindexRight]
            #         gain = self._Gain(indexLeft, indexRight)
            #         if gain > bestGain:
            #             bestGain = gain
            #             bestSplitFeature = Feature
            #             bestSplitValue = Val
            # if bestSplitFeature == -1:
            #     return None, None
            # else:
            #     return bestSplitFeature, bestSplitValue

        def create_tree(X_train, y_train, node, indexlist=np.arange(len(X_train))):
            bestSplitFeature = str(node['split_feature'])
            if (bestSplitFeature=='nan' or len(indexlist)<500):
                # print("yes")
                w = self._w(indexlist)
                self.f[indexlist] = w
                return w
            else:
                # # print(type(bestSplitFeature))
                # print(str(bestSplitFeature))
                # print((re.sub("\D", "", str(bestSplitFeature))))
                bestSplitFeature = int(re.sub("\D", "", str(bestSplitFeature)))
                # print(bestSplitFeature)
                bestSplitFeature, bestSplitValue = BestSplit(X_train, y_train, indexlist, bestSplitFeature)
                # print("this node's bestSplitvalue is: \n")
                # print(bestSplitValue)
                left_index = X_train[:, bestSplitFeature] <= bestSplitValue
                # left_index = X_train[:, bestSplitFeature]<=int(node['threshold'])
                sub_X_train_left, sub_y_train_left = X_train[left_index], y_train[left_index]
                sub_X_train_right, sub_y_train_right = X_train[~left_index], y_train[~left_index]
                indexlist_left = indexlist[left_index]
                indexlist_right = indexlist[~left_index]
                left_nodeindex=str(node['left_child'])
                right_nodeindex=str(node['right_child'])
                left_node=self.oldtree[self.oldtree['node_index']==left_nodeindex]
                left_node=left_node.reset_index(drop=True)
                left_node=left_node.loc[0,:]
                right_node=self.oldtree[self.oldtree['node_index']==right_nodeindex]
                right_node=right_node.reset_index(drop=True)
                right_node=right_node.loc[0,:]
                leftchild = create_tree(sub_X_train_left, sub_y_train_left, left_node, indexlist=indexlist_left)
                rightchild = create_tree(sub_X_train_right, sub_y_train_right, right_node, indexlist=indexlist_right)
                return {bestSplitFeature: {"<={}".format(bestSplitValue): leftchild,
                                            ">{}".format(bestSplitValue): rightchild}}

        self.haty = np.zeros(len(X_train))
        self.h = np.ones(len(X_train)) * 2

        for i in range(self.treenum):
            self.oldtree=self.oldTrees.get_group(i)
            self.oldtree=self.oldtree.reset_index(drop=True)
            self.g = self._G(y_train)
            self.f = np.empty(len(X_train))
            root=self.oldtree.loc[0,:]
            newtree = create_tree(X_train, y_train, root)
            print('have finished one tree: ', i)
            self.ensemble.append(newtree)
            self.haty = self.haty + self.eta * self.f
        f = open("F:/NILM/test/modeldic.txt", 'w')
        f.write(str(self.ensemble))
        f.close()
        return

    def draw_one_tree(self, index):
        from graphviz import Digraph
        def export_graphviz(tree, root_index):
            root = next(iter(tree))
            text_node.append([str(root_index), "feature:{}".format(root)])
            secondDic = tree[root]
            for key in secondDic:
                if type(secondDic[key]) == dict:
                    i[0] += 1
                    secondrootindex = i[0]
                    text_edge.append([str(root_index), str(secondrootindex), str(key)])
                    export_graphviz(secondDic[key], secondrootindex)
                else:
                    i[0] += 1
                    text_node.append([str(i[0]), str(secondDic[key])])
                    text_edge.append([str(root_index), str(i[0]), str(key)])

        tree = self.ensemble[index]
        text_node = []
        text_edge = []
        i = [1]
        export_graphviz(tree, i[0])
        dot = Digraph()
        for line in text_node:
            dot.node(line[0], line[1])
        for line in text_edge:
            dot.edge(line[0], line[1], line[2])

        dot.view()

    def predict(self, X_test):
        return np.array([self._predict(test) for test in X_test])

    def _predict(self, test):
        def __predict(tree, test):
            feature = next(iter(tree))
            secondDic = tree[feature]
            content = test[feature]
            for key in secondDic:
                if eval(str(content) + key):
                    if type(secondDic[key]) == dict:
                        return __predict(secondDic[key], test)
                    else:
                        return secondDic[key]

        assert len(self.ensemble) != 0, "fit before predict"
        res = 0
        for i in range(len(self.ensemble)):
            tree = self.ensemble[i]
            res_temp = __predict(tree, test)
            res += res_temp * self.eta
        return res

    def score(self, X_test, y_test):
        y_pre = self.predict(X_test)
        if self.scoring == "mse":
            return sum((y_test - y_pre) ** 2) / len(X_test)
        elif self.scoring == "r2":
            return 1 - sum((y_test - y_pre) ** 2) / sum((y_test - y_test.mean()) ** 2)

    def get_params(self, deep=False):
        dic = {}
        dic["num_round"] = self.num_round
        dic["eta"] = self.eta
        dic["gamma"] = self.gamma
        dic["Lambda"] = self.Lambda
        dic["scoring"] = self.scoring
        return dic

# windowsize=79
# offset = int(0.5*(windowsize-1.0))
# trainfile1 = "F:/NILM/training_data/fridge_house_2_training_.csv"
# trainfile2 = "F:/NILM/training_data/fridge_house_3_training_.csv"
# #validfile="F:/NILM/datafortrain/dishwasher_validation_.csv"
# testfile="F:/NILM/training_data/fridge_test_.csv"

# trainfile1="F:/NILM/ukdale_training/fridge_house_1_training_.csv"
# trainfile2="F:/NILM/ukdale_training/fridge_house_2_training_.csv"
def dataProvider(train1, train2, train3,windowsize):
    offset = int(0.5 * (windowsize - 1.0))
    data_frame1 = pd.read_csv(train1,
                             #chunksize=10 ** 3,
                             header=0
                             )
    data_frame2 = pd.read_csv(train2,
                             #chunksize=10 ** 3,
                             header=0
                             )
    data_frame3 = pd.read_csv(train3,
                             #chunksize=10 ** 3,
                             header=0
                             )

    np_array = np.array(data_frame1)
    inputs, targets = np_array[:, 0], np_array[:, 1]
    window_num=inputs.size - 2 * offset
    features=list()
    labels=list()
    for i in range(0,window_num):
        inp=inputs[i:i+windowsize]
        tar=targets[i+offset]
        features.append(inp)
        labels.append(tar)
    features0=np.array(features)
    labels0=np.array(labels)

    np_array = np.array(data_frame2)
    inputs, targets = np_array[:, 0], np_array[:, 1]
    window_num=inputs.size - 2 * offset
    features=list()
    labels=list()
    for i in range(0,window_num):
        inp=inputs[i:i+windowsize]
        tar=targets[i+offset]
        features.append(inp)
        labels.append(tar)
    features1=np.array(features)
    labels1=np.array(labels)

    np_array = np.array(data_frame3)
    inputs, targets = np_array[:, 0], np_array[:, 1]
    window_num=inputs.size - 2 * offset
    features=list()
    labels=list()
    for i in range(0,window_num):
        inp=inputs[i:i+windowsize]
        tar=targets[i+offset]
        features.append(inp)
        labels.append(tar)
    features2=np.array(features)
    labels2=np.array(labels)

    feature=np.concatenate((features0, features1), axis=0)
    feature=np.concatenate((feature, features2), axis=0)
    label=np.concatenate((labels0, labels1), axis=0)
    label=np.concatenate((label, labels2), axis=0)
    return feature, label


def dataProvider2(train1, train2, windowsize):
    offset = int(0.5 * (windowsize - 1.0))
    data_frame1 = pd.read_csv(train1,
                             #chunksize=10 ** 3,
                             header=0
                             )
    data_frame2 = pd.read_csv(train2,
                             #chunksize=10 ** 3,
                             header=0
                             )

    np_array = np.array(data_frame1)
    inputs, targets = np_array[:, 0], np_array[:, 1]
    window_num=inputs.size - 2 * offset
    features=list()
    labels=list()
    for i in range(0,window_num):
        inp=inputs[i:i+windowsize]
        tar=targets[i+offset]
        features.append(inp)
        labels.append(tar)
    features0=np.array(features)
    labels0=np.array(labels)

    np_array = np.array(data_frame2)
    inputs, targets = np_array[:, 0], np_array[:, 1]
    window_num=inputs.size - 2 * offset
    features=list()
    labels=list()
    for i in range(0,window_num):
        inp=inputs[i:i+windowsize]
        tar=targets[i+offset]
        features.append(inp)
        labels.append(tar)
    features1=np.array(features)
    labels1=np.array(labels)

    # np_array = np.array(data_frame3)
    # inputs, targets = np_array[:, 0], np_array[:, 1]
    # window_num=inputs.size - 2 * offset
    # features=list()
    # labels=list()
    # for i in range(0,window_num):
    #     inp=inputs[i:i+windowsize]
    #     tar=targets[i+offset]
    #     features.append(inp)
    #     labels.append(tar)
    # features2=np.array(features)
    # labels2=np.array(labels)

    feature=np.concatenate((features0, features1), axis=0)
    label=np.concatenate((labels0, labels1), axis=0)
    return feature, label

def dataProvider3(path, windowsize):
    offset = int(0.5 * (windowsize - 1.0))
    feature_global = list()
    label_global = list()
    feature_global = np.array(feature_global).reshape(-1,windowsize)
    label_global = np.array(label_global)

    for idx, filename in enumerate(os.listdir(path)):
        data_frame=pd.read_csv(path+"/"+filename,
                             #chunksize=10 ** 3,
                             header=0
                             )
        np_array = np.array(data_frame)
        inputs, targets = np_array[:, 0], np_array[:, 1]
        window_num = inputs.size - 2 * offset
        features = list()
        labels = list()
        for i in range(0, window_num):
            inp = inputs[i:i + windowsize]
            tar = targets[i + offset]
            features.append(inp)
            labels.append(tar)
        features = np.array(features)
        print(features.shape)
        labels= np.array(labels)
        feature_global = np.concatenate((feature_global, features), axis=0)
        label_global = np.concatenate((label_global, labels), axis=0)
    return feature_global, label_global

X, Y = dataProvider(testfile, trainfile1, trainfile2, 19)

x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, Y, test_size=0.2, random_state=100)
x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)
booster1=Try_LighGBM()
booster1.fit(x_train_all, y_train_all)
model= '/NILM/model/transfer/model_fridge.pkl'
with open(model, 'wb+') as f:
     boost= pickle.dump(booster1, f)
prediction=booster1.predict(x_predict)
print("the new model mae is :", mean_absolute_error(y_predict, prediction))
r = y_predict.sum()
r0 = prediction.sum()
print("the new model sae is :", abs(r0 - r) / r)















