# import numpy as np
# import pandas as pd
# import lightgbm as lgb

# a=[1,3,5,4]
#
# b=np.array(a).reshape(2,2)
# c=[2,3,4]
# c=np.array(c)
# h=np.array([0,1]).reshape(-1,1)
# d=b[h]
# print(b)
# print(d.shape)
# print(d)

import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import graphviz
# X = np.array([1, 2, 2]).reshape((3, 1))
# label = np.array([1, 2, 3])
# print(X.shape)
# print(label.shape)
# data = lightgbm.Dataset(X, label)
#
# # booster = lightgbm.engine.train(
# #     {
# #         "min_data_in_bin": 1,
# #         "min_data_in_leaf": 1,
# #         "learning_rate": 1,
# #         "boost_from_average": False,
# #     },
# #     data,
# #     num_boost_round=2,
# # )
# booster = lightgbm.train(
#     {
#         "min_data_in_bin": 1,
#         "min_data_in_leaf": 1,
#         "learning_rate": 1,
#         "boost_from_average": False,
#     },
#     data,
#     num_boost_round=2,
# )
# booster.predict(X)
# # array([1. , 2.5, 2.5])
#
# # let's refit (to make sure it works)
# booster_refit = booster.refit(X, label, decay_rate=0.0)
# booster_refit.predict(X)
# # array([1. , 2.5, 2.5])
#
# # use weights (I added data_set_kwargs)
# # booster_refit = booster.refit(
# #     X, label, decay_rate=0.0, data_set_kwargs={"weight": np.array([1.0, 0.0, 1.0])}
# # )
# # booster_refit.predict(X)
# # # array([1., 3., 3.])
# #
# # booster_refit = booster.refit(
# #     X, label, decay_rate=0.0, data_set_kwargs={"weight": np.array([1.0, 1.0, 0.0])}
# # )
# booster_refit.predict(X)
# model=lgb.Booster(model_file="F:/NILM/model/lightGBM_dishwasherN1.txt")
# # lgb.plot_tree(model,tree_index=1)
# graph=lgb.create_tree_digraph(model, tree_index=0,precision=7, show_info=[ 'split_gain', 'internal_value', 'internal_count', 'internal_weight', 'leaf_count', 'leaf_weight', 'data_percentage'])
# graph=lgb.create_tree_digraph(model, precision=6, show_info=[ 'split_gain', 'internal_value', 'internal_count', 'internal_weight', 'leaf_count', 'leaf_weight', 'data_percentage'])
# graph.view()
# plt.show()
#a=model.num_trees()
#print(a)
#b=model.trees_to_dataframe()
#b.to_csv("C:/Users/chang/Desktop/treeStructure.csv",header=True, index=False)
#importance=lgb.plot_importance(model, max_num_features=15, precision=6,importance_type='gain')
#plt.show()
# class test():
#     def __init__(self, num_round=10, eta=0.3, gamma=0, Lambda=1, scoring="mse"):
#         self.scoring = scoring
#         self.num_round = num_round
#         self.eta = eta
#         self.gamma = gamma
#         self.Lambda = Lambda
#         self.ensemble = []
#         self.g = None
#         self.h = None
#         self.haty = None
#         self.f = None
#     def fit(self):
#         def matic():
#             print (self.num_round)
#         matic()

# tester=test()
# tester.fit()

# a={6: {"left": 2, "right": {"left2": 3, "right": 3}}}
# b=next(iter(a))
# print(type(b))
import re
# a = int(re.sub("\D", "", str("Column_215")))
# print(a)
# dic=[]
# dic1 = {'a': 1, 'b': 2, 'c': 3, 'd': 1}
# dic2 = {'a': 1, 'b': 2, 'c': 3, 'd': 1}
# dic3 = {'a': 1, 'b': 2, 'c': 3, 'd': 1}
# dic.append(dic1)
# dic.append(dic2)
# dic.append(dic3)
#
# # save to local
# f = open("F:/NILM/test/dict.txt", 'w')
# f.write(str(dic))
# f.close()
# print("save dict successfully.")
#
# # read from local
# f = open("F:/NILM/test/dict.txt", 'r')
# dict_ = eval(f.read())
# #dict_ = f.read()
# f.close()
# print(type(dict_))
# print("read from local : ", dict_[0]['a'])
import pickle
# class tester():
#     def __init__(self, num_round=10, eta=0.3, gamma=0, Lambda=1, scoring="mse"):
#         self.scoring = scoring
#         self.num_round = num_round
#         self.eta = eta
#         self.gamma = gamma
#         self.Lambda = Lambda
#         self.dic = []
#         dic1 = {'a': 1, 'b': 2, 'c': 3, 'd': 1}
#         dic2 = {'a': 1, 'b': 2, 'c': 3, 'd': 1}
#         dic3 = {'a': 1, 'b': 2, 'c': 3, 'd': 1}
#         self.dic.append(dic1)
#         self.dic.append(dic2)
#         self.dic.append(dic3)
# testerf=tester()
# # print(testerf.dic)
# # model= 'F:/NILM/test/model11_1.pkl'
# # with open(model, 'wb+') as f:
# #     bostring= pickle.dump(testerf, f)
#
# model= 'F:/NILM/test/model11_1.pkl'
# with open(model, 'rb+') as f:
#     booster= pickle.load(f)
#
# print(booster.dic)
#

# valueSet=np.arange(1,10)
# valueSet=valueSet.reshape([3,3])
# indexl=valueSet[:,0]>10
# indexr=~indexl
# new=valueSet[indexl]
# index2=new[:,0]>10
# print(~index2)
# index=valueSet[:,0]<= 3
# print(index)
# print(~index)
import pandas as pd
def dataProvider(train1, train2, train3, windowsize):
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
    #print(features1.shape)
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
    # data_frame3 = pd.read_csv(train3,
    #                          #chunksize=10 ** 3,
    #                          header=0
    #                          )

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
trainfile1="F:/NILM/ukdale_training/dishwasher_house_1_training_.csv"
trainfile2="F:/NILM/ukdale_training/dishwasher_house_2_training_.csv"
# trainfile1 = "F:/NILM/training_data/fridge_house_2_training_.csv"
# trainfile2 = "F:/NILM/training_data/fridge_house_3_training_.csv"
#validfile="F:/NILM/datafortrain/dishwasher_validation_.csv"
# testfile="F:/NILM/training_data/fridge_test_.csv"
# X, Y = dataProvider2(trainfile1, trainfile2, windowsize=9)
# print(X.shape)
#
# import os
#
# def dataProvider3(path, windowsize):
#     offset = int(0.5 * (windowsize - 1.0))
#     feature_global = list()
#     label_global = list()
#     feature_global = np.array(feature_global).reshape(-1,windowsize)
#     label_global = np.array(label_global)
#
#     for idx, filename in enumerate(os.listdir(path)):
#         data_frame=pd.read_csv(path+"/"+filename,
#                              #chunksize=10 ** 3,
#                              header=0
#                              )
#         np_array = np.array(data_frame)
#         inputs, targets = np_array[:, 0], np_array[:, 1]
#         window_num = inputs.size - 2 * offset
#         features = list()
#         labels = list()
#         for i in range(0, window_num):
#             inp = inputs[i:i + windowsize]
#             tar = targets[i + offset]
#             features.append(inp)
#             labels.append(tar)
#         features = np.array(features)
#         print(features.shape)
#         labels= np.array(labels)
#         feature_global = np.concatenate((feature_global, features), axis=0)
#         label_global = np.concatenate((label_global, labels), axis=0)
#     return feature_global, label_global
#
# a,b=dataProvider3("F:/NILM/refit_training/dishwasher", 9)
# print(a.shape)
# print(b.shape)


# import tensorflow as tf
#
# import tensorflow.compat.v1 as tf
#
# import tensorflow as tf
# import os
# os.environ['device']='0'

# import numpy as np
# np.zeros((9000, 36, 53806), dtype='uint8')

# import pandas as pd
# dataset = pd.read_csv('C:/Users/chang/Desktop/test.csv',index_col=0)
# print(dataset)
# from sklearn import preprocessing
#
# le = preprocessing.LabelEncoder()
#
# features = [];
# categorical_features = []
# num_of_columns = dataset.shape[1]
#
# for i in range(0, num_of_columns):
#     column_name = dataset.columns[i]
#     column_type = dataset[column_name].dtypes
#
#     if i != num_of_columns - 1:  # skip target
#         features.append(column_name)
#
#     if column_type == 'object':
#         le.fit(dataset[column_name])
#         feature_classes = list(le.classes_)
#         encoded_feature = le.transform(dataset[column_name])
#         print(encoded_feature)
#         dataset[column_name] = pd.DataFrame(encoded_feature)
#
#         if i != num_of_columns - 1:  # skip target
#             categorical_features.append(column_name)
# print(dataset)
# print(categorical_features)
import math


def sortShap():
    arr = [1, 3, 5, 2, 4, 6]
    print(arr[0:-1])
    arr = np.array(arr)
    print(np.argsort(-arr))
    sum=arr.sum()
    sum0=0
    list=[]
    thresh=0.7
    indexlist=np.argsort(-arr)
    print(indexlist)
    for i in indexlist:
        sum0=sum0+arr[i]
        if sum0/sum<thresh:
            list.append(i)
        else:
            break
    return list
#print(sortShap())

a=np.array([1,2,3,4])
b=np.array([3,4,5,6])
# c=a-b
# print(c)
# print(np.subtract(a,b))
# print(np.abs(c))
# print(np.square(c)/b)
# d=np.square(c)/np.square(b)
# print(d)
# print(np.sqrt(d))
# print(math.sqrt(16))

# dif4=[]
# error=[]
# squarey=[]
# for i in range(len(a)):
#     value=a[i]-b[i]
#     dif4.append(abs(value))
#     error.append(value*value)
#     squarey.append(b[i]*b[i])
# print(type(a[2]))
# print(type(error[2]))
# print(type(squarey[2]))
# print(type(dif4[2]))
# print(sum(error))
# print(type(math.sqrt(sum(error)/sum(squarey))))

# x = [0, 0.2, 0.4, 0.6, 0.8, 1]
# y = [0.20729117617943005, 0.22149162907391304, 0.16502827994382369, 0.16752564629207864, 0.1605997683391488, 0.15955847540969817]
# plt.plot(x, y, "#ff7855", marker='>', ms=10, label="a")
# #plt.xticks(rotation=45)
# plt.xlabel("pp")
# plt.ylabel("De")
# plt.title("Test")
# plt.legend(loc="upper left")
# plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

trainfile1 = "/NILM/training_data/washingmachine_house_2_training_.csv"
trainfile2 = "/NILM/training_data/washingmachine_house_3_training_.csv"
testfile="/NILM/training_data/washingmachine_test_.csv"
X1, Y1 = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)

trainfile1 = "/NILM/training_data/fridge_house_2_training_.csv"
trainfile2 = "/NILM/training_data/fridge_house_3_training_.csv"
testfile="/NILM/training_data/fridge_test_.csv"
X2, Y2 = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)


trainfile1 = "/NILM/training_data/dishwasher_house_2_training_.csv"
trainfile2 = "/NILM/training_data/dishwasher_house_3_training_.csv"
testfile="/NILM/training_data/dishwasher_test_.csv"
X3, Y3 = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)


trainfile1 = "/NILM/training_data/microwave_house_2_training_.csv"
trainfile2 = "/NILM/training_data/microwave_house_3_training_.csv"
testfile="、NILM/training_data/microwave_test_.csv"
X4, Y4 = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)
Y1=Y1.reshape(1,-1)
Y2=Y2.reshape(1,-1)
Y3=Y3.reshape(1,-1)
Y4=Y4.reshape(1,-1)
# data = np.concatenate((Y1, Y2), axis=0)
# data = np.concatenate((data, Y3), axis=0)
# data = np.concatenate((data, Y4), axis=0)
# print(data.shape)
# a0=
a1=[0.1941672508356478,0.559674504617506, 0.6352758381815857, 0.6113036519474904]
a2=[0.22831031916587, 0.4797222398872756, 0.5880857576780106, 0.5920185059387647]
a3=[0.8475189288213154,0.7491697116646506,1.1854558971928055,0.9053759581764005]
a4=[0.14917111665468785, 0.5278742082374758, 0.5084279560421191, 0.5236057942650788]
a5=[0.300,0.940,0.730,0.951]
a6=[0.817,0.895,0.960,0.947]
a7=[0.205,0.551,0.620,0.611]
data=[]
data.append(a1)
data.append(a7)
data.append(a2)
data.append(a3)
data.append(a4)
data.append(a5)
data.append(a6)

np.array(data)
# f, (ax1, ax2) = plt.subplots(figsize=(6, 4), nrows=2)

fig1 = plt.figure(figsize=(6, 6))
ax1 = fig1.add_subplot(111)

#cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
# pt = df.corr()  

#sns.heatmap(data, annot=True,linewidths=0.05, ax=ax1, vmax=1.2, vmin=0, cmap=cmap)
sns.heatmap(data, annot=True,linewidths=1.5, ax=ax1, vmax=1.2, vmin=0, cbar_kws={'label': 'Normalised Disaggregation Error'},cmap='bwr',annot_kws={'size':20,'weight':'bold'})
ax1.figure.axes[-1].yaxis.label.set_size(20)
ax1.figure.axes[-1].yaxis.label.set_weight('semibold')
sns.set(font_scale=1.5)
#ax1.set_title('Normalised Disaggregation Error',fontsize=18, weight='semibold')
ax1.set_ylabel('Method',fontsize=22, weight='semibold')
# ax1.set_ylim([11, 1])
ax1.set_yticklabels(['CNN', 'CNN\n'+'(Domain Adapted)','LightGBM', 'LightGBM\n'+'(Direct Transferred)', 'LightGBM\n'+'(Domain Adapted)', 'GAN', 'GAN\n'+'(Domain Adapted)'])
ax1.set_xticklabels(['Washing Machine','Fridge', 'Dishwasher', 'Microwave'])# 设置x轴图例为空值
ax1.set_xlabel('Appliance', fontsize=22, weight='semibold')
#plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tick_params(labelsize=20)
# plt.rc('font',family='Times New Roman',size=12)
plt.show()


# sns.heatmap(pt, linewidths=0.05, ax=ax2, vmax=900, vmin=0, cmap='rainbow')
# ax2.set_title('matplotlib colormap')
# ax2.set_xlabel('region')
