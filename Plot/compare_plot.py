import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, zero_one_loss,mean_absolute_error,r2_score
import matplotlib.pyplot as plt
import time
from math import sqrt

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

# model= 'F:/NILM/test/model.pkl'
# with open(model, 'rb+') as f:
#     booster= pickle.load(f)
# booster.draw_one_tree(0)
# trainfile1="F:/NILM/ukdale_training/fridge_house_1_training_.csv"
# trainfile2="F:/NILM/ukdale_training/fridge_house_2_training_.csv"

trainfile1 = "F:/NILM/training_data/washingmachine_house_2_training_.csv"
trainfile2 = "F:/NILM/training_data/washingmachine_house_3_training_.csv"
testfile="F:/NILM/training_data/washingmachine_test_.csv"
X1, Y1 = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)

trainfile1 = "F:/NILM/training_data/fridge_house_2_training_.csv"
trainfile2 = "F:/NILM/training_data/fridge_house_3_training_.csv"
testfile="F:/NILM/training_data/fridge_test_.csv"
X2, Y2 = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)


trainfile1 = "F:/NILM/training_data/dishwasher_house_2_training_.csv"
trainfile2 = "F:/NILM/training_data/dishwasher_house_3_training_.csv"
testfile="F:/NILM/training_data/dishwasher_test_.csv"
X3, Y3 = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)


trainfile1 = "F:/NILM/training_data/microwave_house_2_training_.csv"
trainfile2 = "F:/NILM/training_data/microwave_house_3_training_.csv"
testfile="F:/NILM/training_data/microwave_test_.csv"
X4, Y4 = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)

sum_real=Y1.sum()+Y2.sum()+Y3.sum()+Y4.sum()
print("washingmachine rate is: ", Y1.sum()/sum_real)
print("fridge rate is: ", Y2.sum()/sum_real)
print("dishwasher rate is: ", Y3.sum()/sum_real)
print("microwave rate is: ", Y4.sum()/sum_real)

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

labels = ['washingmachine', 'fridge', 'dishwasher', 'microwave']
X = [Y1.sum(),Y2.sum(),Y3.sum(),Y3.sum()]

fig = plt.figure()
plt.pie(X, labels=labels, autopct='%1.2f%%')
#plt.title("Ground Truth")
# plt.show()
plt.savefig("F:/NILM/model/redd/groundtruth_redd.pdf", format='pdf', dpi=1000, bbox_inches='tight')
plt.close()

print("prediction is as follow \n")
lgbm1=lgb.Booster(model_file="F:/NILM/model/ukdale/lightGBM_washingmachineN1.txt")
prediction1=lgbm1.predict(X1)
#
# lgbm2=lgb.Booster(model_file="F:/NILM/model/ukdale/lightGBM_fridgeN1.txt")
# prediction2=lgbm2.predict(X2)
#
# lgbm3=lgb.Booster(model_file="F:/NILM/model/ukdale/lightGBM_dishwasherN1.txt")
# prediction3=lgbm3.predict(X3)
# #
# lgbm4=lgb.Booster(model_file="F:/NILM/model/ukdale/lightGBM_microwaveN1.txt")
# prediction4=lgbm4.predict(X4)
#
# sum_predict=prediction1.sum()+prediction2.sum()+prediction3.sum()+prediction4.sum()
# print("washingmachine rate is: ", prediction1.sum()/sum_predict)
# print("fridge rate is: ", prediction2.sum()/sum_predict)
# print("dishwasher rate is: ", prediction3.sum()/sum_predict)
# print("microwave rate is: ", prediction4.sum()/sum_predict)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

ax1.plot(X1[:,9], color='#7f7f7f', linewidth=1.8)
ax1.plot(Y1[:], color='#FF0000', linewidth=1.6)
ax1.plot(prediction1[:],
         color='#00BFFF',
         # marker='o',
         linewidth=1.5)
ax1.plot(Y2[:], color='#9400D3', linewidth=1.6)
ax1.grid()
#ax1.set_title('Test results on {:}', fontsize=16, fontweight='bold', y=1.08)
#ax1.set_ylabel('W')
ax1.legend(['Aggregate', 'Ground Truth', 'Prediction with REDD Model', 'Prediction with Transferred Model'],prop={'size': 16})

mng = plt.get_current_fig_manager()
plt.ylim(0)
plt.xlabel('Time(mins)',fontsize=16)
plt.ylabel('Power Consumption(W)',fontsize=16)
# mng.resize(*mng.window.maxsize())
plt.tick_params(labelsize=16)
# plt.legend(prop={'size': 16})

plt.show(fig1)

# labels1 = ['washingmachine', 'fridge', 'dishwasher', 'microwave']
# X1= [prediction1.sum(),prediction2.sum(),prediction3.sum(),prediction4.sum()]
#
# fig0 = plt.figure()
# plt.pie(X1, labels=labels1, autopct='%1.2f%%')
# #plt.title("Prediction by UKdale model")
# # plt.show()
# plt.savefig("F:/NILM/model/redd/predict1_redd.pdf", format='pdf', dpi=1000, bbox_inches='tight')
# plt.close()

# print("the original model mae is :", mean_absolute_error(Y, prediction))
