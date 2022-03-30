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


import matplotlib.pyplot as plt


trainfile1 = "/training_data/washingmachine_house_2_training_.csv"
trainfile2 = "/training_data/washingmachine_house_3_training_.csv"
testfile="/training_data/washingmachine_test_.csv"
X1, Y1 = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)

trainfile1 = "/training_data/fridge_house_2_training_.csv"
trainfile2 = "/training_data/fridge_house_3_training_.csv"
testfile="/training_data/fridge_test_.csv"
X2, Y2 = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)


trainfile1 = "/training_data/dishwasher_house_2_training_.csv"
trainfile2 = "training_data/dishwasher_house_3_training_.csv"
testfile="/training_data/dishwasher_test_.csv"
X3, Y3 = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)


trainfile1 = "/training_data/microwave_house_2_training_.csv"
trainfile2 = "/training_data/microwave_house_3_training_.csv"
testfile="/training_data/microwave_test_.csv"
X4, Y4 = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)


lgbm1=lgb.Booster(model_file="/model/redd/lightGBM_washingmachineN1.txt")
prediction01=lgbm1.predict(X1)

lgbm2=lgb.Booster(model_file="/model/redd/lightGBM_fridgeN1.txt")
prediction02=lgbm2.predict(X2)

lgbm3=lgb.Booster(model_file="/model/redd/lightGBM_dishwasherN1.txt")
prediction03=lgbm3.predict(X3)

lgbm4=lgb.Booster(model_file="/model/redd/lightGBM_microwaveN1.txt")
prediction04=lgbm4.predict(X4)


model= 'model/transfer/model_washingmachine.pkl'
with open(model, 'rb+') as f:
    lgbm1= pickle.load(f)
prediction1=lgbm1.predict(X1[438040:438570])

model= '/model/transfer/model_fridge.pkl'
with open(model, 'rb+') as f:
    lgbm2= pickle.load(f)
prediction2=lgbm2.predict(X2[21500:23850])

model= '/model/transfer/model_dishwasher.pkl'
with open(model, 'rb+') as f:
    lgbm3= pickle.load(f)
prediction3=lgbm3.predict(X3[189800:190850])

model= '/model/transfer/model_microwave.pkl'
with open(model, 'rb+') as f:
    lgbm4= pickle.load(f)
prediction4=lgbm4.predict(X4[63150:63850])

#fridge
fig1 = plt.figure()
ax1 = fig1.add_subplot(2,2,1)

ax1.plot(X2[21500:23850,9], color='#7f7f7f', linewidth=1.8)
ax1.plot(Y2[21500:23850], color='r', linewidth=1.4)
ax1.plot(prediction02[21500:23850],
         color='#00BFFF',
         # marker='o',
         linewidth=1.6)
ax1.plot(prediction2, color='#9400D3', linewidth=1.4)
#ax1.grid()

#ax1.legend(['Aggregate', 'Ground Truth', 'Prediction with REDD Model', 'Prediction with Transferred Model'],loc=1,prop={'size': 20, 'weight':'semibold'})

mng = plt.get_current_fig_manager()
plt.title("Fridge",fontsize=24,weight='semibold')
plt.ylim(0)
plt.xlim(-50,2350)
plt.xticks([0,500,1000,1500,2000],[0,4000,8000,12000,16000])
plt.xlabel('Time(s)',fontsize=24,weight='semibold')
plt.ylabel('Power Consumption(W)',fontsize=24,weight='semibold')
plt.grid()
# mng.resize(*mng.window.maxsize())
plt.tick_params(labelsize=24 )
# plt.legend(prop={'size': 16})




#washing machine figure

ax2 = fig1.add_subplot(2,2,2)

ax2.plot(X1[438040:438570,9], color='#7f7f7f', linewidth=1.8)
ax2.plot(Y1[438040:438570], color='r', linewidth=1.4)
ax2.plot(prediction01[438040:438570],
         color='#00BFFF',
         # marker='o',
         linewidth=1.6)
ax2.plot(prediction1, color='#9400D3', linewidth=1.4)
ax2.grid()

#ax2.legend(['Aggregate', 'Ground Truth', 'Prediction with REDD Model', 'Prediction with Transferred Model'],loc=1,prop={'size': 20,'weight':'semibold'})

mng = plt.get_current_fig_manager()
plt.title("Washing Machine",fontsize=24,weight='semibold')
plt.ylim(0,6600)
plt.xlim(-10,520)
plt.xticks([0,100,200,300,400,500],[0,800,1600,2400,3200,4000])
plt.xlabel('Time(s)',fontsize=24,weight='semibold')
plt.ylabel('Power Consumption(W)',fontsize=24,weight='semibold')
# mng.resize(*mng.window.maxsize())
plt.tick_params(labelsize=24)
# plt.legend(prop={'size': 16})



#dishwasher

ax3 = fig1.add_subplot(2,2,3)

ax3.plot(X3[189800:190850,9], color='#7f7f7f', linewidth=1.8)
ax3.plot(Y3[189800:190850], color='r', linewidth=1.4)
ax3.plot(prediction03[189800:190850],
         color='#00BFFF',
         # marker='o',
         linewidth=1.6)
ax3.plot(prediction3, color='#9400D3', linewidth=1.4)
ax3.grid()

#ax3.legend(['Aggregate', 'Ground Truth', 'Prediction with REDD Model', 'Prediction with Transferred Model'],loc=1,prop={'size': 20,'weight':'semibold'})

mng = plt.get_current_fig_manager()
plt.title("Dishwasher",fontsize=24,weight='semibold')
plt.ylim(0)
plt.xlim(-20,1050)
plt.xticks([0,200,400,600,800,1000],[0,1600,3200,4800,6400,8000])
plt.xlabel('Time(s)',fontsize=24,weight='semibold')
plt.ylabel('Power Consumption(W)',fontsize=24,weight='semibold')
# mng.resize(*mng.window.maxsize())
plt.tick_params(labelsize=24)
# plt.legend(prop={'size': 16})



#microwave

ax4 = fig1.add_subplot(2,2,4)

ax4.plot(X4[63150:63850,9], color='#7f7f7f', linewidth=1.8)
ax4.plot(Y4[63150:63850], color='r', linewidth=1.4)
ax4.plot(prediction04[63150:63850],
         color='#00BFFF',
         # marker='o',
         linewidth=1.6)
ax4.plot(prediction4, color='#9400D3', linewidth=1.4)
ax4.grid()

#ax4.legend(['Aggregate', 'Ground Truth', 'Prediction with REDD Model', 'Prediction with Transferred Model'],loc=1,prop={'size': 20,'weight':'semibold'})

mng = plt.get_current_fig_manager()
plt.title("Microwave",fontsize=24,weight='semibold')
plt.ylim(0)
plt.xlim(-20,650)
plt.xticks([0,100,200,300,400,500,600],[0,800,1600,2400,3200,4000,4800])
plt.xlabel('Time(s)',fontsize=24,weight='semibold')
plt.ylabel('Power Consumption(W)',fontsize=24,weight='semibold')
# mng.resize(*mng.window.maxsize())
plt.tick_params(labelsize=24)
# plt.legend(prop={'size': 16})
plt.legend(['Aggregate', 'Ground Truth', 'Prediction with REDD Model', 'Prediction with Transferred Model'],bbox_to_anchor=(0.84, 2.75),ncol=4,prop={'size': 20, 'weight':'semibold'})
plt.show()



















# sum_predict=prediction1.sum()+prediction2.sum()+prediction3.sum()+prediction4.sum()
# print("washingmachine rate is: ", prediction1.sum()/sum_predict)
# print("fridge rate is: ", prediction2.sum()/sum_predict)
# print("dishwasher rate is: ", prediction3.sum()/sum_predict)
# print("microwave rate is: ", prediction4.sum()/sum_predict)















# labels1 = ['washingmachine', 'fridge', 'dishwasher', 'microwave']
# X1= [prediction1.sum(),prediction2.sum(),prediction3.sum(),prediction4.sum()]
#
# fig0 = plt.figure()
# plt.pie(X1, labels=labels1, autopct='%1.2f%%')
# #plt.title("Prediction by UKdale model")
# # plt.show()
# plt.savefig("F:/NILM/model/redd/predict1_newforests.pdf", format='pdf', dpi=1000, bbox_inches='tight')
# plt.close()








# booster1=lgb.Booster(model_file="F:/NILM/model/ukdale/lightGBM_fridgeN1.txt")
# prediction=booster1.predict(x_predict)
# print("the new model mae is :", mean_absolute_error(y_predict, prediction))
# r = y_predict.sum()
# r0 = prediction.sum()
# print("the new model sae is :", abs(r0 - r) / r)





# model= 'F:/NILM/model/transfer/model_washingmachine.pkl'
# with open(model, 'rb+') as f:
#     lgbm1= pickle.load(f)
# prediction1=lgbm1.predict(X1[438040:438570])

# model= 'F:/NILM/model/transfer/model_fridge.pkl'
# with open(model, 'rb+') as f:
#     lgbm2= pickle.load(f)
# prediction2=lgbm2.predict(X2[21500:23850])
#
# model= 'F:/NILM/model/transfer/model_dishwasher.pkl'
# with open(model, 'rb+') as f:
#     lgbm3= pickle.load(f)
# prediction3=lgbm3.predict(X3[189800:190850])
#
# model= 'F:/NILM/model/transfer/model_microwave.pkl'
# with open(model, 'rb+') as f:
#     lgbm4= pickle.load(f)
# prediction4=lgbm4.predict(X4[63150:63850])


# x_train_all1, x_predict1, y_train_all1, y_predict1 = train_test_split(X1, Y1, test_size=0.2, random_state=100)
# Y1=y_predict1
# X1=x_predict1
# x_train_all1, x_predict1, y_train_all1, y_predict1 = train_test_split(X1, Y1, test_size=0.2, random_state=100)
# Y1=y_predict1
# X1=x_predict1
# model= 'F:/NILM/model/transfer/model_washingmachine.pkl'
# with open(model, 'rb+') as f:
#     lgbm1= pickle.load(f)
# prediction1=lgbm1.predict(X1)
# dif1=[]
# error=[]
# squarey=[]
# for i in range(len(Y1)):
#     value=prediction1[i]-Y1[i]
#     dif1.append(abs(value))
#     error.append(value*value)
#     squarey.append(Y1[i]*Y1[i])
# de1=sqrt(sum(error)/sum(squarey))
# print("washing machine de1 is :", de1)
#print("square is:",prediction1)
# de1=np.sqrt(np.square(prediction1-Y1).sum()/np.square(Y1).sum())


#print("mae is :", mean_absolute_error(Y1, prediction1))
# r = y_predict.sum()
# r0 = prediction.sum()
# sae = abs(r0 - r) / r
# print("washing machine sae is :", sae)

# trainfile1="F:/NILM/ukdale_training/microwave_house_1_training_.csv"
# trainfile2="F:/NILM/ukdale_training/microwave_house_2_training_.csv"
# X2,Y2=dataProvider4(trainfile2, windowsize=19)
# x_train_all2, x_predict2, y_train_all2, y_predict2 = train_test_split(X, Y, test_size=0.2, random_state=100)
# del X, Y
# trainfile1 = "F:/NILM/training_data/microwave_house_2_training_.csv"
# trainfile2 = "F:/NILM/training_data/microwave_house_3_training_.csv"
# testfile="F:/NILM/training_data/microwave_test_.csv"
# X2, Y2 = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)
# trainfile1="F:/NILM/ukdale_training/microwave_house_1_training_.csv"
# trainfile2="F:/NILM/ukdale_training/microwave_house_2_training_.csv"
# X2,Y2=dataProvider4(trainfile2, windowsize=19)
# x_train_all2, x_predict2, y_train_all2, y_predict2 = train_test_split(X, Y, test_size=0.2, random_state=100)
# del X, Y
# X2,Y2=dataProvider2(trainfile1,trainfile2, windowsize=19)
# x_train_all2, x_predict2, y_train_all2, y_predict2 = train_test_split(X2, Y2, test_size=0.2, random_state=100)
# Y2=y_predict2
# X2=x_predict2
# model= 'F:/NILM/model/transfer/model_fridge.pkl'
# with open(model, 'rb+') as f:
#     lgbm2= pickle.load(f)
# prediction2=lgbm2.predict(X2)
# dif2=[]
# error=[]
# squarey=[]
# for i in range(len(Y2)):
#     value=prediction2[i]-Y2[i]
#     dif2.append(abs(value))
#     error.append(value*value)
#     squarey.append(Y2[i]*Y2[i])
# de2=sqrt(sum(error)/sum(squarey))
# print("fridge de1 is :", de2)
#print("mae is :", mean_absolute_error(Y2, prediction2))
# r = y_predict.sum()
# r0 = prediction.sum()
# sae = abs(r0 - r) / r
# print("microwave sae is :", sae)

# trainfile1="F:/NILM/ukdale_training/fridge_house_2_training_.csv"
# X3,Y3=dataProvider4(trainfile1, windowsize=19)
# x_train_all3, x_predict3, y_train_all3, y_predict3 = train_test_split(X, Y, test_size=0.2, random_state=100)
# del X, Y
# trainfile1 = "F:/NILM/training_data/fridge_house_2_training_.csv"
# trainfile2 = "F:/NILM/training_data/fridge_house_3_training_.csv"
# testfile="F:/NILM/training_data/fridge_test_.csv"
# X3, Y3 = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)
# trainfile1="F:/NILM/ukdale_training/fridge_house_2_training_.csv"
# X3,Y3=dataProvider4(trainfile1, windowsize=19)
# x_train_all3, x_predict3, y_train_all3, y_predict3 = train_test_split(X, Y, test_size=0.2, random_state=100)
# del X, Y
# x_train_all3, x_predict3, y_train_all3, y_predict3 = train_test_split(X3, Y3, test_size=0.2, random_state=100)
# # Y3=y_predict3
# # X3=x_predict3
# model= 'F:/NILM/model/transfer/model_dishwasher.pkl'
# with open(model, 'rb+') as f:
#     lgbm3= pickle.load(f)
# prediction3=lgbm3.predict(X3)
# dif3=[]
# error=[]
# squarey=[]
# for i in range(len(Y3)):
#     value=prediction3[i]-Y3[i]
#     dif3.append(abs(value))
#     error.append(value*value)
#     squarey.append(Y3[i]*Y3[i])
# de3=sqrt(sum(error)/sum(squarey))
# print("dishwasher de1 is :", de3)
#print("mae is :", mean_absolute_error(Y3, prediction3))
# r = y_predict.sum()
# r0 = prediction.sum()
# sae = abs(r0 - r) / r
# print("fridge sae is :", sae)

# trainfile1="F:/NILM/ukdale_training/dishwasher_house_1_training_.csv"
# trainfile2="F:/NILM/ukdale_training/dishwasher_house_2_training_.csv"
# X4,Y4=dataProvider4(trainfile2, windowsize=19)
# x_train_all4, x_predict4, y_train_all4, y_predict4 = train_test_split(X, Y, test_size=0.2, random_state=100)
# del X, Y
# trainfile1 = "F:/NILM/training_data/dishwasher_house_2_training_.csv"
# # trainfile2 = "F:/NILM/training_data/dishwasher_house_3_training_.csv"
# # testfile="F:/NILM/training_data/dishwasher_test_.csv"
# # X4, Y4 = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)
# trainfile1="F:/NILM/ukdale_training/dishwasher_house_1_training_.csv"
# trainfile2="F:/NILM/ukdale_training/dishwasher_house_2_training_.csv"
# # X4,Y4=dataProvider4(trainfile2, windowsize=19)
# # x_train_all4, x_predict4, y_train_all4, y_predict4 = train_test_split(X, Y, test_size=0.2, random_state=100)
# # del X, Y
# X4,Y4=dataProvider2(trainfile1,trainfile2, windowsize=19)
# x_train_all4, x_predict4, y_train_all4, y_predict4 = train_test_split(X4, Y4, test_size=0.2, random_state=100)
# Y4=y_predict4
# X4=x_predict4
# model= 'F:/NILM/model/transfer/model_microwave.pkl'
# with open(model, 'rb+') as f:
#     lgbm4= pickle.load(f)
# prediction4=lgbm4.predict(X4)
# dif4=[]
# error=[]
# squarey=[]
# for i in range(len(Y4)):
#     value=prediction4[i]-Y4[i]
#     dif4.append(abs(value))
#     error.append(value*value)
#     squarey.append(Y4[i]*Y4[i])
# de4=sqrt(sum(error)/sum(squarey))
# print("microwave de1 is :", de4)
#
#
# first=sum(dif1)+sum(dif2)+sum(dif3)+sum(dif4)
# second=2*(np.abs(Y1).sum()+np.abs(Y2).sum()+np.abs(Y3).sum()+np.abs(Y4).sum())
# EAcc=1-first/second
# print("LightGBM Transfer EAcc is", EAcc)








