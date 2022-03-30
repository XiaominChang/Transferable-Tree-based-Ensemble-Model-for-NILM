import numpy as np
import pandas as pd
import lightgbm as lgb
from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK,STATUS_FAIL
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, zero_one_loss,mean_absolute_error,r2_score
import matplotlib.pyplot as plt
import time
from math import sqrt
import os
from tensorflow import keras
from keras_layer_normalization import LayerNormalization
# from loss import LossHistory
import math
import tensorflow.compat.v1 as tf
# testfile="F:/NILM/training_data/fridge_test_.csv"
# trainfile1="F:/NILM/ukdale_training/fridge_house_1_training_.csv"
# trainfile2="F:/NILM/ukdale_training/fridge_house_2_training_.csv"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
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


def get_sae(target, prediction):
    r = target.sum()
    r0 = prediction.sum()
    sae = abs(r0 - r) / r
    return sae


trainfile1 = "F:/NILM/training_data/dishwasher_house_2_training_.csv"
trainfile2 = "F:/NILM/training_data/dishwasher_house_3_training_.csv"
testfile="F:/NILM/training_data/dishwasher_test_.csv"
X1,Y1=dataProvider(testfile,trainfile1,trainfile2 ,19)
Y1=Y1.reshape(-1,1)
x_train_all1, x_predict1, y_train_all1, y_predict1 = train_test_split(X1, Y1, test_size=0.2, random_state=100)


trainfile1 = "F:/NILM/training_data/fridge_house_2_training_.csv"
trainfile2 = "F:/NILM/training_data/fridge_house_3_training_.csv"
testfile="F:/NILM/training_data/fridge_test_.csv"
X2,Y2=dataProvider(testfile,trainfile1,trainfile2 ,19)
Y2=Y2.reshape(-1,1)
x_train_all2, x_predict2, y_train_all2, y_predict2 = train_test_split(X2, Y2, test_size=0.2, random_state=100)


trainfile1 = "F:/NILM/training_data/washingmachine_house_2_training_.csv"
trainfile2 = "F:/NILM/training_data/washingmachine_house_3_training_.csv"
testfile="F:/NILM/training_data/washingmachine_test_.csv"
X3,Y3=dataProvider(testfile,trainfile1,trainfile2 ,19)
Y3=Y3.reshape(-1,1)
x_train_all3, x_predict3, y_train_all3, y_predict3 = train_test_split(X3, Y3, test_size=0.2, random_state=100)


trainfile1 = "F:/NILM/training_data/microwave_house_2_training_.csv"
trainfile2 = "F:/NILM/training_data/microwave_house_3_training_.csv"
testfile="F:/NILM/training_data/microwave_test_.csv"
X4,Y4=dataProvider(testfile,trainfile1,trainfile2 ,19)
Y4=Y4.reshape(-1,1)
x_train_all4, x_predict4, y_train_all4, y_predict4 = train_test_split(X4, Y4, test_size=0.2, random_state=100)

model1=keras.models.load_model('F:/NILM/CNN/ukdale/dishwasher.h5')
for layer in model1.layers:
    print(layer.name, "is trainable?", layer.trainable)
model1.layers[0].trainable= False
model1.layers[1].trainable= False
model1.layers[2].trainable= False
model1.layers[3].trainable= False
model1.layers[4].trainable= True
model1.layers[5].trainable= True
model1.layers[6].trainable= True


model2=keras.models.load_model('F:/NILM/CNN/ukdale/fridge.h5')
for layer in model2.layers:
    print(layer.name, "is trainable?", layer.trainable)
model2.layers[0].trainable= False
model2.layers[1].trainable= False
model2.layers[2].trainable= False
model2.layers[3].trainable= False
model2.layers[4].trainable= True
model2.layers[5].trainable= True
model2.layers[6].trainable= True

model3=keras.models.load_model('F:/NILM/CNN/ukdale/washingmachine.h5')
for layer in model3.layers:
    print(layer.name, "is trainable?", layer.trainable)
model3.layers[0].trainable= False
model3.layers[1].trainable= False
model3.layers[2].trainable= False
model3.layers[3].trainable= False
model3.layers[4].trainable= True
model3.layers[5].trainable= True
model3.layers[6].trainable= True

model4=keras.models.load_model('F:/NILM/CNN/ukdale/microwave.h5')
for layer in model4.layers:
    print(layer.name, "is trainable?", layer.trainable)
model4.layers[0].trainable= False
model4.layers[1].trainable= False
model4.layers[2].trainable= False
model4.layers[3].trainable= False
model4.layers[4].trainable= True
model4.layers[5].trainable= True
model4.layers[6].trainable= True

model1.fit(x_train_all1, y_train_all1, epochs=50, batch_size=1000, validation_split=0.2)
model2.fit(x_train_all2, y_train_all2, epochs=50, batch_size=1000, validation_split=0.2)
model3.fit(x_train_all3, y_train_all3, epochs=50, batch_size=1000, validation_split=0.2)
model4.fit(x_train_all4, y_train_all4, epochs=50, batch_size=1000, validation_split=0.2)

prediction1=model1.predict(x_predict1)
prediction2=model2.predict(x_predict2)
prediction3=model3.predict(x_predict3)
prediction4=model4.predict(x_predict4)
mae1=mean_absolute_error(y_predict1, prediction1)
mae2=mean_absolute_error(y_predict2, prediction2)
mae3=mean_absolute_error(y_predict3, prediction3)
mae4=mean_absolute_error(y_predict4, prediction4)
print("dishwasher's MAE", mae1)
print("fridge's MAE", mae2)
print("washingmachine's MAE", mae3)
print("microwave's MAE", mae4)
predict1=model1.predict(X1)
predict2=model2.predict(X2)
predict3=model3.predict(X3)
predict4=model4.predict(X4)


dif1=[]
error=[]
squarey=[]
for i in range(len(y_predict1)):
    value=prediction1[i]-y_predict1[i]
    dif1.append(abs(value))
    error.append(value*value)
    squarey.append(y_predict1[i]*y_predict1[i])
de1=sqrt(sum(error)/sum(squarey))
print("dishwasher de1 is :", de1)
sae1=get_sae(y_predict1,prediction1)
print("SAE is", sae1)
sum_predict1=predict1.sum()
print("sum consump", sum_predict1)



dif2=[]
error=[]
squarey=[]
for i in range(len(y_predict2)):
    value=prediction2[i]-y_predict2[i]
    dif2.append(abs(value))
    error.append(value*value)
    squarey.append(y_predict2[i]*y_predict2[i])
de1=sqrt(sum(error)/sum(squarey))
print("fridge de1 is :", de1)
sae2=get_sae(y_predict2,prediction2)
print("SAE is", sae2)
sum_predict2=predict2.sum()
print("sum consump", sum_predict2)



dif3=[]
error=[]
squarey=[]
for i in range(len(y_predict3)):
    value=prediction3[i]-y_predict3[i]
    dif3.append(abs(value))
    error.append(value*value)
    squarey.append(y_predict3[i]*y_predict3[i])
de1=sqrt(sum(error)/sum(squarey))
print("washingmachine de1 is :", de1)
sae3=get_sae(y_predict3,prediction3)
print("SAE is", sae3)
sum_predict3=predict3.sum()
print("sum consump", sum_predict3)


dif4=[]
error=[]
squarey=[]
for i in range(len(y_predict1)):
    value=prediction4[i]-y_predict4[i]
    dif4.append(abs(value))
    error.append(value*value)
    squarey.append(y_predict4[i]*y_predict4[i])
de1=sqrt(sum(error)/sum(squarey))
print("microwave de1 is :", de1)
sae4=get_sae(y_predict4,prediction4)
print("SAE is", sae4)
sum_predict4=predict4.sum()
print("sum consump", sum_predict4)


first=sum(dif1)+sum(dif2)+sum(dif3)+sum(dif4)
second=2*(np.abs(y_predict1).sum()+np.abs(y_predict2).sum()+np.abs(y_predict3).sum()+np.abs(y_predict4).sum())
EAcc=1-first/second
print("Transferred CNN EAcc is", EAcc)

sum_predict=sum_predict1+sum_predict2+sum_predict3+sum_predict4
print("washingmachine rate is: ", sum_predict3/sum_predict)
print("fridge rate is: ", sum_predict2/sum_predict)
print("dishwasher rate is: ", sum_predict1/sum_predict)
print("microwave rate is: ", sum_predict4/sum_predict)