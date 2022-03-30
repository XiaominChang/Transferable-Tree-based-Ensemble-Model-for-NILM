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
print(tf.__version__)
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
# feature,label=dataProvider(testfile, trainfile1, trainfile2)

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
def dataProvider4(train1,  windowsize):
    offset = int(0.5 * (windowsize - 1.0))
    data_frame1 = pd.read_csv(train1,
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
    return features0, labels0

space = {"layer1_output": hp.randint("layer1_output", 200),
         "layer2_output": hp.randint("layer2_output", 200),
         "layer1_dropout": hp.uniform("layer1_dropout", 0, 1),
         "layer2_dropout": hp.uniform("layer2_dropout", 0, 1),
         "layer1_rdropout": hp.uniform('layer1_rdropout', 0, 1),
         "layer2_rdropout": hp.uniform('layer2_rdropout', 0, 1),
         "layer3_dropout": hp.uniform('layer3_dropout', 0, 1),
         #"optimizer": hp.choice('optimizer', ['adam', 'sgd']),
         "momentum": hp.uniform('momentum', 0,1),
         "lr": hp.uniform('lr', 1e-9, 1e-3),
         "decay": hp.uniform('decay', 1e-9, 1e-3),
         'epochs': hp.randint('epochs', 250),
         'batch_size': hp.randint('batch_size', 100),
         'time_step':hp.randint('time_step',13)
         }

def argsDict_tranform(argsDict):
    argsDict["layer1_output"] = argsDict["layer1_output"] + 20
    argsDict['layer2_output'] = argsDict['layer2_output'] + 20
    argsDict['epochs'] = argsDict['epochs'] + 50
    argsDict['batch_size'] = argsDict['batch_size'] + 32
    argsDict['time_step']=argsDict['time_step']+1
    return argsDict
windowsize=19
path="F:/NILM/refit_training/fridge"
trainfile1 = "F:/NILM/training_data/fridge_house_2_training_.csv"
trainfile2 = "F:/NILM/training_data/fridge_house_3_training_.csv"
#validfile="F:/NILM/datafortrain/dishwasher_validation_.csv"
testfile="F:/NILM/training_data/fridge_test_.csv"
#

#X,Y= dataProvider3(path, 159)
X, Y = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)
# path="F:/NILM/refit_training/washingmachine"
# X,Y= dataProvider3(path, 19)
# print(X.shape)
# print(Y.shape)
# x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, Y, test_size=0.2, random_state=100)
# del X,Y
# x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)

# def CNN_training(argsDic):
#     argsDic=argsDict_tranform(argsDic)
#     model=keras.models.Sequential()
#     # model.add(LayerNormalization())
#     model.add(keras.layers.Reshape((-1,windowsize, 1), input_shape=(19,)))
#     model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.Conv2D(filters=30,
#                   kernel_size=(10, 1),
#                   strides=(1, 1),
#                   padding='same',
#                   activation='relu',
#                   ))
#     model.add(keras.layers.Conv2D(filters=30,
#                   kernel_size=(8, 1),
#                   strides=(1, 1),
#                   padding='same',
#                   activation='relu',
#                   ))
#     model.add(keras.layers.Conv2D(filters=40,
#                   kernel_size=(6, 1),
#                   strides=(1, 1),
#                   padding='same',
#                   activation='relu',
#                   ))
#     model.add(keras.layers.Conv2D(filters=50,
#                   kernel_size=(5, 1),
#                   strides=(1, 1),
#                   padding='same',
#                   activation='relu',
#                   ))
#     model.add(keras.layers.Conv2D(filters=50,
#                   kernel_size=(5, 1),
#                   strides=(1, 1),
#                   padding='same',
#                   activation='relu',
#                   ))
#     #model.add(LayerNormalization())
#     model.add(keras.layers.Flatten(name='flatten'))
#     # model.add(keras.layers.Dropout(argsDic['layer3_dropout']))
#     #model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.Dense(1024, activation='relu', name='dense'))
#     #model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.Dense(1, activation='linear', name='output'))
#     adam = keras.optimizers.Adam(learning_rate=0.001,
#                                   beta_1=0.9,
#                                   beta_2=0.999,
#                                   epsilon=1e-08)
#                                   # use_locking=False)
#     model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])
#     print('start training')
#     model.fit(x_train_all, y_train_all, epochs=50, batch_size=10000, validation_split=0.2)
#     loss=get_tranformer_score(model, x_predict, y_predict)
#     if(loss==10):
#         return {'loss':loss, 'status':STATUS_FAIL}
#     #model.save('/home/xcha8737/Solar_Forecast/trainning_data/SolarPrediction.csv/GRU.h5')
#     else:
#         return {'loss':loss, 'status':STATUS_OK}

def get_tranformer_score(tranformer,x_predict, y_predict):
    gru = tranformer
    prediction = gru.predict(x_predict)
    # for i in prediction:
    #     if math.isnan(i[0]):
    #         print('nan number is found')
    #         return 10
    r = y_predict.sum()
    r0 = prediction.sum()
    sae=abs(r0 - r) / r
    #print("the new model sae is :", abs(r0 - r) / r)
    return mean_absolute_error(y_predict, prediction), sae

def CNN_training_best(X, Y, num):
    x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, Y, test_size=0.2, random_state=100)
    del X, Y
    #argsDic=argsDict_tranform(argsDic)
    model=keras.models.Sequential()
    # model.add(LayerNormalization())
    model.add(keras.layers.Reshape((-1, windowsize, 1),input_shape=(19,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(filters=30,
                  kernel_size=(10, 1),
                  strides=(1, 1),
                  padding='same',
                  activation='relu',
                  ))
    model.add(keras.layers.Conv2D(filters=30,
                  kernel_size=(8, 1),
                  strides=(1, 1),
                  padding='same',
                  activation='relu',
                  ))
    model.add(keras.layers.Conv2D(filters=40,
                  kernel_size=(6, 1),
                  strides=(1, 1),
                  padding='same',
                  activation='relu',
                  ))
    model.add(keras.layers.Conv2D(filters=50,
                  kernel_size=(5, 1),
                  strides=(1, 1),
                  padding='same',
                  activation='relu',
                  ))
    model.add(keras.layers.Conv2D(filters=50,
                  kernel_size=(5, 1),
                  strides=(1, 1),
                  padding='same',
                  activation='relu',
                  ))
    #model.add(LayerNormalization())
    model.add(keras.layers.Flatten(name='flatten'))
    # model.add(keras.layers.Dropout(argsDic['layer3_dropout']))
    #model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1024, activation='relu', name='dense'))
    #model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1, activation='linear', name='output'))
    adam = keras.optimizers.Adam(learning_rate=0.001,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  epsilon=1e-08)
                                  # use_locking=False)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])
    print('start training')
    model.fit(x_train_all,y_train_all, epochs=50, batch_size=1000, validation_split=0.2)
    loss,sae=get_tranformer_score(model, x_predict, y_predict)
    # model.save('F:/NILM/CNN/redd/'+str(num)+'.h5')
    # time_start=time.time()
    # result=model.predict(x_predict)
    # time_end=time.time()
    # result = result * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    # y_real = y_predict * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    # print('totally cost', time_end - time_start)
    # print("rmse is ：", sqrt(mean_squared_error(y_real, result)))
    # print("mae is ：", mean_absolute_error(y_real, result))
    # print('r2 is :', r2_score(y_real, result))
    return model
    # if(loss==10):
    #     return {'loss':loss, 'status':STATUS_FAIL}
    # #model.save('/home/xcha8737/Solar_Forecast/trainning_data/SolarPrediction.csv/GRU.h5')
    # else:
    #     return {'loss':loss, 'status':STATUS_OK}
def get_sae(target, prediction):
    # assert (target.shape == prediction.shape)

    r = target.sum()
    r0 = prediction.sum()
    sae = abs(r0 - r) / r
    print("targe sum is :", r)
    print("prediction sum is:", r0)
    return sae
# trainfile1="F:/NILM/ukdale_training/fridge_house_1_training_.csv"
# trainfile2="F:/NILM/ukdale_training/fridge_house_2_training_.csv"
# X, Y = dataProvider2(trainfile1, trainfile2,  windowsize=19)
# x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, Y, test_size=0.2, random_state=100)
# # x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)
# time_start=time.time()
model=CNN_training_best(X, Y, 19)
# time_end=time.time()
# print('training cost is: ', time_end-time_start)
#
# time_start=time.time()
# model.predict(x_predict)
# time_end=time.time()
# print('training cost is: ', time_end-time_start)
# x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, Y, test_size=0.2, random_state=100)
# x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)


# trainfile1 = "F:/NILM/training_data/fridge_house_2_training_.csv"
# trainfile2 = "F:/NILM/training_data/fridge_house_3_training_.csv"
# testfile="F:/NILM/training_data/fridge_test_.csv"
# X, Y = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)
# x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, Y, test_size=0.2, random_state=100)
# model=keras.models.load_model('F:/NILM/CNN/redd/2.h5')
# prediction = model.predict(x_predict)
# sae=get_sae(y_predict,prediction)
# print(sae)
# trials = Trials()
# algo = partial(tpe.suggest, n_startup_jobs=20)
# best = fmin(CNN_training, space, algo=algo, max_evals=1, pass_expr_memo_ctrl=None, trials=trials)
# trainfile1="F:/NILM/ukdale_training/washingmachine_house_1_training_.csv"
# trainfile2="F:/NILM/ukdale_training/washingmachine_house_2_training_.csv"
# X, Y = dataProvider2(trainfile1, trainfile2,  windowsize=19)
# time_start=time.time()
# mae1,sae1= CNN_training_best(X,Y,1)
# time_end=time.time()
# cost1=time_end-time_start
#
#
# trainfile1 = "F:/NILM/training_data/washingmachine_house_2_training_.csv"
# trainfile2 = "F:/NILM/training_data/washingmachine_house_3_training_.csv"
# testfile="F:/NILM/training_data/washingmachine_test_.csv"
# X, Y = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)
# time_start=time.time()
# mae1,sae1 = CNN_training_best(X,Y,1)
# time_end=time.time()
# cost1=time_end-time_start
#
#
#
# trainfile1 = "F:/NILM/training_data/fridge_house_2_training_.csv"
# trainfile2 = "F:/NILM/training_data/fridge_house_3_training_.csv"
# testfile="F:/NILM/training_data/fridge_test_.csv"
# X, Y = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)
# time_start=time.time()
# mae2,sae2 = CNN_training_best(X,Y,2)
# time_end=time.time()
# cost2=time_end-time_start
#
#
# trainfile1 = "F:/NILM/training_data/dishwasher_house_2_training_.csv"
# trainfile2 = "F:/NILM/training_data/dishwasher_house_3_training_.csv"
# testfile="F:/NILM/training_data/dishwasher_test_.csv"
# X, Y = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)
# time_start=time.time()
# mae3,sae3 = CNN_training_best(X,Y,3)
# time_end=time.time()
# cost3=time_end-time_start
#
#
#
# trainfile1 = "F:/NILM/training_data/microwave_house_2_training_.csv"
# trainfile2 = "F:/NILM/training_data/microwave_house_3_training_.csv"
# testfile="F:/NILM/training_data/microwave_test_.csv"
# X, Y = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)
# time_start=time.time()
# mae4,sae4 = CNN_training_best(X,Y,4)
# time_end=time.time()
# cost4=time_end-time_start



# print('training cost wash is: ', cost1)
# # # print('best :', best)
# # print('best param after transform :')
# # print(argsDict_tranform(best))
# print('\nMAE of the best gru:', mae1)
# print('\nSAE of the best gru:', sae1)
#
#
#
# print('training cost fridge is: ', cost2)
# # # print('best :', best)
# # print('best param after transform :')
# # print(argsDict_tranform(best))
# print('\nMAE of the best gru:', mae2)
# print('\nSAE of the best gru:', sae2)
#
#
# print('training cost dishwasher is: ', cost3)
# # # print('best :', best)
# # print('best param after transform :')
# # print(argsDict_tranform(best))
# print('\nMAE of the best gru:', mae3)
# print('\nSAE of the best gru:', sae3)
#
# print('training cost microwave is: ', cost4)
# # # print('best :', best)
# # print('best param after transform :')
# # print(argsDict_tranform(best))
# print('\nMAE of the best gru:', mae4)
# print('\nSAE of the best gru:', sae4)




####################TEST ON UK-DALE house-2################################################

# trainfile1="F:/NILM/ukdale_training/washingmachine_house_1_training_.csv"
# trainfile2="F:/NILM/ukdale_training/washingmachine_house_2_training_.csv"
# #X1,Y1=dataProvider4(trainfile2, windowsize=19)
# X1,Y1=dataProvider2(trainfile1,trainfile2, windowsize=19)
# x_train_all1, x_predict1, y_train_all1, y_predict1 = train_test_split(X1, Y1, test_size=0.2, random_state=100)
# # Y1=y_predict1
# # X1=x_predict1
# print(x_predict1.shape)
# model=keras.models.load_model('F:/NILM/CNN/ukdale/1.h5')
# # lgbm1=lgb.Booster(model_file="F:/NILM/model/ukdale/lightGBM_washingmachineN1.txt")
# # prediction1=lgbm1.predict(X1)
# time_start=time.time()
# prediction1 = model.predict(X1)
# time_end=time.time()
# print(time_end-time_start)
# #print("square is:",prediction1)
# # de1=np.sqrt(np.square(prediction1-Y1).sum()/np.square(Y1).sum())
#
#
# #print("mae is :", mean_absolute_error(Y1, prediction1))
# # r = y_predict.sum()
# # r0 = prediction.sum()
# # sae = abs(r0 - r) / r
# # print("washing machine sae is :", sae)
#
#
#
#
# trainfile1="F:/NILM/ukdale_training/microwave_house_1_training_.csv"
# trainfile2="F:/NILM/ukdale_training/microwave_house_2_training_.csv"
# # X4,Y4=dataProvider4(trainfile2, windowsize=19)
# # x_train_all2, x_predict2, y_train_all2, y_predict2 = train_test_split(X, Y, test_size=0.2, random_state=100)
# # del X, Y
# X2,Y2=dataProvider2(trainfile1,trainfile2, windowsize=19)
# #x_train_all2, x_predict2, y_train_all2, y_predict2 = train_test_split(X4, Y4, test_size=0.2, random_state=100)
# # Y2=y_predict2
# # X2=x_predict2
# model=keras.models.load_model('F:/NILM/CNN/ukdale/2.h5')
# # lgbm4=lgb.Booster(model_file="F:/NILM/model/ukdale/lightGBM_microwaveN1.txt")
# # prediction4=lgbm4.predict(X4)
# prediction2 = model.predict(X2)
# dif2=[]
# error=[]
# squarey=[]
# for i in range(len(Y2)):
#     value=prediction2[i]-Y2[i]
#     dif2.append(abs(value))
#     error.append(value*value)
#     squarey.append(Y2[i]*Y2[i])
# de2=sqrt(sum(error)/sum(squarey))
# print("microwave de1 is :", de2)
# # #print("mae is :", mean_absolute_error(Y2, prediction2))
# # # r = y_predict.sum()
# # # r0 = prediction.sum()
# # # sae = abs(r0 - r) / r
# # # print("microwave sae is :", sae)
# #
# trainfile1="F:/NILM/ukdale_training/fridge_house_1_training_.csv"
# trainfile2 = "F:/NILM/ukdale_training/fridge_house_2_training_.csv"
# X3,Y3=dataProvider2(trainfile1, trainfile2,windowsize=19)
# # x_train_all3, x_predict3, y_train_all3, y_predict3 = train_test_split(X, Y, test_size=0.2, random_state=100)
# # del X, Y
# x_train_all3, x_predict3, y_train_all3, y_predict3 = train_test_split(X3, Y3, test_size=0.2, random_state=100)
# # Y3=y_predict3
# # X3=x_predict3
# print(x_predict3.shape)
# model=keras.models.load_model('F:/NILM/CNN/ukdale/3.h5')
# # # lgbm2=lgb.Booster(model_file="F:/NILM/model/ukdale/lightGBM_fridgeN1.txt")
# # # prediction2=lgbm2.predict(X2)
# time_start=time.time()
# prediction3 = model.predict(x_predict3)
# time_end=time.time()
# print(time_end-time_start)
# dif3=[]
# error=[]
# squarey=[]
# for i in range(len(Y3)):
#     value=prediction3[i]-Y3[i]
#     dif3.append(abs(value))
#     error.append(value*value)
#     squarey.append(Y3[i]*Y3[i])
# de3=sqrt(sum(error)/sum(squarey))
# print("fridge de1 is :", de3)
# # #print("mae is :", mean_absolute_error(Y3, prediction3))
# # # r = y_predict.sum()
# # # r0 = prediction.sum()
# # # sae = abs(r0 - r) / r
# # # print("fridge sae is :", sae)
# #
# trainfile1="F:/NILM/ukdale_training/dishwasher_house_1_training_.csv"
# trainfile2="F:/NILM/ukdale_training/dishwasher_house_2_training_.csv"
# # X3,Y3=dataProvider4(trainfile2, windowsize=19)
# # x_train_all4, x_predict4, y_train_all4, y_predict4 = train_test_split(X, Y, test_size=0.2, random_state=100)
# # del X, Y
# X4,Y4=dataProvider2(trainfile1,trainfile2, windowsize=19)
# # x_train_all4, x_predict4, y_train_all4, y_predict4 = train_test_split(X3, Y3, test_size=0.2, random_state=100)
# # Y4=y_predict4
# # X4=x_predict4
# model=keras.models.load_model('F:/NILM/CNN/ukdale/dishwasher.h5')
# for layer in model.layers:
#     print(layer.name, "is trainable?", layer.trainable)
# model.layers[0].trainable= False
# model.layers[1].trainable= False
# model.layers[2].trainable= False
# model.layers[3].trainable= False
# model.layers[4].trainable= False
# model.layers[5].trainable= False
# model.layers[6].trainable= False

# # lgbm3=lgb.Booster(model_file="F:/NILM/model/ukdale/lightGBM_dishwasherN1.txt")
# # prediction3=lgbm3.predict(X3)
# prediction4 = model.predict(X4)
# dif4=[]
# error=[]
# squarey=[]
# for i in range(len(Y4)):
#     value=prediction4[i]-Y4[i]
#     dif4.append(abs(value))
#     error.append(value*value)
#     squarey.append(Y4[i]*Y4[i])
# de4=sqrt(sum(error)/sum(squarey))
# print("dishwasher de1 is :", de4)

# first=sum(dif1)+sum(dif2)+sum(dif3)+sum(dif4)
# second=2*(np.abs(Y1).sum()+np.abs(Y2).sum()+np.abs(Y3).sum()+np.abs(Y4).sum())
# print(first)
# print(second)
# EAcc=1-first/second
# print("CNN EAcc is", EAcc)


# lgbm1=lgb.Booster(model_file="F:/NILM/model/ukdale/lightGBM_washingmachineN1.txt")
# prediction1=lgbm1.predict(X1)
#
# lgbm2=lgb.Booster(model_file="F:/NILM/model/ukdale/lightGBM_fridgeN1.txt")
# prediction2=lgbm2.predict(X2)
#
# lgbm3=lgb.Booster(model_file="F:/NILM/model/ukdale/lightGBM_dishwasherN1.txt")
# prediction3=lgbm3.predict(X3)
# #
# lgbm4=lgb.Booster(model_file="F:/NILM/model/ukdale/lightGBM_microwaveN1.txt")
# prediction4=lgbm4.predict(X4)


#print("mae is :", mean_absolute_error(Y4, prediction4))
# r = y_predict.sum()
# r0 = prediction.sum()
# sae = abs(r0 - r) / r
# print("washing machine sae is :", sae)
#
#
# trainfile1="F:/NILM/ukdale_training/dishwasher_house_1_training_.csv"
# trainfile2="F:/NILM/ukdale_training/dishwasher_house_2_training_.csv"
# X, Y = dataProvider2(trainfile1, trainfile2,  windowsize=19)
# x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, Y, test_size=0.2, random_state=100)
# del X, Y
# model=keras.models.load_model('F:/NILM/CNN/redd/dishwasher.h5')
# prediction = model.predict(x_predict)
# r = y_predict.sum()
# r0 = prediction.sum()
# sae = abs(r0 - r) / r
# print("dishwasher sae is :", sae)


# trainfile1 = "F:/NILM/training_data/washingmachine_house_2_training_.csv"
# trainfile2 = "F:/NILM/training_data/washingmachine_house_3_training_.csv"
# testfile="F:/NILM/training_data/washingmachine_test_.csv"
# X1, Y1 = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)
# model=keras.models.load_model('F:/NILM/CNN/redd/1.h5')
# prediction1 = model.predict(X1)
# print("mae is:", mean_absolute_error(prediction1, Y1))
# # dif1=[]
# # error=[]
# # squarey=[]
# # for i in range(len(Y1)):
# #     value=prediction1[i]-Y1[i]
# #     dif1.append(abs(value))
# #     error.append(value*value)
# #     squarey.append(Y1[i]*Y1[i])
# # de1=sqrt(sum(error)/sum(squarey))
# # print("washing machine de1 is :", de1)
# #
# #
# #
# trainfile1 = "F:/NILM/training_data/fridge_house_2_training_.csv"
# trainfile2 = "F:/NILM/training_data/fridge_house_3_training_.csv"
# testfile="F:/NILM/training_data/fridge_test_.csv"
# X2, Y2 = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)
# model=keras.models.load_model('F:/NILM/CNN/redd/2.h5')
# prediction2 = model.predict(X2)
# # dif2=[]
# # error=[]
# # squarey=[]
# # for i in range(len(Y2)):
# #     value=prediction2[i]-Y2[i]
# #     dif2.append(abs(value))
# #     error.append(value*value)
# #     squarey.append(Y2[i]*Y2[i])
# # de2=sqrt(sum(error)/sum(squarey))
# # print("fridge de1 is :", de2)
# #
# trainfile1 = "F:/NILM/training_data/dishwasher_house_2_training_.csv"
# trainfile2 = "F:/NILM/training_data/dishwasher_house_3_training_.csv"
# testfile="F:/NILM/training_data/dishwasher_test_.csv"
# X3, Y3 = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)
# model=keras.models.load_model('F:/NILM/CNN/redd/3.h5')
# prediction3 = model.predict(X3)
# # dif3=[]
# # error=[]
# # squarey=[]
# # for i in range(len(Y3)):
# #     value=prediction3[i]-Y3[i]
# #     dif3.append(abs(value))
# #     error.append(value*value)
# #     squarey.append(Y3[i]*Y3[i])
# # de3=sqrt(sum(error)/sum(squarey))
# # print("dishwasher de1 is :", de3)
# #
# #
# trainfile1 = "F:/NILM/training_data/microwave_house_2_training_.csv"
# trainfile2 = "F:/NILM/training_data/microwave_house_3_training_.csv"
# testfile="F:/NILM/training_data/microwave_test_.csv"
# X4, Y4 = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)
# model=keras.models.load_model('F:/NILM/CNN/redd/4.h5')
# prediction4 = model.predict(X4)
# # dif4=[]
# # error=[]
# # squarey=[]
# # for i in range(len(Y4)):
# #     value=prediction4[i]-Y4[i]
# #     dif4.append(abs(value))
# #     error.append(value*value)
# #     squarey.append(Y4[i]*Y4[i])
# # de4=sqrt(sum(error)/sum(squarey))
# # print("microwave de1 is :", de4)
# #
# # first=sum(dif1)+sum(dif2)+sum(dif3)+sum(dif4)
# # second=2*(np.abs(Y1).sum()+np.abs(Y2).sum()+np.abs(Y3).sum()+np.abs(Y4).sum())
# # print(first)
# # print(second)
# # EAcc=1-first/second
# # print("CNN EAcc is", EAcc)


# labels = ['washingmachine', 'fridge', 'dishwasher', 'microwave']
# #X= [prediction1.sum(),prediction2.sum(),prediction3.sum(),prediction4.sum()]
# X= [Y1.sum(),Y2.sum(),Y3.sum(),Y4.sum()]
# fig = plt.figure()
# plt.pie(X, labels=labels, autopct='%1.2f%%')
# #plt.title("Ground Truth")
# plt.show()
