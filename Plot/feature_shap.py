import shap
import numpy as np
import pandas as pd
import lightgbm as lgb
from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, zero_one_loss,mean_absolute_error,r2_score
import matplotlib.pyplot as plt
import time
from math import sqrt
import os
from matplotlib.backends.backend_pdf import PdfPages


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

# def dataProvider2(train1, train2, windowsize):
#     offset = int(0.5 * (windowsize - 1.0))
#     data_frame1 = pd.read_csv(train1,
#                              #chunksize=10 ** 3,
#                              header=0
#                              )
#     data_frame2 = pd.read_csv(train2,
#                              #chunksize=10 ** 3,
#                              header=0
#                              )
#     # data_frame3 = pd.read_csv(train3,
#     #                          #chunksize=10 ** 3,
#     #                          header=0
#     #                          )
#
#     np_array = np.array(data_frame1,dtype=np.float16)
#     inputs, targets = np_array[:, 0], np_array[:, 1]
#     window_num=inputs.size - 2 * offset
#     features=list()
#     labels=list()
#     for i in range(0,window_num):
#         inp=inputs[i:i+windowsize]
#         tar=targets[i+offset]
#         features.append(inp)
#         labels.append(tar)
#     features0=np.array(features,dtype=np.float16)
#     labels0=np.array(labels,dtype=np.float16)
#
#     np_array = np.array(data_frame2,dtype=np.float16)
#     inputs, targets = np_array[:, 0], np_array[:, 1]
#     window_num=inputs.size - 2 * offset
#     features=list()
#     labels=list()
#     for i in range(0,window_num):
#         inp=inputs[i:i+windowsize]
#         tar=targets[i+offset]
#         features.append(inp)
#         labels.append(tar)
#     features1=np.array(features,dtype=np.float16)
#     labels1=np.array(labels,dtype=np.float16)
#
#     # np_array = np.array(data_frame3)
#     # inputs, targets = np_array[:, 0], np_array[:, 1]
#     # window_num=inputs.size - 2 * offset
#     # features=list()
#     # labels=list()
#     # for i in range(0,window_num):
#     #     inp=inputs[i:i+windowsize]
#     #     tar=targets[i+offset]
#     #     features.append(inp)
#     #     labels.append(tar)
#     # features2=np.array(features)
#     # labels2=np.array(labels)
#
#     feature=np.concatenate((features0, features1), axis=0)
#     label=np.concatenate((labels0, labels1), axis=0)
#     return feature, label
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
X, Y = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)
#X, Y = dataProvider2(trainfile1, trainfile2,  windowsize=19)
#X, Y = dataProvider4(trainfile1,   windowsize=19)
x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, Y, test_size=0.2, random_state=100)
#model=lgb.Booster(model_file="/NILM/model/redd/lightGBM_microwaveN1.txt")
model=lgb.Booster(model_file="/NILM/model/ukdale/lightGBM_washingmachineN1.txt")
model.params['objective'] = 'regression'
def get_Mae(tranformer, x_predict, y_predict):
    gbm = tranformer
    prediction = gbm.predict(x_predict)
    return mean_absolute_error(y_predict, prediction)

def get_Sae(tranformer, x_predict, y_predict):
    gbm = tranformer
    prediction = gbm.predict(x_predict)
    r=y_predict.sum()
    print(r)
    r0=prediction.sum()
    print(r0)
    return abs(r0-r)/r

# print("MAE is:", get_Mae(model, x_predict, y_predict))
# print("SAE is:", get_Sae(model, x_predict, y_predict))

#x_summary = shap.kmeans(X, 10)
#shap_value = shap.TreeExplainer(model,x_summary,feature_perturbation='tree_path_dependent').shap_values(X)
shap_value = shap.TreeExplainer(model,feature_perturbation='tree_path_dependent').shap_values(X)
shap= np.abs(shap_value).mean(0)
print(shap)
def sortShap(arr, thresh):
    sum=arr.sum()
    sum0=0
    list=[]
    indexlist=np.argsort(-arr)
    print(indexlist)
    for i in indexlist:
        sum0=sum0+arr[i]
        if (sum0/sum)<=thresh:
            list.append(i)
        else:
            break
    return list
print(sortShap(shap))
# shap.summary_plot(shap_value, X, show=False)
# # pdf = PdfPages('F:/NILM/model/ukdale/microwave.pdf')
# # fig = plt.gcf()
# # fig.set_size_inches(150, 100)
# # pdf.savefig(fig)
# # pdf.close()
# # plt.savefig("/NILM/model/ukdale/fridge_ukdale.pdf", format='pdf', dpi=1000, bbox_inches='tight')
# plt.savefig("/NILM/model/redd/washingmachine_redd.pdf", format='pdf', dpi=1000, bbox_inches='tight')
# plt.close()


