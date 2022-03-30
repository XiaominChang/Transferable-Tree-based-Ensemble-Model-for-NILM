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
class loss_man:
    loss_global=100000
    dtrain_global = None
    dtest_global = None
    model=None
    x_predict=None
    y_predict=None
    time_step=None
    def set_loss(self,loss):
       self.loss_global=loss
    def set_model(self,model):
       self.model=model
    def setData(self,dtrain,dtest):
        self.dtrain_global=dtrain
        self.dtest_global=dtest
    def setPredict(self, x_predict, y_predict):
        self.x_predict=x_predict
        self.y_predict=y_predict
    def setTime(self, time_step):
        self.time_step=time_step

loss_glo=loss_man()

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


    np_array = np.array(data_frame1,dtype=np.float16)
    inputs, targets = np_array[:, 0], np_array[:, 1]
    window_num=inputs.size - 2 * offset
    features=list()
    labels=list()
    for i in range(0,window_num):
        inp=inputs[i:i+windowsize]
        tar=targets[i+offset]
        features.append(inp)
        labels.append(tar)
    features0=np.array(features,dtype=np.float16)
    labels0=np.array(labels,dtype=np.float16)

    np_array = np.array(data_frame2,dtype=np.float16)
    inputs, targets = np_array[:, 0], np_array[:, 1]
    window_num=inputs.size - 2 * offset
    features=list()
    labels=list()
    for i in range(0,window_num):
        inp=inputs[i:i+windowsize]
        tar=targets[i+offset]
        features.append(inp)
        labels.append(tar)
    features1=np.array(features,dtype=np.float16)
    labels1=np.array(labels,dtype=np.float16)
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

    np_array = np.array(data_frame1,dtype=np.float16)
    inputs, targets = np_array[:, 0], np_array[:, 1]
    window_num=inputs.size - 2 * offset
    features=list()
    labels=list()
    for i in range(0,window_num):
        inp=inputs[i:i+windowsize]
        tar=targets[i+offset]
        features.append(inp)
        labels.append(tar)
    features0=np.array(features,dtype=np.float16)
    labels0=np.array(labels,dtype=np.float16)
    return features0, labels0



space = {'num_leaves': hp.randint("num_leaves", 1000),
         'min_data_in_leaf': hp.randint("min_data_in_leaf", 100),
         'max_depth': hp.randint("max_depth", 20),
         'learning_rate':  hp.uniform('learning_rate', 0, 0.5),
         'feature_fraction': hp.uniform("feature_fraction", 0,1),
         #'bagging_fraction': hp.uniform("bagging_fraction", 0,1),
         'max_bin':hp.randint('max_bin', 2000),
         'num_boost_round': hp.randint('num_boost_round', 200),
         #'bagging_freq': hp.randint('bagging_freq', 10),
         'min_data_in_bin':hp.randint('min_data_in_bin', 500),
         'lambda_l2': hp.uniform("lambda_l2", 0,1),
         'lambda_l1': hp.uniform("lambda_l1", 0, 0.004),
         'bin_construct_sample_cnt': hp.randint('bin_construct_sample_cnt', 1000000),
         "time_step": hp.randint("time_step", 14),
         "output": hp.randint("output", 200),
         'window': hp.randint("window", 600)
}


def argsDict_tranform(argsDict, isPrint=False):
    argsDict["max_depth"] = argsDict["max_depth"] + 1
    argsDict['min_data_in_leaf'] = argsDict['min_data_in_leaf'] + 500
    argsDict["num_leaves"] = argsDict["num_leaves"] + 100
    argsDict["learning_rate"] = argsDict["learning_rate"] + 0.05
    argsDict["lambda_l1"] = argsDict["lambda_l1"] + 0.0005
    argsDict['max_bin'] = argsDict['max_bin'] + 200
    argsDict['num_boost_round'] = argsDict['num_boost_round'] + 100
    #argsDict['bagging_freq'] = argsDict['bagging_freq'] + 1
    argsDict['min_data_in_bin'] = argsDict['min_data_in_bin'] + 20
    argsDict['bin_construct_sample_cnt']=argsDict['bin_construct_sample_cnt']+200000
    argsDict['time_step']=argsDict['time_step']+1
    argsDict['output']=argsDict["output"]+32
    if argsDict['window']%2==0:
        argsDict['window']=argsDict['window']+3
    return argsDict


trainfile1 = "/training_data/fridge_house_2_training_.csv"
trainfile2 = "training_data/fridge_house_3_training_.csv"
testfile="F:/NILM/training_data/fridge_test_.csv"
X, Y = dataProvider(testfile, trainfile1, trainfile2, windowsize=19)
print(X.shape)
print(Y.shape)
x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, Y, test_size=0.2, random_state=100)
#x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)
del X,Y
dtrainv1 = lgb.Dataset(x_train_all, y_train_all)
# dvalidv1= lgb.Dataset(x_test,y_test)
dvalidv1 = dtrainv1.create_valid(x_train_all, y_train_all)
os.remove('/NILM/bins/train_v1.bin')
dtrainv1.save_binary('/NILM/bins/train_v1.bin')
os.remove('/NILM/bins/valid_v1.bin')
dvalidv1.save_binary('/NILM/bins/valid_v1.bin')


def lgb_factory(argsDict):
    argsDict = argsDict_tranform(argsDict)
    dtrain = lgb.Dataset('F:/NILM/bins/train_v1.bin')
    dvalid = lgb.Dataset('F:/NILM/bins/valid_v1.bin')

    params = {
        'boosting_type': 'goss',
        'objective': 'regression',
        'metric': {'l1', 'l2'},
        'num_leaves': argsDict['num_leaves'],
        # 'num_leaves': 2,
        'min_data_in_leaf': argsDict['min_data_in_leaf'],
        'max_depth': argsDict['max_depth'],
        'learning_rate': argsDict['learning_rate'],
        # 'feature_fraction':argsDict['feature_fraction'],
        # 'bagging_fraction': argsDict['bagging_fraction'],
        # 'bagging_freq': argsDict['bagging_freq'],
        'max_bin': argsDict['max_bin'],
        # 'num_boost_round':argsDict['num_boost_rount'] ,
        'min_data_in_bin': argsDict['min_data_in_bin'],
        'lambda_l2': argsDict['lambda_l2'],
        'verbose': -1,
        'is_provide_training_metric': True,
        'bin_construct_sample_cnt': argsDict['bin_construct_sample_cnt']
    }

    gbm = lgb.train(params, dtrain, num_boost_round=100, valid_sets=dvalid, early_stopping_rounds=10)
    loss=get_tranformer_score(gbm, x_predict, y_predict)
    if(loss<loss_glo.loss_global):
        loss_glo.set_loss(loss)
        #loss_glo.set_model(model)
        loss_glo.setData(dtrain, dvalid)
        loss_glo.setPredict(x_predict, y_predict)
    return {'loss': loss, 'status': STATUS_OK}

def get_tranformer_score(tranformer, x_predict, y_predict):
    gbm = tranformer
    prediction = gbm.predict(x_predict)
    return mean_absolute_error(y_predict, prediction)
    #return np.sqrt(mean_squared_error(y_predict, prediction))


def lgbbest_train(argsDict):
    argsDict = argsDict_tranform(argsDict)
    dtrain = loss_glo.dtrain_global
    dvalid = loss_glo.dtest_global
    xtest=loss_glo.x_predict
    ytest=loss_glo.y_predict
    params = {
        'boosting_type': 'goss',
        'objective': 'regression',
        'metric': {'l1', 'l2'},
        'num_leaves': argsDict['num_leaves'],
        # 'num_leaves': 2,
        'min_data_in_leaf': argsDict['min_data_in_leaf'],
        'max_depth': argsDict['max_depth'],
        'learning_rate': argsDict['learning_rate'],
        # 'feature_fraction':argsDict['feature_fraction'],
        # 'bagging_fraction': argsDict['bagging_fraction'],
        # 'bagging_freq': argsDict['bagging_freq'],
        'max_bin': argsDict['max_bin'],
        # 'num_boost_round':argsDict['num_boost_rount'] ,
        'min_data_in_bin': argsDict['min_data_in_bin'],
        'lambda_l2': argsDict['lambda_l2'],
        'verbose': -1,
        'is_provide_training_metric': True,
        'bin_construct_sample_cnt': argsDict['bin_construct_sample_cnt']
    }

    evals_result={}
    gbm = lgb.train(params, dtrain, num_boost_round=100, valid_sets=[dtrain,dvalid], valid_names=["training","validation"], evals_result=evals_result, early_stopping_rounds=10)
    # gbm.save_model('F:/NILM/model/redd/lightGBM_fridgeN1.txt')
    loss=get_tranformer_score(gbm,xtest,ytest)
    print(type(gbm))
    print(evals_result)
    lgb.plot_metric(evals_result, metric='l2', ylabel="MSE")
    plt.legend( prop={'size': 16})
    plt.show()
    # modelpd= gbm.trees_to_dataframe()
    # modelpd.to_csv("C:/Users/chang/Desktop/treeStructure/redd/treeStructure_fridge.csv",header=True, index=False)
    return {'loss': loss, 'status': STATUS_OK}
    #return gbm
time_start=time.time()
trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=20)
best = fmin(lgb_factory, space, algo=algo, max_evals=2, pass_expr_memo_ctrl=None, trials=trials)


lgbbest_train(best)
time_end=time.time()
print('training cost is: ', time_end-time_start)
print('best :', best)
# print('best param after transform :')
# print(argsDict_tranform(best,isPrint=True))
# print('\nrmse of the best xgboost:', gbm['loss'])

