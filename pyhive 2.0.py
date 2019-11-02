from PIL import Image
import os
import numpy as np
from functools import partial
import multiprocessing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sub_modules import my_hog
from sub_modules import my_lbp
from sub_modules import my_glcm
from sub_modules import my_hessian
from sub_modules import my_pca
from sub_modules import my_canny
from sub_modules import my_daisy
from sub_modules import my_structure
from sub_modules import my_eigval
from sub_modules import my_corner
from sub_modules import my_kitchen
from sub_modules import my_triz

from matplotlib import pyplot as plt
from io import BytesIO

import warnings
warnings.filterwarnings("ignore")


try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser

try:
    import cPickle as pickle
except ImportError:
    import pickle




class ImageProcess(object):
    """
    the main class of pyHIVES    
    """
    def select_estimator(self, case):

        if case == 0:
            print("SVM")
            estimator = SVC()
        elif case == 1:
            print("RF")
            estimator = RandomForestClassifier(random_state = 7)
        elif case == 2:
            print("DTree")
            estimator = DecisionTreeClassifier(random_state = 7)
        elif case == 3:
            print("NBayes")
            estimator = GaussianNB()
        elif case == 4:
            print("LR")
            estimator = LogisticRegression()   
        elif case == 5:
            print("KN")
            estimator = KNeighborsClassifier()     
        return estimator

    def evaluate(self,estimator,X,y,skf):
        acc_list,sn_list,sp_list,mcc_list = [],[],[],[]
        for train_index, test_index in skf.split(X, y):
            estimator.fit(X[train_index],y[train_index])
            y_predict = estimator.predict(X[test_index])
            y_true = y[test_index]

            #索引
            predict_index_p = (y_predict == 1)  #预测为正类的
            predict_index_n = (y_predict == 0)  #预测为负类

            index_p = (y_true==1)  #实际为正类
            index_n = (y_true==0)  #实际为负类

            Tp = sum(y_true[predict_index_p])       #正确预测的正类  （实际为正类 预测为正类）
            Tn = sum([1 for x in list(y_true[predict_index_n]) if x == 0]) #正确预测的负类   (实际为负类 预测为负类)
            Fn = sum(y_predict[index_n])       #错误预测的负类  （实际为负类 预测为正类）
            Fp = sum(y_true[predict_index_n])       #错误预测的正类   (实际为正类 预测为负类)

            try:
                acc = (Tp+Tn)/(Tp+Tn+Fp+Fn)
            except:
                acc = 0
            try:    
                sn = Tp/(Tp+Fn)
            except:
                sn = 0
            try:    
                sp = Tn/(Tn+Fp)
            except:
                sp = 0

            try:        
                mcc = matthews_corrcoef(y_true,y_predict)
            except:
                mcc = 0    

            acc_list.append(acc)
            sn_list.append(sn)
            sp_list.append(sp)
            mcc_list.append(mcc)

        return np.mean(acc_list),np.mean(sn_list),np.mean(sp_list),np.mean(mcc_list),Tp,Tn,Fn,Fp
        
    def run(self, seed):
        """ high-level function to run the entire class.
        """ 

        df = pd.read_csv('D:/school_stuff/HILab/pyHIVE-1-0-8/b.csv', header=0)
        df = df.drop(columns = ['a'])
        dataset = df.values
        print(dataset.shape)
        # print(dataset.shape)   #"[sample,feature]"  
        #生成类标
        labels = []
        n = 0
        for file in os.listdir("normal"):
            n = n + 1
        for i in range(dataset.shape[0] - n):
            labels.append(0)

        for i in range(n):
            labels.append(1)
        labels = np.array(labels)
        print(dataset.shape[0] - n)
        estimator_list = [0,1,2,3,4,5]
        skf = RepeatedKFold(n_repeats=20,n_splits= 10,random_state = 7)

        for i in estimator_list:    
            acc,sn,sp,mcc,Tp,Tn,Fn,Fp = self.evaluate(self.select_estimator(i),dataset,labels,skf)
            print("Acc: ",acc)
            print("Sn: ",sn)
            print("Sp: ",sp)
            print("Mcc: ",mcc)
            
            print(Tp,Tn,Fn,Fp)
            
    def twenty_seed(self):
        for seed in range(1):
            print(seed)
            self.run(seed)

if __name__ == '__main__':
    processor = ImageProcess()
    processor.twenty_seed()
