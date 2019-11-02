import pandas as pd
import csv
import sys
import os
import numpy as np
from PIL import Image
from functools import partial
import multiprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from matplotlib import pyplot as plt
'''
df=pd.read_csv('D:/school_stuff/HILab/pyHIVE-1-0-8/features/images_CORNER.csv', keep_default_na=False)
my_list = ['a']
for i in range(1,574):
    my_list.append( 'CORNER_' + str(i))
df.columns=my_list
df.to_csv('D:/school_stuff/HILab/pyHIVE-1-0-8/features/images_CORNER.csv',index = 0)
'''
'''
df = df.drop(columns = ['a'])
print(df.values)
'''
'''
df2=pd.read_csv('D:/school_stuff/HILab/pyHIVE-1-0-8/features/images_STRUCTURE.csv', header=0)
df3 = df1.drop(columns = ['a']).sample(frac = 0.5, axis = 1)
df4 = df2.drop(columns = ['a']).sample(frac = 0.5, axis = 1)
df = pd.concat([df3,df4],axis = 1)
df.to_csv("b.csv")
'''
def select_estimator(case):

        if case == 0:
            #print("SVM")
            estimator = SVC(gamma = 'scale')
        elif case == 1:
            #print("RF")
            estimator = RandomForestClassifier(random_state = 7,n_estimators = 100)
        elif case == 2:
            #print("DTree")
            estimator = DecisionTreeClassifier(random_state = 7)
        elif case == 3:
            #print("NBayes")
            estimator = GaussianNB()
        elif case == 4:
            #print("LR")
            estimator = LogisticRegression(solver = 'liblinear')   
        elif case == 5:
            #print("KN")
            estimator = KNeighborsClassifier()     
        return estimator

def evaluate(estimator,X,y,skf,num):
        acc_list = []
        for train_index, test_index in skf.split(X, y):
            estimator.fit(X[train_index],y[train_index])
            y_predict = estimator.predict(X[test_index])
            y_true = y[test_index]

            #索引
            predict_index_p = (y_predict == 1)  #预测为正类的
            predict_index_n = (y_predict == 0)  #预测为负类
            a = y_true-y_predict

            for i in range(len(a)):
                if a[i] != 0:
                    num.append(test_index[i])

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
            '''
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
            '''
            acc_list.append(acc)


        return np.mean(acc_list)
def run(nd):
        """ high-level function to run the entire class.
        """ 
        dataset = nd
        # print(dataset.shape)   #"[sample,feature]"  
        #生成类标
        labels = []
        n = 120
        maxacc = []
        for i in range(dataset.shape[0] - n):
            labels.append(0)

        for i in range(n):
            labels.append(1)
        labels = np.array(labels)
        estimator_list = [0,1,2,3,4,5]
        skf = StratifiedKFold(n_splits= 10,random_state = 7)
        num = []
        for i in estimator_list:    
            acc = evaluate(select_estimator(i),dataset,labels,skf,num)
            '''
            print("Acc: ",acc)
            print("Sn: ",sn)
            print("Sp: ",sp)
            print("Mcc: ",mcc)
            '''
            maxacc.append(acc)
        return np.max(maxacc)  

def fLoadDataMatrix(a,b):
    np.seterr(invalid='ignore')
    df = pd.read_csv('features/images_{}.csv'.format(a),keep_default_na=False)
    
    df2= pd.read_csv('features/images_{}.csv'.format(b),keep_default_na=False)

    rdr = pd.merge(df,df2,on = 'a')
    rdr1 = rdr.drop(columns = ['a'])
    index = 0
    Sample = []
    Feature = []
    Matrix = []
    NegIndex = []
    PosIndex = []
    max = []
    filename = b
    a = 92
    b = 120
    

    for i in range(a):   
        NegIndex.append(i)
    for i in range(b):
        PosIndex.append(i+a)
    Matrix = rdr1.values
    Feature = rdr1.keys()

    mt = np.array(Matrix,dtype=np.float)
    NegMatrix = mt[NegIndex]
    PosMatrix = mt[PosIndex]
    SPF = []
    for i in range(mt.shape[1]):
        s, p = stats.ttest_ind(NegMatrix[:, i], PosMatrix[:, i],equal_var = False)
        SPF.append([s,p,Feature[i],i])

    
    SPF_sorted = sorted(SPF,key=(lambda x:x[1]))
    SPF_s_np = np.array(SPF_sorted)
    nameTK = [SPF_s_np[index][2] for index in range(SPF_s_np.shape[0])]
    nd = pd.DataFrame(rdr1, columns=nameTK).values 
    for i in range(1,SPF_s_np.shape[0]):
        nd1 = nd[:,list(range(i))]
        max.append(run(nd1))
    print(np.max(max))
    '''
    nd.to_csv('ttest/HOG+{}.csv'.format(filename))
    '''


if __name__=="__main__":
    list1 = ['STRUCTURE','TRIZ']
    list2 = ['KITCHEN','LBP','STRUCTURE','TRIZ']

    for file in list2:
        print('\n')
        print(file)
        fLoadDataMatrix('HOG',file)

 
