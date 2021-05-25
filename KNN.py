from scipy.io import loadmat
from sklearn import neighbors
from matplotlib import pyplot as plt
import numpy as np
import math

if __name__ == "__main__":
    AllIn_test =     np.array(loadmat("Data/ALL_AllIn_test.mat")['AllIn_test'])
    AllIn_train =    np.array(loadmat("Data/ALL_AllIn_train.mat")['AllIn_train'])
    Disjunct_train = np.array(loadmat("Data/ALL_Disjunct_train.mat")['Disjunct_train'])
    Disjunct_test =  np.array(loadmat("Data/ALL_Disjunct_test.mat")['Disjunct_test'])

    lab_train_All = AllIn_train[:,0]
    data_train_All = AllIn_train[:,1:]
    lab_test_All = AllIn_test[:,0]
    data_test_All = AllIn_test[:,1:]
    lab_train_Dis = Disjunct_train[:,0]
    data_train_Dis =Disjunct_train[:,1:]
    lab_test_Dis = Disjunct_test[:,0]
    data_test_Dis =Disjunct_test[:,1:]

    print("start")
    acc_list_test = []
    acc_list_train = []
    for i in range(1, 11):
        print("k: ", i)
        knn = neighbors.KNeighborsClassifier(n_neighbors=i)
        knn.fit(data_train_All, lab_train_All)
        preds_test = knn.predict(data_test_All)
        acc_test = np.sum(preds_test == lab_test_All)/len(lab_test_All)
        acc_list_test.append(acc_test)

        preds_train = knn.predict(data_train_All)
        acc_train = np.sum(preds_train == lab_train_All)/len(lab_train_All)
        acc_list_train.append(acc_train)

        
        print("acc test: ", acc_test)
        print("acc train: ", acc_train)

    print("test: ", acc_list_test)
    print("train: ", acc_list_train)
    
