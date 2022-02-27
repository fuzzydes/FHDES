"""
====================================================================
Dynamic selection with linear classifiers: Statistical Experiment
====================================================================
"""

import pickle
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from deslib.dcs import LCA
from deslib.dcs import MLA
from deslib.dcs import OLA
from deslib.dcs import MCB
from deslib.dcs import Rank

from deslib.des import DESKNN
from deslib.des import KNORAE
from deslib.des import KNORAU
from deslib.des import KNOP
from deslib.des import METADES
from deslib.des import FHDES
from deslib.static.oracle import Oracle
from deslib.static.single_best import SingleBest
from deslib.util.datasets import make_P2

import sklearn.preprocessing as preprocessing
import scipy.io as sio
import time
import os
import warnings
import math
from myfunctions import *

warnings.filterwarnings("ignore")


#+##############################################################################


# Prepare the DS techniques. Changing k value to 7.
def initialize_ds(pool_classifiers, uncalibratedpool, X_DSEL, y_DSEL, k=7):
    # knorau = KNORAU(pool_classifiers, k=k)
    # kne = KNORAE(pool_classifiers, k=k)
    desknn = DESKNN(pool_classifiers, k=k)
    # ola = OLA(pool_classifiers, k=k)
    # lca = LCA(pool_classifiers, k=k)
    # mla = MLA(pool_classifiers, k=k)
    # mcb = MCB(pool_classifiers, k=k)
    # rank = Rank(pool_classifiers, k=k)
    # knop = KNOP(pool_classifiers, k=k)
    meta = METADES(pool_classifiers, k=k)
    desfh_w = FHDES(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=False)
    desfh_m = FHDES(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True)
    oracle = Oracle(pool_classifiers)
    # single_best = SingleBest(pool_classifiers,n_jobs=-1)
    # majority_voting = pool_classifiers

    # UC_knorau = KNORAU(uncalibratedpool, k=k)
    # UC_kne = KNORAE(uncalibratedpool, k=k)
    # UC_desknn = DESKNN(uncalibratedpool, k=k)
    # UC_ola = OLA(uncalibratedpool, k=k)
    # UC_lca = LCA(uncalibratedpool, k=k)
    # UC_mla = MLA(uncalibratedpool, k=k)
    # UC_mcb = MCB(uncalibratedpool, k=k)
    # UC_rank = Rank(uncalibratedpool, k=k)
    # UC_knop = KNOP(uncalibratedpool, k=k)
    # UC_meta = METADES(uncalibratedpool, k=k)
    # UC_desfh_w = FHDES(uncalibratedpool, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=False)
    # UC_desfh_m = FHDES(uncalibratedpool, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True)
    # UC_oracle = Oracle(uncalibratedpool)
    # UC_single_best = SingleBest(uncalibratedpool, n_jobs=-1)
    UC_majority_voting = uncalibratedpool
    list_ds = [ oracle,desknn,meta,desfh_w,desfh_m]
    methods_names = ['Oracle','DESKNN' ,'META-DES' ,'FH-DES-C', 'FH-DES-M' ]
    # fit the ds techniques
    for ds in list_ds:
        if ds != majority_voting and ds != UC_majority_voting:
            ds.fit(X_DSEL, y_DSEL)

    return list_ds, methods_names
def write_NO_Hbox(Dataset,NO_HBox_C,NO_HBox_M):
    wpath = ExperimentPath + '/0' + Dataset + '.xlsx'

    workbook = xlsxwriter.Workbook(wpath)

    # Write Accuracy Sheet
    worksheet = workbook.add_worksheet('NO_HBox')
    worksheet.write(0,0,"NO_HBox_C")
    worksheet.write_column(1, 0, NO_HBox_C)

    worksheet.write(0,1,"NO_HBox_M")
    worksheet.write_column(1, 1, NO_HBox_M)

    worksheet.write(0, 2, Dataset)

    workbook.close()


def write_results_to_file(EPath, accuracy,labels,yhat, methods, datasetName):
    path =  EPath + "/" + datasetName + "Final Results.p"
    rfile = open(path, mode="wb")
    pickle.dump(methods,rfile)
    pickle.dump(accuracy,rfile)
    pickle.dump(labels,rfile)
    pickle.dump(yhat,rfile)
    rfile.close()

def run_process(X_train, X_DSEL, X_test, y_train, y_DSEL, y_test,n):

    state = 0
    rng = np.random.RandomState(state)
    result_one_dataset = np.zeros((NO_techniques, no_itr))
    predicted_labels = np.zeros((NO_techniques, no_itr,len(y_test)))
    yhat = np.zeros((no_itr, len(y_test)))
    NO_HBox_M = np.zeros((no_itr,1))
    NO_HBox_C = np.zeros((no_itr, 1))
    for itr in range(0, no_itr):
        if do_train:

            yhat[itr, :] = y_test

            ###########################################################################
            #                               Training                                  #
            ###########################################################################
            learner = Perceptron(max_iter=100, tol=10e-3, alpha=0.001, penalty=None, random_state=state)
            calibratedmodel = CalibratedClassifierCV(learner, cv=5,method='isotonic')
            # learner = MLPClassifier(hidden_layer_sizes=(10, 10), random_state=state)
            # model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, ), random_state=state)
            uncalibratedpool = BaggingClassifier(learner,n_estimators=NO_classifiers,bootstrap=True,
                                                 max_samples=1.0,
                                                 random_state=state)
            # uncalibratedpool.fit(X_train, y_train)

            pool_classifiers = BaggingClassifier(calibratedmodel, n_estimators=NO_classifiers, bootstrap=True,
                                                 max_samples=1.0,
                                                 random_state=state)
            pool_classifiers.fit(X_train,y_train)

            list_ds, methods_names = initialize_ds(pool_classifiers,uncalibratedpool, X_DSEL, y_DSEL, k=7)
            print("No-HBox for ", len(y_DSEL), ' samples is:', len(list_ds[4].HBoxes))
            NO_HBox_M[itr] = len(list_ds[4].HBoxes)
            NO_HBox_C[itr] = len(list_ds[3].HBoxes)
            if(save_all_results):

                save_elements(ExperimentPath+"/Pools" ,datasetName + np.str(n),itr,state,pool_classifiers,X_train,y_train,X_test,y_test,X_DSEL,y_DSEL,list_ds,methods_names)
        else: # do_not_train
            pool_classifiers,X_train,y_train,X_test,y_test,X_DSEL,y_DSEL,list_ds,methods_names = load_elements(ExperimentPath , datasetName + np.str(n),itr,state)
        ###########################################################################
        #                               Generalization                            #
        ###########################################################################

        for ind in range(0, len(list_ds)):
            result_one_dataset[ind, itr] = list_ds[ind].score(X_test, y_test) * 100
            if ind==0: # Oracle results --> y should be passed too.
                predicted_labels[ind, itr, :] = list_ds[ind].predict(X_test,y_test)
                continue
            predicted_labels[ind, itr,:] = list_ds[ind].predict(X_test)
        state += 1
    write_NO_Hbox(datasetName+np.str(n),NO_HBox_C, NO_HBox_M)
    write_results_to_file(ExperimentPath, result_one_dataset, predicted_labels, yhat, methods_names, datasetName+np.str(n))
    return result_one_dataset,methods_names,list_ds

theta = .27
NO_Hyperbox_Thereshold = 0.99
ExperimentPath = "LargeScaleExperiment"
NO_classifiers =100
no_itr = 2
save_all_results = False
do_train = True
NO_techniques = 5

list_ds = []
methods_names = []
n_samples_ = [ 1000, 10000]#, 100000, 300000, 500000, 700000,900000]
NO_datasets = len(n_samples_)
whole_results = np.zeros([NO_datasets,NO_techniques,no_itr])


dataset_count = 0
done_list = []

# Sensor Dataset ############################################################################################
#redata = sio.loadmat("LargDatasets/Sensor_900.mat")
#data = redata['dataset']
#X_DSE = data[:, 0:-1]
#y_DSE = data[:, -1]

#redata = sio.loadmat("LargDatasets/Sensor_tt.mat")
#ttdata = redata['dataset']
#X_tt = ttdata[:, 0:-1] # Tarin and Test data
#y_tt = ttdata[:, -1]
#datasetName = "Sensor"

# Artificial Dataset ###################################################################################
X, y = make_classification(n_samples=n_samples_[-1] + 2000,
                           n_features=5,
                           random_state=1)

scaler = preprocessing.MinMaxScaler()
X = scaler.fit_transform(X)

X_DSE, X_tt, y_DSE , y_tt = train_test_split(X, y, test_size=2000, stratify=y, random_state=1)  # stratify=y
datasetName = "Data"
####################################################################################################
# Spliting the Tarin and Test data
X_train, X_test, y_train, y_test = train_test_split(X_tt, y_tt, test_size=0.5, stratify=y_tt,
                                                                random_state=1)  # stratify=y

for n in n_samples_:
    X_DSEL = X_DSE[:n,:]
    y_DSEL = y_DSE[:n]
    # print("X_DSEL size is:",X_DSEL.shape)
    result,methods_names,list_ds = run_process(X_train, X_DSEL, X_test, y_train, y_DSEL,  y_test,n)
    whole_results[dataset_count,:,:] = result
    dataset_count +=1
    done_list.append(datasetName+np.str(n))


write_whole_results_into_excel(ExperimentPath,whole_results, done_list.copy(), methods_names)
path = ExperimentPath + "/WholeResults.p"
rfile = open(path, mode="wb")
pickle.dump(whole_results,rfile)
datasets = done_list
pickle.dump(datasets,rfile)
pickle.dump(methods_names,rfile)
rfile.close()


write_in_latex_table(whole_results,done_list,methods_names,rows="datasets")


duration = 4  # seconds
freq = 440  # Hz
os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
print("STD:" , np.average(np.std(whole_results,2),0))

# methods_names[0:3]+ methods_names[10:14] + methods_names[21:22]