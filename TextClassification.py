"""
Keep in mind: this class is a work in progress. Use the buildingAClassifier script if certain functions don't work.
"""
import collections
from collections import Counter
from inspect import signature
import kpss_py3 as kps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pattern.nl as patNL
import pattern.de as patDE
import pattern.en as patEN
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance_ndarray
import re
from scipy import stats, interp
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn import metrics # 
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
from statistics import mean
import unicodedata
from yellowbrick.target import FeatureCorrelation
from yellowbrick.features.importances import FeatureImportances
from yellowbrick.text import DispersionPlot
from sklearn.feature_selection import chi2
SEED = 26062019
OUTPUT_PATH = r'output_files/'

def func(value):
    return value[1][0]

class TextClassification(object):
    ## TODO : fixing feature importance
    ##  and integrating simple word matching
    def __init__(self, X, y, model_list=[], names=[]):
        """
        Initialize the machine learning class where the 
        performance of multiple classifiers can be evaluated.
        
        Input:
            df = pandas Dataframe with text and labels
            X = text data from the entries
            y = labels
            Model list = list of sklearn Pipelines with 
                different classifiers
        """
        self.model_list = model_list
        self.X = X
        self.y = y
        #self.k_folds = preset_CV10Folds(X)
        # of median iteration
        self.fittedmodels = {}
        self.d_conf = {}
        self.l_method = []
        self.medIter = {} # dictionary key : medIter
        self.colors = {}
        ## Functions to add : getDict(), createDict()
    
    def assignPalette(self, colors=['c', 'b', 'g', 'magenta', 'indigo', 'orange', 'black']):
        l_lbls = [self.model_list[i]['clf'].__class__.__name__ for i in range(len(self.model_list))]
        for i in range(len(l_lbls)):
            self.colors[l_lbls[i]] = colors[i]
        print(self.colors)
        
    def preset_CV10Folds(self):
        """
        Split the dataset randomly 10 times for k-fold
        cross validation. 
        
        Output:
            l_folds = list containing the indices for the train/test
                entries, required to contstruct the k-folds.
        """
        ss = ShuffleSplit(n_splits=10, test_size=0.5, random_state=SEED)
        l_folds = ss.split(self.X)
        return l_folds
    
    def binarizeLabel(self, l, true_label='y'):
        """
        This function codifies the provided binary labels
         to 1 and 0 according to the true label that was specified
        """
        self.y = np.array([int(i == true_label) for i in self.y])
    
    def fitModels(self):
        # AUC weghalen 
        # Later -> identify proba methods & non-proba methods and account for those
        #p#rint(self.model_list)
        d_conf = {}
        for clf in range(len(self.model_list)):
            lbl = self.model_list[clf]['clf'].__class__.__name__
            print('loading model: ', lbl)
            estimator = self.model_list[clf]
            self.l_method.append(self.model_list[clf])
            d_conf[lbl] = self.fitting(estimator, lbl) # plt, mean_auc, aucs, medianModel = 
            print('nr of iterations: ', len(d_conf[lbl]))
            #break
        self.d_conf = d_conf
            
    def createDictionary(self, clf='Gradient Boosting', **kwargs):
        """
        Creates a dictionary of terms for the median iteration
        
        Input:
            clf = name of classifier
        """
        X_train_fold = self.X[self.fitted_models[clf][4]] # 4 - train index
        if 'ngram_range' in kwargs.keys():
            count_vect = TfidfVectorizer(ngram_range=kwargs['ngram_range'])
        else :
            count_vect = TfidfVectorizer()
        X_train_tfidf = count_vect.fit_transform(X_train_fold) 
        print(len(X_train_tfidf))
        return
    
    
    def scores(self, lbl, iteration):
        # 0=tp, 1=fp, 2=tn, 3=fn
        d = self.d_conf[lbl][iteration]
        print(d[len(d)-1])
        acc = [(d[i][0] + d[i][2])/(d[i][0]+d[i][1]+d[i][2]+d[i][3]) for i in d.keys()]
        tpr = [d[i][0]/(d[i][0]+d[i][3]) if (d[i][0]+d[i][3]) != 0 else 0 for i in d.keys()]
        fpr = [d[i][1]/(d[i][1]+d[i][2]) if (d[i][1]+d[i][2]) != 0 else 0 for i in d.keys()]
        prec = [d[i][0]/(d[i][0]+d[i][1]) if (d[i][0]+d[i][1]) != 0 else 0 for i in d.keys()]
        #print(tpr)
        #exit()
        return [acc, tpr, prec, fpr]
    
    def plotROC(self, lbl_list=[]):
        """
        Calculates the specificity and sensitivity for the provided 
        classifier (clf). 

        Input: 
            X = array with text data (EHR entries)
            y_b = array with corresponding labels (binarized to 1/0)
            clf = classifier object (Pipeline)
            color = color to represent classifier in the plot 

        Output:
            plt = matplotlib pyplot featuring the ROC curve
                of a single or a multitude of classifiers
        """
        if lbl_list == []:
            lbl_list = list(self.d_conf.keys())
        if type(lbl_list) == list:
            for lbl in lbl_list:
                plt = self.modelROC(lbl)
        elif type(lbl_list) == str:
            plt = self.modelROC(lbl_list)
        return plt
    
    def plotPR(self, l_prec, aucs, color, lbl, linestyle='-', lw=5, std=False):
        """
        Plot the precision recall curve by taking the
        mean of the k-folds. The standard deviation is also
        calculated and plotted on the screen.

        Input:
            l_prec = list of precision scores per iteration
            l_rec = list of recall scores per iteration
            aucs = list of area under the curve per iteration
            color = color for line
            lbl = name of classifier
            linestyle = linestyle (matplotlib.pyplot)
            lw = linewidth

        Output:
            plt = Precision Recall curve with standard deviation 
                (matplotlib.pyplot)
        """
        mean_precision = [*map(mean, zip(*l_prec))]
        mean_precision[-1] = 0.0
        recall_scale = np.linspace(0, 1, 100)
        mean_auc = self.calculateAUC(recall_scale, mean_precision)
        std_auc = np.std(aucs)
        std_precision = np.std(mean_precision, axis=0)
        if std:
            precision_upper = np.minimum(mean_precision + std_precision, 1)
            precision_lower = np.maximum(mean_precision - std_precision, 0)
            plt.fill_between(recall_scale, precision_lower, precision_upper, color=color, alpha=.1)
        plt.plot(recall_scale, mean_precision, color=color, alpha=0.6, 
                 label=lbl + r' mean kfold (AUPRC = %0.2f $\pm$ %s)' % (mean_auc, std_auc),
                 linestyle=linestyle, linewidth=lw)
        print(lbl + ' ' + str(mean_auc) +' (std : +/-' + str(std_auc) + ' )')
        return plt
    
    def calculateAUC(self, x, y):
        """
        Calculate AUC by parts by calculating the surface area of a 
        trapzoid.

        x = x-axes 
        y = y-axes (interpolated with x)
        """
        auc = 0
        for i in range(1,len(y)):
            last_x = x[i-1]
            last_y = y[i-1]
            cur_x = x[i]
            cur_y = y[i]
            auc += np.trapz([last_y, cur_y], [last_x, cur_x])
        return auc
    
    def modelROC(self, lbl):
        """
        Calculates the specificity and sensitivity for the provided 
        classifier (clf). 

        Input: 
           lbl = name of classifier

        Output:
            plt = matplotlib pyplot featuring the ROC curve
                of one classifier
        """
        l_roc, aucs = [], []
        fpr_scale = np.linspace(0, 1, 100)
        tprs = []
        for x in range(len(self.d_conf[lbl])): # loop through iterations
            #scores = self.scores(lbl, x)
            tpr, fpr = self.d_conf[lbl][x]['tpr'], self.d_conf[lbl][x]['fpr']#scores[1], scores[3]
            tprs.append(interp(fpr_scale, fpr, tpr))
            tprs[-1][0] = 0.0
            auc = self.calculateAUC(fpr, tpr)
            self.d_conf[lbl][x]['auc'] = auc
            aucs.append(auc) # remove later on
        plt, mean_auc = self.plotSTD(tprs, aucs, self.colors[lbl], lbl)
        #self.plotSTD(tprs_t, aucs_t, color, 'Train-score ' + lbl, '-', 5, 0)
        return plt
    
    def writePredictionsToFile(self, name, pred, true):
        """
        Write predictions of the classifier to a simple CSV file. 
        These files can be processed in pROC for the Delong test

        Input:
            name = name of classifier
            pred = list of predictions
            true = list of true labels
        """
        d = {'PRED': pred, 'TRUE': true}
        df = pd.DataFrame(data=d)
        df.to_csv(OUTPUT_PATH + 'pred' + name.replace(" ", "") + '.csv', sep='|', index=False)
        return
    
    def modelPrecisionRecall(self, lbl):
        """
        Calculates the precision and recall for the provided 
        classifier (clf). 

        Input: 
            lbl = name of classifier

        Output:
            plt = matplotlib pyplot featuring the ROC curve
                of one classifier
        """
        l_prec, aucs = [], []
        recall_scale = np.linspace(0, 1, 100)
        for x in range(len(self.d_conf[lbl])): # loop through iterations
            tpr, prec = self.d_conf[lbl][x]['tpr'], self.d_conf[lbl][x]['prc']
            prec[0] = 0.0
            inter_prec = interp(recall_scale, tpr, prec)
            inter_prec[0] = 1.0 
            auc = self.calculateAUC(recall_scale, inter_prec)
            self.d_conf[lbl][x]['auc'] = auc
            aucs.append(auc)
            l_prec.append(inter_prec)
        aucs = [self.d_conf[lbl][j]['auc'] for j in range(len(self.d_conf[lbl]))]
        aucs.sort()
        middleIndex = round((len(aucs) - 1)/2) # normally 5 -> if 10 fold
        medianModel = self.fittedmodels[lbl][middleIndex] # change to actual label
        foldTrueLbl = medianModel[3]
        self.medIter[lbl] = middleIndex # toDo
        self.writePredictionsToFile(lbl, medianModel[1], foldTrueLbl)
        plt = self.plotPR(l_prec, aucs, self.colors[lbl], lbl)
        return plt
    
    def getMedianModel(self, lbl):
        """
        Required : fitted models & assessed median model
        
        This function retrieves the median iteration of the
        specified classifier (lbl)
        
        Input 
            lbl = name of classifier
        Output:
            medianModel = median iteration of classifier
        """
        middleIndex = self.medIter[lbl]
        medianModel = self.fittedmodels[lbl][middleIndex] # change to actual label
        if (write): # write actual predictions
            foldTrueLbl = medianModel[3]
            self.writePredictionsToFile(lbl, medianModel[1], foldTrueLbl)
        self.medIter[lbl] = middleIndex # toDo
        return medianModel[0]
    
    def plotPrecisionRecall(self, lbl_list=[]):
        """
        Calculates the precision and recall for the provided 
        classifier (clf). 

        Input: 
            X = array with text data (EHR entries)
            y_b = array with corresponding labels (binarized to 1/0)
            clf = classifier object (Pipeline)
            color = color to represent classifier in the plot 

        Output:
            plt = matplotlib pyplot featuring the Precision Recall curve
                of one classifier
        """
        if lbl_list == []:
            lbl_list = list(self.d_conf.keys())
        if type(lbl_list) == list:
            for lbl in lbl_list:
                plt = self.modelPrecisionRecall(lbl)
        elif type(lbl_list) == str:
            plt = self.modelPrecisionRecall(lbl_list)
        return plt
    
    def assessPerformance(self, lbl, clf, iteration, X_test, y_test, proba=True):
        """
        Applies the fitted models on the test set (X_test) and returns probabilities or 
        boolean predictions. This performance is then measured at different probabilities
        a.k.a cut-off scores.
        
        Input: 
            lbl = name of classifier
            clf = classifier object (Pipeline)
            iteration = integer implying the current iteration of the 10 fold CV
            X_test = unlabeled test set
            y_test = labels of the test set
            
        Output: 
            d_conf = dictionary with confusion matrix & other performance characteristics 
                over the range of probabilities. 
        """
        recall_scale = np.linspace(0, 1, 100)
        if (proba) :
            probas_ = clf.predict_proba(X_test)
            pred = probas_[:,1]
        else :
            pred = clf.predict(X_test)
        l_sorted_true, l_sorted_pred = self.sortedPredictionList(pred, y_test)
        self.fittedmodels[lbl][iteration] = [clf, pred, X_test, y_test] 
        d_conf = self.score_binary(l_sorted_true, l_sorted_pred) # score binary equiv -> met prec/ recall
        return d_conf
    
    def sortedPredictionList(self, b_pred, y_test):
        """
        This function sorts the list of true labels by the
        list of predictions. The sorted list of true labels
        can be used to create a ROC-curve for a non-probability
        classifier (a.k.a. a binary classifier like decision tree).

        Input:
            b_pred = list of hard-predictions (0 or 1) 
                or probabilities (0-1)
            y_test = list of actual labels, binarized to 
                1 or 0. 

        Example for generating 'l_sorted_true':
            Before sort:
               pred: 0 1 1 0 1 0 1 0 1 1 1 0 1 0
               true: 0 1 0 0 1 0 0 1 1 0 1 0 1 1
            After sort:
               pred: 1 1 1 1 1 1 1 1 0 0 0 0 0 0 
            -> true: 1 1 0 1 0 0 1 1 0 0 1 1 0 0

        Output:
            l_sorted_true = list of true label sorted on the 
                predictions label:
        """
        d_perf_dt = {}
        count = 0
        for i in range(0,len(y_test)):
            d_perf_dt[count] = [b_pred[count], y_test[count]]
            count += 1
        orderedDict = collections.OrderedDict(sorted(d_perf_dt.items(), key=lambda k: func(k), reverse=True))
        l_sorted_pred= []
        l_sorted_true = []
        for x in orderedDict.items():
            l_sorted_pred.append(x[1][0])
            l_sorted_true.append(x[1][1])
        return l_sorted_true, l_sorted_pred
    
    def confusion_window(self, l_true, l_pred):
        """
        makes a record at every point it registers the confusion matrix
        l_true and l_pred are sorted 
        """
        dummi = l_true
        dummi = [2 if x==0 else x for x in dummi]
        dummi = [x -1 for x in dummi]
        
        # Compute basic statistics:
        d_conf = {}
        tp, fp, tn, fn = 0, 0, 0, 0
        l_tp, l_fp, l_tn, l_fn = [], [], [], []
        for i in range(len(l_true)) :
            if l_true[i]==1 : # actual case
                fn +=  1 - l_pred[i]
                tp += l_pred[i]
            elif l_true[i]==0 : # no case
                fp += l_pred[i]
                tn += 1 - l_pred[i]
            l_tp.append(tp)
            l_fp.append(fp)
            l_tn.append(tn)
            l_fn.append(fn)
        l_true.insert(0,0)
        dummi.insert(0,0)
        P = sum(l_true)
        N = sum(dummi)
        TPR = pd.Series(l_tp).divide(P) # sensitivity / hit rate / recall
        FPR = pd.Series(l_fp).divide(N)  # fall-out
        PRC = pd.Series(l_tp).divide(pd.Series(l_tp) + pd.Series(l_fp))
        d_conf = {'tp': l_tp, 'fp': l_fp, 'fn': l_fn, 'fn': l_tn, 'tpr': TPR, 'fpr': FPR, 'prc': PRC}
        return d_conf
    
    def score_binary(self, l_true, l_pred):
        """
        Calculates the dummy true en false positive rate for 
        a classifier that doesn't calculate probabilities 

        Input:
            l_true = list of true label sorted on the 
                predictions label.
                The function sortedPredictionList can
                be used to generate such a list!
        Output:
            TPR = list with true positive rates 
            FPR = list with false positive rates 
            PRC = list with precision (PPV)
        """
        dummi = l_true
        dummi = [2 if x==0 else x for x in dummi]
        dummi = [x -1 for x in dummi]
        l_pred.insert(0,0)
        l_true.insert(0,0)
        dummi.insert(0,0)
        # Compute basic statistics:
        TP = pd.Series(l_true).cumsum()
        FP = pd.Series(dummi).cumsum()
        P = sum(l_true)
        N = sum(dummi)
        TPR = TP.divide(P) # sensitivity / hit rate / recall
        FPR = FP.divide(N)  # fall-out
        PRC = TP.divide(TP + FP) # precision
        #d_conf = {}
        d_conf = {'tpr': TPR, 'fpr': FPR, 'prc': PRC, 'threshold': l_pred}
        return d_conf 
    
    
    def fitting(self, clf, lbl):
        """
        Calculates the sensitivity and auc for the provided 
        classifier (clf) for every fold in the k-fold validation. 
        The average auc is determined for both the trainingsset & the 
        testset.

        K-fold crossvalidation is utilized to give a higher resolution
        of the performance of the classifier.

        The median model (with the median AUC) will be used to 
        calculate the optimal cut-off & will be returned as 
        output.

        Input: 
            X = array with text data (EHR entries)
            y = array with corresponding labels (binarized to 1/0)
            l_folds = array of k-folds 
            clf = classifier object (Pipeline)
            color = color to represent classifier in the plot 
            lbl = name of classifier

        Output:
            plt = matplotlib pyplot featuring the Precision Recall curve
                of one classifier
            mean_auc = float of mean auc
            aucs = array of auc scores (used to assess medianModel)
            medianModel = median iteration of the classifier ->
                the median iteration is chosen because the validation is
                done k-times with a different train/test set each time. 
        """
        d_conf = {}
        iteration = 0
        self.fittedmodels[lbl] = {} # initialize entry
        for train_index, test_index in self.preset_CV10Folds():
            fold = [test_index, train_index]
            Xtr, Xte = self.X[train_index], self.X[test_index]
            estimator = clf.fit(Xtr, self.y[train_index])
            d_conf[iteration] = self.assessPerformance(lbl, estimator, iteration, Xte, self.y[test_index], fold)
            iteration += 1
        return d_conf
    
    def getConfusionMatrix(self, lbl, desired=0.9, most_val='tpr', **kwargs): # , **kwargs
        """
        Retrieve confusion matrix corresponding with preferred characteristics:
        Either sensitivity or precision at a specified cut-off value (desired). Furthermore, 
        the other characteristic is also weighed into the equation to find the optimal balance.
        """
        medIter = self.medIter[lbl] # change to dict with label todo
        print('Generating confusion matrix for', lbl, 'based on median Iteration (AUPRC): ', medIter)
        d = self.d_conf[lbl][medIter][most_val]
        l_cols = [i for i in self.d_conf[lbl][medIter].keys() if i not in [most_val, 'threshold', 'fpr', 'auc']] # 'prc', 
        l_candidates = [i for i in range(len(d)) if d[i]>desired]
        max_l = [0]
        thresh = 0
        print('Other weighing variables: ', l_cols)
        for i in l_candidates:
            l = [self.d_conf[lbl][medIter][c][i] for c in self.d_conf[lbl][medIter].keys() if c in l_cols ]
            #print(l)
            if sum(l) > sum(max_l):
                max_l = l
                #print(l)
                thresh = self.d_conf[lbl][medIter]['threshold'][i]
                fpr = self.d_conf[lbl][medIter]['fpr'][i]
                prc = self.d_conf[lbl][medIter]['prc'][i]
                tpr = self.d_conf[lbl][medIter]['tpr'][i]
        print('Thresh: %.2f' % thresh, '\nPRC: \t%.2f' % prc, '\nSens: \t%.2f' % tpr, '\nSpec: \t%.2f' % float(1-fpr))
        self.plot_confusion_matrix(lbl, thresh)
    
    def plotSTD(self, tprs, aucs, color, lbl, linestyle='-', lw=5, vis=1):
        """
        Plot the standard deviation of the ROC-curves

        Input:
            tprs = list of true positive rates per iteration
            aucs = list of area under the curve per iteration
            lbl = name of the classifier (string)
            color = specify color of the classifier
            lw = linewidth (float)
            vis = visualize

        Output:
            plt = matplotlib pyplot featuring the standard 
                deviation of the ROC curve from the classifier
            std_auc = standard deviation of the auc (float)
        """
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr [-1] = 1.0

        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        print(lbl + ' ' + str(mean_auc) +' (std : +/-' + str(std_auc) + ' )')
        if vis==1:
            plt.plot(mean_fpr, mean_tpr, color=color,
                label=lbl + r' mean kfold (AUC = %0.2f $\pm$ %s)' % (mean_auc, std_auc),
                alpha=.5, linestyle=linestyle, linewidth=lw)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.1)
            return plt, std_auc
        else :
            return
    
    def plot_confusion_matrix(self, lbl, desired, classes=[0,1],
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
        """
        [SKLEARN function]
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        medIter = self.medIter[lbl]
        y_true =  self.fittedmodels[lbl][medIter][3]
        y_proba = self.fittedmodels[lbl][medIter][1]
        y_pred = [int(i >= desired) for i in y_proba]
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        #classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax