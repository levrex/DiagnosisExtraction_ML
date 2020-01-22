"""
Keep in mind: this class is a work in progress. Use the buildingAClassifier script if certain functions don't work.
"""
import collections
from collections import Counter
from inspect import signature
import kpss_py3 as kps
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pattern.nl as patNL
import pattern.de as patDE
import pattern.en as patEN
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance_ndarray
import re
from scipy import stats, interp
import seaborn as sns
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

def func(value):
    return value[1][0]

class CustomBinaryModel(object):
    """
    Summary class:
        Use this class to create a binary prediction model
        that classifies entries according to the presence of
        targets (words) in the free written text. The targets
        can be provided during the creation of the object or 
        by calling the setTargets function
    """
    def __init__(self, targets):
        self.targets = targets
        
    def setTargets(self, targets):
        self.targets = targets
    
    def getTargets(self):
        return self.targets
    
    def binarize(self, value):
        """
        This function codifies the binary labels 'y' and 'n'
         to 1 and 0.
        """
        return int(value == 'y')
    
    def judgeEntry(self, report):
        """
        Predict the class on the presence of certain words (targets)
        in the free written text from EHR records
        """
        regexp = re.compile(r'\b('+r'|'.join(self.targets)+r')\b')
        if regexp.search(report):
            return 'y'
        else :
            return 'n'

    def predict(self, X):
        """
        Predict multiple EHR entries and return the predictions
        """
        l_context= [self.judgeEntry(str(x)) for x in X]
        pred = [l_context[x][0] for x in range(len(l_context))]
        pred = np.array([self.binarize(val) for val in pred])
        return pred

class TextClassification(object):
    ## TODO : fixing feature importance
    ##  and integrating simple word matching
    def __init__(self, X, y, model_list=[], names=[], seed=26062019):
        """
        Initialize the machine learning class where the 
        performance of multiple classifiers can be evaluated.
        
        Input:
            df = pandas Dataframe with text and labels
            X = text data from the entries
            y = labels
            Model list = list of sklearn Pipelines with 
                different classifiers
            seed = seed for creating the folds to ensure same output
        """
        self.model_list = model_list
        self.X = X
        self.y = y
        self.seed = seed
        self.iterations = 10
        self.test_size = 0.5
        self.names = names
        #self.k_folds = preset_CV10Folds(X)
        # of median iteration
        self.fittedmodels = {}
        self.d_conf = {}
        self.d_aucs = {}
        self.d_auprcs = {}
        self.l_method = []
        self.medIter = {} # dictionary key : medIter
        self.palette = ['r', 'y', 'c', 'b', 'g', 'magenta', 'indigo', 'black', 'orange'] 
        self.output_path = r'output_files/'
        self.colors = {}
        self.ref = ''
    
    def setREF(self, ref):
        self.ref = ref
    
    def getREF(self):
        return self.ref
    
    def getAUC(self):
        return self.d_aucs
    
    def getAUPRC(self):
        return self.d_auprcs
    
    def setOutputPath(self, output_path):
        """
        Update output path with the pathway to the directory requested by the user
        """
        self.output_path = output_path
        
    def getOutputPath(self):
        """
        Retrieve pathway to current directory for output
        """
        return self.output_path
        
        
    def setSeed(self, seed):
        """
        Update seed with the provided user input
        """
        self.seed = seed 

    def getSeed(self):
        """
        Retrieve current seed that is used for the fold creation
        Reminder: if you change the seed you are likely to get a different
            output because you will have a different random train/test-split. 
        """
        return self.seed
    
    def setIterations(self, iterations):
        """
        Update nr of iterations with the provided user input
        """
        self.iterations = iterations

    def getIterations(self):
        """
        Retrieve current nr of iterations that is used for the fold creation
        """
        return self.iterations
    
    def getFittedModels(self):
        """
        Retrieve list with fitted models
        """
        return self.fittedmodels
        
    def assignPalette(self, palette=[]):
        l_lbls = self.names #[self.model_list[i]['clf'].__class__.__name__ for i in range(len(self.model_list))]
        if palette == []:
            palette = self.palette
        for i in range(len(l_lbls)):
            self.colors[l_lbls[i]] = palette[i]
        print(self.colors)
        
    def splitData(self):
        """
        Split the dataset randomly n-times (defined by self.iterations) for k-fold
        cross validation. The size of the test_size is defined by self.test_size. 
        
        Output:
            l_folds = list containing the indices for the train/test
                entries, required to contstruct the k-folds.
        """
        ss = ShuffleSplit(n_splits=self.iterations, test_size=self.test_size, random_state=self.seed)
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
        iterat = 0
        print('\nGeneral settings for training/testing:')
        print('Method = Cross Validation ' + str(self.iterations) + '-fold')
        print('\tfraction test:\t', self.test_size, '\n')
        for clf in range(len(self.model_list)):
            lbl = self.names[iterat]
            print('loading model: ', lbl)
            estimator = self.model_list[clf]
            self.l_method.append(self.model_list[clf])
            d_conf[lbl] = self.fitting(estimator, lbl) # plt, mean_auc, aucs, medianModel = 
            iterat += 1
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
        """
        Assess different performance characteristicat every point in the 
        sorted list of true labels. The list is sorted according to the probabilities. 
        In methods that don't use probabilities, pseudo-metrics are calculated.
        """
        d = self.d_conf[lbl][iteration]
        print(d[len(d)-1])
        acc = [(d[i][0] + d[i][2])/(d[i][0]+d[i][1]+d[i][2]+d[i][3]) for i in d.keys()]
        tpr = [d[i][0]/(d[i][0]+d[i][3]) if (d[i][0]+d[i][3]) != 0 else 0 for i in d.keys()]
        fpr = [d[i][1]/(d[i][1]+d[i][2]) if (d[i][1]+d[i][2]) != 0 else 0 for i in d.keys()]
        prec = [d[i][0]/(d[i][0]+d[i][1]) if (d[i][0]+d[i][1]) != 0 else 0 for i in d.keys()]
        return [acc, tpr, prec, fpr]
    
    def plotSettings(self, category='ROC'):
        """
        Setting for creating an ROC-plot or PR-plot
        """
        params = {'legend.fontsize': 10,
          'figure.figsize': (14+2*(category=='PR'),10),
          'axes.grid': False,
         'axes.labelsize': 26,
         'axes.titlesize':'xx-small',
         'xtick.labelsize':22,
          'axes.labelcolor' : 'k',
          'ytick.color' : 'k',
          'xtick.color': 'k',
        'font.weight':'regular',
         'ytick.labelsize':22}
        plt.rcParams.update(params)
        plt.figure()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        #plt.rcParams.update({'font.size': 12})
        return plt

    def plotROC(self, lbl_list=[], legend=True):
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
        plt = self.plotSettings()
        plt.ylabel('Sensitivity (TPR)')
        plt.xlabel('1 - Specificity (FPR)')
        
        if lbl_list == []:
            lbl_list = list(self.d_conf.keys())
        if type(lbl_list) == list:
            for lbl in lbl_list:
                plt, aucs = self.modelROC(lbl)
                self.d_aucs[lbl] = aucs
        elif type(lbl_list) == str:
            plt, aucs = self.modelROC(lbl_list)
            self.d_aucs[lbl_list] = aucs
        plt.legend(loc = 'lower right')
        if legend :
            plt.rcParams.update({'font.size': 55})
        return plt
    
    def plotPR(self, l_prec, aucs, color, lbl, linestyle='-', lw=5, std=False, kfold=True):
        """
        Plot the precision recall curve by taking the
        mean of the k-folds. The standard deviation is also
        calculated, though not visualized.

        Input:
            l_prec = list of precision scores per iteration
            aucs = list of area under the curve per iteration
            color = color for line
            lbl = name of classifier
            linestyle = linestyle (matplotlib.pyplot)
            lw = linewidth
            std = visualize the standard deviation (default=False)
            kfold = calculate the standard deviation (default=True)

        Output:
            plt = Precision Recall curve with standard deviation 
                (matplotlib.pyplot)
        """
        if lbl == self.ref:
            linestyle = '--'
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
        if kfold:
            lbl += r' mean kfold (AUPRC = %.2f +/- %.2f)' % (mean_auc, std_auc)
            print(lbl)
        else : 
            lbl += r' (AUPRC = %.2f)' % (mean_auc)
            print(lbl + ' ' + str(mean_auc))
        plt.plot(recall_scale, mean_precision, color=color, alpha=0.6, 
                 label=lbl, linestyle=linestyle, linewidth=lw)
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
        plt= self.plotSTD(tprs, aucs, self.colors[lbl], lbl)
        #self.plotSTD(tprs_t, aucs_t, color, 'Train-score ' + lbl, '-', 5, 0)
        return plt, aucs
    
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
        df.to_csv(self.output_path + 'pred' + name.replace(" ", "") + '.csv', sep='|', index=False)
        return
    
    def modelPrecisionRecall(self, lbl, plot=True):
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
        if plot : 
            plt = self.plotPR(l_prec, aucs, self.colors[lbl], lbl)
            return plt, aucs
        else :
            return 
    
    def plotScatter(self, lbl):
        """
        Required : fitted models
        
        This function creates a scatter plot based
        on the median iteration of the provided classifier (lbl)
        """
        params = {'legend.fontsize': 10,
          'figure.figsize': (8,6),
          'axes.grid': False,
         'axes.labelsize': 26,
         'axes.titlesize':'xx-small',
         'xtick.labelsize':22,
          'axes.labelcolor' : 'k',
          'ytick.color' : 'k',
          'xtick.color': 'k',
        'font.weight':'regular',
         'ytick.labelsize':22}
        plt.rcParams.update(params)
        
        try :
            middleIndex = self.medIter[lbl]
        except :
            print('Retrieving fitted ' + str(lbl) + ' (default=Median Iteration)')
            self.modelPrecisionRecall(lbl, plot=False)
            middleIndex = self.medIter[lbl]
        medianModel = self.fittedmodels[lbl][middleIndex] 
        foldTrueLbl = medianModel[3]
        plt.scatter(x=medianModel[1], y=list(range(len(medianModel[1]))), c=foldTrueLbl, cmap='viridis')    
        return plt
    
    def getTrainedClassifier(self, lbl, clf=True, write=True):
        """
        Required : fitted models
        
        This function retrieves the median iteration of the
        specified classifier (lbl). If the median iteration
        has not yet been determined than this iteration is 
        calculated. 
        
        Input 
            lbl = name of classifier
            clf = boolean to specify wheter you want to return
                only the fitted classifier or additional information 
                (Useful if you want to replicate the learning process)
                Like:
                    - train_index (for recreating trainingsset)
                    - test_index
        Output:
            medianModel = median iteration of classifier
        """
        try :
            middleIndex = self.medIter[lbl]
        except :
            print('Retrieving fitted ' + str(lbl) + ' (default=Median Iteration)')
            self.modelPrecisionRecall(lbl, plot=False)
            middleIndex = self.medIter[lbl]
        medianModel = self.fittedmodels[lbl][middleIndex]
        if (write): # write actual predictions
            foldTrueLbl = medianModel[3]
            self.writePredictionsToFile(lbl, medianModel[1], foldTrueLbl)
        self.medIter[lbl] = middleIndex # toDo
        if clf:
            return medianModel[0]
        else :
            return medianModel
        
    def plotPrecisionRecall(self, lbl_list=[], legend=True):
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
        plt = self.plotSettings('PR')
        plt.ylabel('Precision (PPV)')
        plt.xlabel('Recall (TPR)')
        if lbl_list == []:
            lbl_list = list(self.d_conf.keys())
        if type(lbl_list) == list:
            for lbl in lbl_list:
                plt, aucs = self.modelPrecisionRecall(lbl)
                self.d_auprcs[lbl] = aucs
        elif type(lbl_list) == str:
            plt, aucs = self.modelPrecisionRecall(lbl_list)
            self.d_auprcs[lbl_list] = aucs
        if legend :
            plt.legend(loc = 'lower right')
        plt.rcParams.update({'font.size': 55})
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
            cat = type/category of classifier: whether a classifier returns a hardcase label
                or a probability or if it is a word matching method
        Output: 
            d_conf = dictionary with confusion matrix & other performance characteristics 
                over the range of probabilities. 
        """
        recall_scale = np.linspace(0, 1, 100)
        if proba :
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
        Makes a record at every data point in an array, which makes it
        possible to retrieve a confusion matrix that corresponds with a certain 
        threshold. 
        
        Input: 
            Sorted l_true and l_pred lists that correspond with eachother
            
        Output:
            d_conf = dictionary that stores measurements over the whole dataset
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
        proba = True
        self.fittedmodels[lbl] = {} # initialize entry
        for train_index, test_index in self.splitData():
            fold = [test_index, train_index]
            Xtr, Xte = self.X[train_index], self.X[test_index]
            try :
                estimator = clf.fit(Xtr, self.y[train_index])
            except: 
                if iteration == 0: 
                    print(str(lbl), 'is assumed to be a word matching method and is therefore not fitted')
                estimator = clf
                proba = False
            d_conf[iteration] = self.assessPerformance(lbl, estimator, iteration, Xte, self.y[test_index], proba)
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
        try: 
            print('Thresh: %.2f' % thresh, '\nPRC: \t%.2f' % prc, '\nSens: \t%.2f' % tpr, '\nSpec: \t%.2f' % float(1-fpr))
            self.plot_confusion_matrix(lbl, thresh)
        except: 
            print('No situation found where %s > %.2f ' % (lbl, desired))
        
    
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
        if lbl == self.ref:
            linestyle = '--'
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr [-1] = 1.0

        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        print(lbl + ' ' + str(mean_auc) +' (std : +/-' + "%.2f" % std_auc + ' )')
        if vis==1:
            plt.plot(mean_fpr, mean_tpr, color=color,
                label=lbl + r' mean kfold (AUC = %.2f +/- %.2f)' % (mean_auc, std_auc),
                alpha=.5, linestyle=linestyle, linewidth=lw)
            #plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.1)
            return plt #, std_auc
        else :
            return
        
    def scoresCM(self, CM):
        """
        Derive performance characteristics from the confusion matrix
        """
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)

        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)
        return [TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC]
        
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
                title = 'Confusion matrix'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        #classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #    print("Normalized confusion matrix")
        #else:
        #    print('Confusion matrix')

        cm_chars = self.scoresCM(cm)
        print('NPV:\t' + str(cm_chars[3]) + '\nACC:\t' + str(cm_chars[7]))
        
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
    
    def plotSwarm(self, lbl):
        """
        Required : fitted models
        
        This function creates a swarm plot based
        on the median iteration of the provided classifier (lbl)
        """
        params = {'legend.fontsize': 10,
          'figure.figsize': (8,6),
          'axes.grid': False,
         'axes.labelsize': 26,
         'axes.titlesize':'xx-small',
         'xtick.labelsize':22,
          'axes.labelcolor' : 'k',
          'ytick.color' : 'k',
          'xtick.color': 'k',
        'font.weight':'regular',
         'ytick.labelsize':22}
        plt.rcParams.update(params)
        
        try :
            middleIndex = self.medIter[lbl]
        except :
            print('Retrieving fitted ' + str(lbl) + ' (default=Median Iteration)')
            self.modelPrecisionRecall(lbl, plot=False)
            middleIndex = self.medIter[lbl]
        medianModel = self.fittedmodels[lbl][middleIndex] 
        foldTrueLbl = medianModel[3]
        df = pd.DataFrame(data={'Y': medianModel[1], 'X': foldTrueLbl})
        ax = sns.swarmplot(x="X", y="Y", data=df)
        plt.ylabel('Probability')
        plt.xlabel('RA')
        plt.title('Swarm plot ' + str(lbl))
        return plt
    
    def samplingCurveROC(self, name, stepsize=0, cv=True):
        """
        This function calculates the ROC-curve AUC with differing sample 
        sizes

        Reminder - same K-folds as plotted in the ROC-curve

        Input:
            name = name of classifier (string)
            stepsize = size of steps
            cv = apply cross fold (warning: not suggested if training takes a long time)

        Output:
            plt = matplotlib pyplot featuring the Precision Recall curve
                of one classifier
            mean_auc = float of mean auc
        """
        model_id = self.names.index(name)
        clf = self.model_list[model_id]
        colors=['r', 'y', 'c', 'b', 'g', 'magenta', 'indigo', 'black', 'orange'] 
        fpr_scale = np.linspace(0, 1, 100)
        l_folds = [(train, test) for train, test in self.splitData()]
        d_aucs = {}
        
        if stepsize == 0: # if not defined
            stepsize = int(len(l_folds[0][0])/5)
        
        for i in range(stepsize, len(l_folds[0][0]), stepsize):
            tprs = []
            aucs = []
            for train_ix, test_ix in l_folds:
                df_test = pd.DataFrame(data={'IX': test_ix, 'Outcome': self.y[test_ix], 
                                            'Text' : self.X[test_ix]})
                df_train = pd.DataFrame(data={'IX': train_ix, 'Outcome': self.y[train_ix], 
                                            'Text' : self.X[train_ix]})
                
                df_sub = df_train.sample(n=i, random_state=self.seed)
                # Assess wheter or not the entries from the smaller sample (within the same fold) are conserved
                # print('HEAD:', list(df_sub['IX'].head()), len(df_sub), 'TAIL:', list(df_sub['IX'].tail())) 
                estimator = clf.fit(df_sub['Text'], df_sub['Outcome'])
                probas_ = estimator.predict_proba(df_test['Text'])
                fpr, tpr, thresholds = metrics.roc_curve(df_test['Outcome'], probas_[:, 1])
                tprs.append(interp(fpr_scale, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = metrics.auc(fpr, tpr)
                aucs.append(roc_auc)
                if cv==False:
                    break
            plt = self.plotSTD(tprs, aucs, colors[int(i/stepsize)-1], 'n=' + str(i)) # , mean_auc
            d_aucs['n=' + str(i)] = aucs
        
        # Also for final sample size
        tprs = []
        aucs = []
        for train_ix, test_ix in l_folds:
            estimator = clf.fit(self.X[train_ix], self.y[train_ix])
            probas_ = estimator.predict_proba(self.X[test_ix])
            fpr, tpr, thresholds = metrics.roc_curve(self.y[test_ix], probas_[:, 1])
            tprs.append(interp(fpr_scale, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = metrics.auc(fpr, tpr)
            aucs.append(roc_auc)
            if cv==False:
                break
        plt = self.plotSTD(tprs, aucs, colors[int(i/stepsize)], 'n=' + str(len(l_folds[0][0]))) # , mean_auc
        d_aucs['n=' + str(len(l_folds[0][0]))] = aucs
        
        
        plt.rcParams.update({'font.size': 20})
        plt.legend()
        plt.title(name + ' ROC-curve for different sample sizes')
        plt.figure(figsize=(14,10))
        return plt, d_aucs
    
    def samplingCurvePR(self, name, stepsize=0, cv=True):
        """
        This function calculates the Precision Recall-curve AUC with 
        differing sample sizes

        Reminder - same K-folds as plotted in the ROC-curve

        Input:
            name = name of classifier (string)
            stepsize = size of steps
            cv = apply cross fold (warning: not suggested if training takes a long time)

        Output:
            plt = matplotlib pyplot featuring the Precision Recall curve
                of one classifier
            mean_auc = float of mean auc
        """
        model_id = self.names.index(name)
        clf = self.model_list[model_id]
        colors=['r', 'y', 'c', 'b', 'g', 'magenta', 'indigo', 'black', 'orange'] 
        recall_scale = np.linspace(0, 1, 100)
        l_folds = [(train, test) for train, test in self.splitData()]
        d_aucs = {}
        
        if stepsize == 0: # if not defined
            stepsize = int(len(l_folds[0][0])/5)
        
        for i in range(stepsize, len(l_folds[0][0]), stepsize):
            l_prec = []
            aucs = []
            for train_ix, test_ix in l_folds:
                df_test = pd.DataFrame(data={'IX': test_ix, 'Outcome': self.y[test_ix], 
                                            'Text' : self.X[test_ix]})
                df_train = pd.DataFrame(data={'IX': train_ix, 'Outcome': self.y[train_ix], 
                                            'Text' : self.X[train_ix]})
                df_sub = df_train.sample(n=i, random_state=self.seed)
                # print('HEAD:', list(df_sub['IX'].head()), len(df_sub), 'TAIL:', list(df_sub['IX'].tail()))
                estimator = clf.fit(df_sub['Text'], df_sub['Outcome'])
                probas_ = estimator.predict_proba(df_test['Text'])
                prec, tpr, thresholds = precision_recall_curve(df_test['Outcome'], probas_[:, 1])
                prec[0] = 0.0
                inter_prec = interp(recall_scale, prec, tpr)
                inter_prec[0] = 1.0 
                l_prec.append(inter_prec)
                auc = self.calculateAUC(recall_scale, inter_prec)
                aucs.append(auc)
                if cv==False:
                    break
            plt = self.plotPR(l_prec, aucs, colors[int(i/stepsize)-1], 'n=' + str(i))
            d_aucs['n=' + str(i)] = aucs
        
        # Also for final sample size
        l_prec = []
        aucs = []
        for train_ix, test_ix in l_folds:
            estimator = clf.fit(self.X[train_ix], self.y[train_ix])
            probas_ = estimator.predict_proba(self.X[test_ix])
            prec, tpr, thresholds = precision_recall_curve(self.y[test_ix], probas_[:, 1])
            prec[0] = 0.0
            inter_prec = interp(recall_scale, prec, tpr)
            inter_prec[0] = 1.0 
            l_prec.append(inter_prec)
            auc = self.calculateAUC(recall_scale, inter_prec)
            aucs.append(auc)
            if cv==False:
                break
        plt = self.plotPR(l_prec, aucs, colors[int(i/stepsize)], 'n=' + str(len(l_folds[0][0]))) # , mean_auc
        d_aucs['n=' + str(len(l_folds[0][0]))] = aucs
        
        plt.rcParams.update({'font.size': 20})
        plt.legend()
        plt.title(name + ' precision recall curve for different sample sizes')
        plt.figure(figsize=(8,6))
        return plt, d_aucs
    
    def plotPrevalencePR(self, name, cv=True, l_range_prev=[0.1, 0.25, 0.5, 0.75, 0.9]):
        """
        This function generates a precision recall curve and visualizes how 
        the precision is affected by the prevalence in the dataset by asserting
        a list with chosen prevalences (l_range_prev). This function doesn't 
        require fitted models.

        Depending on the fraction & the ratio of controls / cases, either the
        negative cases or positive cases are reduce to achieve the desired ratio.

        Important to note: Only the test set retains its initial prevalence.

        Input:
            name = name of classifier (string)
            l_range_prev = list of different prevalence fractions that are 
                measured.
            cv = apply cross fold (warning: not suggested if training takes a long time)

        Output:
            plt = matplotlib pyplot featuring the Precision Recall curve
                of one classifier
        """
        model_id = self.names.index(name)
        clf = self.model_list[model_id]
        colors=['r', 'y', 'c', 'b', 'g', 'magenta', 'indigo', 'black', 'orange'] 
        recall_scale = np.linspace(0, 1, 100)
        d_aucs = {}

        l_folds = [(train, test) for train, test in self.splitData()]
        counter = 0 
        
        df = pd.DataFrame(data={'IX': l_folds[0][0], 'Outcome': self.y[l_folds[0][0]], 
                                        'Text' : self.X[l_folds[0][0]]})
        l_range_prev.append(len(df[df['Outcome']==1])/(len(df[df['Outcome']==1])+ len(df[df['Outcome']==0])))
        
        for pref_prev in l_range_prev:
            tprs = []
            aucs = []
            for train_ix, test_ix in l_folds:
                df_test = pd.DataFrame(data={'IX': test_ix, 'Outcome': self.y[test_ix], 
                                'Text' : self.X[test_ix]})
                df = pd.DataFrame(data={'IX': train_ix, 'Outcome': self.y[train_ix], 
                                        'Text' : self.X[train_ix]})
                sum_total = (len(df[df['Outcome']==1])+ len(df[df['Outcome']==0]))
                prev_y = len(df[df['Outcome']==1])/sum_total
                prev_n = len(df[df['Outcome']==0])/sum_total
                pref_prev = pref_prev

                if len(df[df['Outcome']==1])/sum_total < pref_prev:
                    # REDUCE NEGATIVE CASES
                    factor = (prev_y/prev_n)/ (pref_prev)
                    y_neg = df[df['Outcome']==0].sample(n=math.trunc((len(df[df['Outcome']==0])*factor)*(1-pref_prev)), random_state=self.seed)
                    y_pos = df[df['Outcome']==1].sample(frac=1, random_state=self.seed)
                elif len(df[df['Outcome']==1])/sum_total > pref_prev: 
                    # REDUCE POSITIVE CASES
                    factor = (pref_prev/prev_n)/ (prev_y)
                    y_pos = df[df['Outcome']==1].sample(n=math.trunc((len(df[df['Outcome']==1])*factor)*(1-pref_prev)), random_state=self.seed)
                    y_neg = df[df['Outcome']==0].sample(frac=1, random_state=self.seed)
                else : 
                    # remain the same - default
                    y_pos = df[df['Outcome']==1]
                    y_neg = df[df['Outcome']==0]
                df_sub = pd.concat([y_pos, y_neg])
                df_sub = df_sub.sample(frac=1, random_state=self.seed) # shuffle
                estimator = clf.fit(df_sub['Text'], df_sub['Outcome'])
                probas_ = estimator.predict_proba(df_test['Text'])

                prec, tpr, thresholds = precision_recall_curve(df_test['Outcome'], probas_[:, 1])
                prec[0] = 0.0
                inter_prec = interp(recall_scale, prec, tpr)
                inter_prec[0] = 1.0 
                tprs.append(inter_prec)
                auc = self.calculateAUC(recall_scale, inter_prec)
                aucs.append(auc)
            d_aucs[str(pref_prev*100)] = aucs
            print('Prevalence (last iter):', len(y_pos) / (len(y_pos) + len(y_neg) ))
            plt = self.plotPR(tprs, aucs, colors[counter], str(pref_prev*100) + '% cases')
            counter += 1
        plt.rcParams.update({'font.size': 20})
        plt.legend()
        plt.title(name + ' performance on different proportions')
        plt.figure(figsize=(6,6))
        return plt, d_aucs