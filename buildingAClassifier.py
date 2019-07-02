# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:17:52 2019

@author: tdmaarseveen
"""
import collections
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pattern.nl as patNL
import re
from scipy import stats
from scipy import interp
from sklearn.model_selection import learning_curve
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit

SEED = 26062019
OUTPUT_PATH = r'output_files/'

class CustomBinaryModel(object):
    def __init__(self, targets):
        self.targets = targets
        
    def setTargets(self, targets):
        self.targets = targets
    
    def getTargets(self):
        return self.targets
    
    def predict(self, report):
        regexp = re.compile(r'\b('+r'|'.join(self.targets)+r')\b')
        if regexp.search(report):
            return 'y'
        else :
            return 'n'

def lemmatizingText(sentence):
    """
    This function normalizes words with the pattern.nl package. 
    Lemmatisation returns words to the base form.

    Example: Walking, Walks and Walked are all translated to 
        Walk
    """
    return ' '.join(patNL.Sentence(patNL.parse(sentence, lemmata=True)).lemmata)

def simpleCleaning(sentence, lemma=False): # Keep in mind: this function removes numbers
    sticky_chars = r'([!#?,.:";@\-\+\\/&=$\]\[<>\'^\*`â€™\(\)\d])'
    sentence = re.sub(sticky_chars, r' ', sentence)
    sentence = sentence.lower()
    if (lemma):
        return lemmatizingText(sentence)
    else :
        return sentence

def score_binary(CL, inclFirst = True ):
    dummi = CL
    dummi = [2 if x==0 else x for x in dummi]
    dummi = [x -1 for x in dummi]
    if (inclFirst):
        CL.insert(0,0)
        dummi.insert(0,0)
    # Compute basic statistics:
    TP = pd.Series(CL).cumsum()
    FP = pd.Series(dummi).cumsum()
    P = sum(CL)
    N = sum(dummi)
    TPR = TP.divide(P) # sensitivity / hit rate / recall
    FPR = FP.divide(N)  # fall-out
    return TPR, FPR

def binarize(value):
    return int(value == 'y')

def func(value):
    return value[1][0]

def sortedPredictionList(pred, y_test):
    d_perf_dt = {}
    b_pred = []
    for x in pred:
        if x == 'y':
            b_pred.append(1)
        elif x == 'n':
            b_pred.append(0)
    count = 0
    for i in range(0,len(y_test)):
        d_perf_dt[count] = [b_pred[count], binarize(y_test[count])]
        count += 1
    orderedDict = collections.OrderedDict(sorted(d_perf_dt.items(), key=lambda k: func(k), reverse=True))
    l_sorted_pred= []
    l_sorted_true = []
    for x in orderedDict.items():
        l_sorted_pred.append(x[1][0])
        l_sorted_true.append(x[1][1]) 
    return l_sorted_true

def preset_CV10Folds(X_s):
    ss = ShuffleSplit(n_splits=10, test_size=0.5, random_state=SEED)
    l_folds = ss.split(X_s)
    return l_folds
    
def writePredictionsToFile(name, pred, true):
    """
    Write predictions of the classifier to a simple CSV file. 
    These files can be processed in pROC for the Delong test
    """
    d = {'PRED': pred, 'TRUE': true}
    df = pd.DataFrame(data=d)
    df.to_csv(OUTPUT_PATH + 'pred' + name.replace(" ", "") + '.csv', sep='|', index=False)
    return

def plotFolds(clf, X, y, l_folds, color, lbl):
    """
    y_train should be binarized
    """
    tprs = []
    aucs = []
    fold = 0
    fpr_scale = np.linspace(0, 1, 100)
    d_aucs = {}
    for train_index, test_index in l_folds:
        #print(train_index, test_index)
        estimator = clf.fit(X[train_index], y[train_index])
        probas_ = estimator.predict_proba(X[test_index])
        fpr, tpr, thresholds = metrics.roc_curve(y[test_index], probas_[:, 1])
        tprs.append(interp(fpr_scale, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        d_aucs[roc_auc] = [probas_[:,1], interp(fpr_scale, fpr, tpr), 
               estimator, test_index, train_index]# tpr # interp(fpr_scale, fpr, tpr)
        
        #plt.plot(fpr, tpr, lw=1, alpha=0.3,
        #        label='ROC fold %d (AUC = %0.2f)' % (fold, roc_auc), color=color)
        fold += 1
        
    
    aucs.sort()
    middleIndex = round((len(aucs) - 1)/2) # normally 5 -> if 10 fold
    #print(lbl + ': ' + str(aucs[middleIndex]))
    medianModel = d_aucs[aucs[middleIndex]]
    #plt = classifyOnLowerPrevalence(clf, medianModel, X_train, y_train, .2)
    foldTrueLbl = y[medianModel[3]]
    writePredictionsToFile(lbl, medianModel[0], foldTrueLbl)
    plt, mean_auc = plotSTD(tprs, aucs, color, lbl)
    return plt, mean_auc, aucs, medianModel

def classifyOnLowerPrevalence(clf, X, y, positive_prev, lbl, color):
    """
    Test the performance of the classifier on a unbalanced test set
    
    Reminder - same K-folds as plotted in the ROC-curve
    
    Variables:
        positive_prev = prevalence of positive class (RA = True)
    """
    fpr_scale = np.linspace(0, 1, 100)
    l_folds = preset_CV10Folds(X)
    tprs = []
    aucs = []
    for train_ix, test_ix in l_folds:
        y_test = y[test_ix]
        df = pd.DataFrame(data={'IX': test_ix, 'Outcome': y_test, 
                                'XANTWOORD' : X[test_ix]})
        y_pos = df[df['Outcome']==1].sample(frac=positive_prev, random_state=SEED)
        if round(len(df[df['Outcome']==1])-len(df[df['Outcome']==0])) < 0:
            y_neg = df[df['Outcome']==0].sample(n= len(df[df['Outcome']==0]) + \
                      round(len(df[df['Outcome']==1])-len(df[df['Outcome']==0])), random_state=SEED)
        else :
            y_neg = df[df['Outcome']==0].sample(n= len(df[df['Outcome']==0]), random_state=SEED)
        df_sub = pd.concat([y_pos, y_neg])
        #print(df_sub.index)
        #print(df_sub['Outcome'].value_counts()) # -> verify if it works
        df_sub = df_sub.sample(frac=1, random_state=SEED) # shuffle
        estimator = clf.fit(X[train_ix], y[train_ix])
        probas_ = estimator.predict_proba(df_sub['XANTWOORD'])
        #print(probas_[:,1])
        fpr, tpr, thresholds = metrics.roc_curve(df_sub['Outcome'], probas_[:, 1])
        tprs.append(interp(fpr_scale, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
    #print(df_sub['Outcome'].value_counts()) 
    plt, mean_auc = plotSTD(tprs, aucs, color, lbl)
    plt.rcParams.update({'font.size': 20})
    plt.legend()
    return plt, mean_auc

def AUCtoCI(auc, std_auc, alpha=.95):
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=std_auc)
    ci[ci > 1] = 1
    return ci 

def plotSTD(tprs, aucs, color, lbl, linestyle='-', lw=5):
    """
    Plot the standard deviation of the ROC-curves
    """
    #print(medianModel)
    #print(medianModel[3])
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr [-1] = 1.0
    
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color=color,
        label=lbl + r' mean kfold (AUC = %0.2f $\pm$ %s)' % (mean_auc, std_auc),
        alpha=.5, linestyle=linestyle, linewidth=lw)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.1)
    print(lbl + ' ' + str(mean_auc) +' (std : +/-' + str(std_auc) + ' )')
    return plt, std_auc

def plotTrainSplit(clf, X_train, y_train, color, lbl, lw=3):
    """
    y_train should be binarized
    """
    pred_t = clf.predict_proba(X_train)[:,1]
    fpr_t, tpr_t, threshold_t = metrics.roc_curve(y_train, list(pred_t), pos_label=1)
    auc = np.trapz(tpr_t,fpr_t)
    plt.plot(fpr_t, tpr_t, color, lw=lw, label = lbl + ' (AUC = %0.2f' % (auc) + ')', alpha=0.1)
    return plt

def plotBinaryROC(clf, lbl, X, y, l_folds, color):
    """
    Plot pseudo AUC for models that don't predict a probability
    """
    l_folds = preset_CV10Folds(X)
    tprs = []
    aucs = []
    fold = 0
    fpr_scale = np.linspace(0, 1, 100)
    d_aucs = {}
    for train_index, test_index in l_folds:
        estimator = clf.fit(X[train_index], y[train_index])
        pred = estimator.predict(X[test_index])
        l_sorted_true = sortedPredictionList(pred, y[test_index])
        tpr, fpr = score_binary(l_sorted_true)
        roc_auc = np.trapz(tpr,fpr)
        tprs.append(interp(fpr_scale, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        d_aucs[roc_auc] = [np.array([binarize(val) for val in pred]), \
              interp(fpr_scale, fpr, tpr), 
              estimator, test_index, train_index]
        fold += 1
    aucs.sort()
    middleIndex = round((len(aucs) - 1)/2) # normally 5 -> if 10 fold
    medianModel = d_aucs[aucs[middleIndex]]
    #print(lbl + ': ' + str(aucs[middleIndex]))
    foldTrueLbl = np.array([binarize(val) for val in y[medianModel[3]]])
    #plotTrainSplit(clf, X[medianModel[4]], np.array([binarize(val) for val in y[medianModel[4]]]), 
    #                   color, lbl)
    writePredictionsToFile(lbl, medianModel[0], foldTrueLbl)
    ## onderstaande ook nodig?
    plt, mean_auc = plotSTD(tprs, aucs, color, lbl)
    return plt, mean_auc, aucs, medianModel

def plotCustomModelROC(clf, X, y, l_folds, lbl, color, linestyle='-'):
    l_folds = preset_CV10Folds(X)
    tprs = []
    aucs = []
    fold = 0
    fpr_scale = np.linspace(0, 1, 100)
    d_aucs = {}
    for train_index, test_index in l_folds:
        l_context= [clf.predict(str(x)) for x in X[test_index]]
        pred = [l_context[x][0] for x in range(len(l_context))]
        l_sorted_true = sortedPredictionList(pred, y[test_index])
        tpr, fpr = score_binary(l_sorted_true)
        roc_auc = np.trapz(tpr,fpr)
        tprs.append(interp(fpr_scale, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        d_aucs[roc_auc] = [np.array([binarize(val) for val in pred]), \
              interp(fpr_scale, fpr, tpr), 
              clf, test_index, train_index]
        fold += 1
    aucs.sort()
    middleIndex = round((len(aucs) - 1)/2) # normally 5 -> if 10 fold
    medianModel = d_aucs[aucs[middleIndex]]
    #print(lbl + ': ' + str(aucs[middleIndex]))
    foldTrueLbl = np.array([binarize(val) for val in y[medianModel[3]]])
    writePredictionsToFile(lbl, medianModel[0], foldTrueLbl)
    plt, mean_auc = plotSTD(tprs, aucs, color, lbl, linestyle)
    return plt, mean_auc, aucs, medianModel
    
def holdOutSplitPerformance(clf, lbl, X, y):
    ss = ShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    l_folds = ss.split(X)
    train_ix, test_ix = l_folds
    estimator = clf.fit(X[train_ix], y[train_ix])
    pred = estimator.predict_proba(X[test_ix])[:,1]
    y_b = y[test_ix].copy()
    for i in range(len(y[test_ix])): # MAKE BINARY (y = 1, n = 0)
        y_b[i] = int(y_b[i] == 'y')
        fpr, tpr, threshold = metrics.roc_curve(list(y_b), list(pred), pos_label=1)
        writePredictionsToFile(lbl, pred, y_b)
    return
    
def plotCrossValidationROC(models, title, lbls, X, y, l_folds, ref_auc):
    """ 
    models = list of Pipelines (sklearn)
    """
    colors = ['c', 'b', 'g', 'magenta', 'indigo', 'black']
    d_aucs = {}
    fitted_models = {}
    for x in range(len(models)):
        
        l_folds = preset_CV10Folds(X)
        
        #holdOutSplitPerformance(models[x], lbls[x], X[train_index], 
        #                        X[test_index], y[train_index], y[test_index])
        
        #plt.plot(fpr, tpr, colors[x], lw=lw, label = lbls[x] + ' (AUC = %0.2f' % (auc) + '; p = %s)' % (p_val[x]))
        plt, mean_auc, aucs, medianModel = plotFolds(models[x], X, 
                    np.array([binarize(val) for val in y]), l_folds, colors[x], lbls[x])
        d_aucs[lbls[x]] = aucs
        fitted_models[lbls[x]] = medianModel
        #plotTrainSplit(models[x], X[medianModel[4]], np.array([binarize(val) for val in y[medianModel[4]]]), 
        #              colors[x], lbls[x])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.rcParams.update({'font.size': 12})
    plt.legend(loc = 'lower right')
    plt.rcParams.update({'font.size': 55})
    plt.ylabel('Sensitivity (TPR)')
    plt.xlabel('1 - Specificity (FPR)')
    return plt, d_aucs, fitted_models

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
        
    Code from sklearn tutorial 
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt