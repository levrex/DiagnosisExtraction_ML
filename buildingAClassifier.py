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
from sklearn.metrics import confusion_matrix

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
    
def processArtefactsXML(entry):
    """
    Apply this function if your data contains XML artefacts 
    """
    correction_map ={'ã«' : 'e', 'ã¨' : 'e', 'ã¶': 'o', '\r' : ' ', '\n' : ' ', '\t': ' ', '·' : ' ', 
                     'ã©' : 'e', 'ã¯' : 'i', 'ãº':'u', 'ã³' : 'o', '\xa0' : ' '}
    for char in correction_map.keys():
        entry = entry.replace(char, correction_map[char])
    return entry

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

def assessPerformance_proba(estimator, X_test, y_test, fold, tprs, aucs, d_aucs={}): 
    """
    
    Calculates the true positive rate, the false positive rate and 
    the Area under the Curve for the Receiver Operator Curve (ROC)
    for the provided classifier (that calculates probabilities!)
    
    tprs = list with true positive rates 
    aucs = list with area under the curves
    d_aucs = dictionary with the following attributes:
        0. the predictions (proba), 
        1. true positive rate
        2. classifier (a.k.a. model)
        3. test index  (of fold)
        4. train index (of fold)
    X_test = values of test set
    y_test = actual label of the test set
    Disclaimer: you could also use this function to assess the auc 
        of the trainingsset in addition to the testset. If you are
        interested if the fitted model covers the whole trainingsset.
        
    """
    fpr_scale = np.linspace(0, 1, 100)
    probas_ = estimator.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    aucs.append(roc_auc)
    tprs.append(interp(fpr_scale, fpr, tpr))
    tprs[-1][0] = 0.0
    d_aucs[roc_auc] = [probas_[:,1], \
          interp(fpr_scale, fpr, tpr), 
          estimator, fold[0], fold[1]]
    return tprs, aucs, d_aucs

def optimalCutoff(pred, true, lbl, plot=False):
    """
    Input:
    true = true label
    pred = prediction of classifier
    
    Description:
    Determine the optimal cutoff / threshold for classification.
    The optimal cut-off is the balance between sensitivity and specificity
    """
    fpr, tpr, thresholds = metrics.roc_curve(true, pred)
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
    cutoff = roc.ix[(roc.tf-0).abs().argsort()[:1]]['thresholds']
    if plot == True:
        fig, ax = plt.subplots()
        plt.plot(roc['tpr'])
        plt.plot(roc['1-fpr'], color = 'red')
        plt.xlabel('1-False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        ax.set_xticklabels([])
        plt.savefig('figures/cutoff_plot/CutOffPlot_' + lbl + '.png')
        print(roc.ix[(roc.tf-0).abs().argsort()[:1]])
    return cutoff

def plotFolds(clf, X, y, l_folds, color, lbl):
    """
    y_train should be binarized
    """
    tprs, aucs = [], []
    tprs_t, aucs_t = [], []
    d_aucs = {}
    for train_index, test_index in l_folds:
        #print(train_index, test_index)
        
        fold = [test_index, train_index]
        estimator = clf.fit(X[train_index], y[train_index])
        tprs, aucs, d_aucs = assessPerformance_proba(estimator, X[test_index], 
                                              y[test_index], fold, tprs, aucs, 
                                              d_aucs)
        tprs_t, aucs_t = assessPerformance_proba(estimator, X[train_index], 
                                              y[train_index], fold, tprs_t, 
                                              aucs_t)[:2]
    aucs.sort()
    middleIndex = round((len(aucs) - 1)/2) # normally 5 -> if 10 fold
    #print(lbl + ': ' + str(aucs[middleIndex]))
    medianModel = d_aucs[aucs[middleIndex]]
    #plt = classifyOnLowerPrevalence(clf, medianModel, X_train, y_train, .2)
    foldTrueLbl = y[medianModel[3]]
    writePredictionsToFile(lbl, medianModel[0], foldTrueLbl)
    cut_off = optimalCutoff(medianModel[0], foldTrueLbl, lbl, False)
    medianModel.append(cut_off)
    plt, mean_auc = plotSTD(tprs, aucs, color, lbl)
    plotSTD(tprs_t, aucs_t, color, 'Train-score ' + lbl, '-', 5, 0)
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

def plotSTD(tprs, aucs, color, lbl, linestyle='-', lw=5, vis=1):
    """
    Plot the standard deviation of the ROC-curves
    
    Input:
        tprs = list of true positive rates per iteration
        aucs = list of area under the curve per iteration
        lbl = classifier
        vis = visualize
    """
    #print(medianModel)
    #print(medianModel[3])
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

def plotTrainSplit(clf, X_train, y_train, color, lbl, lw=3):
    """
    y_train should be binarized
    """
    pred_t = clf.predict_proba(X_train)[:,1]
    fpr_t, tpr_t, threshold_t = metrics.roc_curve(y_train, list(pred_t), pos_label=1)
    auc = np.trapz(tpr_t,fpr_t)
    plt.plot(fpr_t, tpr_t, color, lw=lw, label = lbl + ' (AUC = %0.2f' % (auc) + ')', alpha=0.1)
    return plt

def assessPerformance(estimator, X_test, y_test, fold, tprs, aucs, d_aucs={}): 
    """
    Calculates the true positive rate, the false positive rate and 
    the Area under the Curve for the Receiver Operator Curve (ROC) 
    
    tprs = list with true positive rates 
    aucs = list with area under the curves
    d_aucs = dictionary with aucs/ tprs/ estimator and test/train fold 
        for every iteration
    X_test = values of test set
    y_test = actual label of the test set
    Disclaimer: you could also use this function to assess the auc 
        of the trainingsset in addition to the testset. If you are
        interested if the fitted model covers the whole trainingsset.
    """
    fpr_scale = np.linspace(0, 1, 100)
    pred = estimator.predict(X_test)
    l_sorted_true = sortedPredictionList(pred, y_test)
    tpr, fpr = score_binary(l_sorted_true)
    roc_auc = np.trapz(tpr,fpr)
    aucs.append(roc_auc)
    tprs.append(interp(fpr_scale, fpr, tpr))
    tprs[-1][0] = 0.0
    d_aucs[roc_auc] = [np.array([binarize(val) for val in pred]), \
          interp(fpr_scale, fpr, tpr), 
          estimator, fold[0], fold[1]]
    return tprs, aucs, d_aucs

def plotBinaryROC(clf, lbl, X, y, l_folds, color):
    """
    Plot pseudo AUC for models that don't predict a probability
    """
    l_folds = preset_CV10Folds(X)
    tprs, aucs = [], []
    tprs_t, aucs_t = [], []
    d_aucs = {}
    for train_index, test_index in l_folds:
        fold = [test_index, train_index]
        estimator = clf.fit(X[train_index], y[train_index])
        tprs, aucs, d_aucs = assessPerformance(estimator, X[test_index], 
                                              y[test_index], fold, tprs, aucs, 
                                              d_aucs)
        tprs_t, aucs_t = assessPerformance(estimator, X[train_index], 
                                              y[train_index], fold, tprs_t, aucs_t, 
                                              {})[:2]
    aucs.sort()
    middleIndex = round((len(aucs) - 1)/2) # normally 5 -> if 10 fold
    medianModel = d_aucs[aucs[middleIndex]]
    foldTrueLbl = np.array([binarize(val) for val in y[medianModel[3]]])
    writePredictionsToFile(lbl, medianModel[0], foldTrueLbl)
    cut_off = optimalCutoff(medianModel[0], foldTrueLbl, lbl)
    medianModel.append(cut_off)
    plt, mean_auc = plotSTD(tprs, aucs, color, lbl)
    plotSTD(tprs_t, aucs_t, color, 'Train-score ' + lbl, '-', 5, 0)
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

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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