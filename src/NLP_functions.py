# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:17:52 2019

@author: tdmaarseveen
"""
import collections
from collections import Counter
from inspect import signature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import pattern.nl as patNL -> outdated (are not compatible with Python 3.7+)
#import pattern.de as patDE
#import pattern.en as patEN
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance_seqs
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
from yellowbrick.text import DispersionPlot
from sklearn.feature_selection import chi2
from sklearn.metrics import precision_recall_curve

SEED = 26062019
OUTPUT_PATH = r'output_files/'


import numpy as np

class TypoCorrection(object):
    def __init__(self, vocab, thresh=0.20):
        """
        Initialize the object required for the typo correction.
        D_fix keeps track of the changes made -> and speeds up 
        the typo correction by looking back at the history.
        
        Input:
            vocab = numpy array with words relevant to the data
            thresh = threshold which indicates the difference 
                between words in a range from 0 to 1 
                (0=perfect match, 1=highly different)
        """
        self.d_fix = {}
        self.dictionary = vocab
        self.thresh = thresh
        
    def correct(self, sentence):
        """
        Corrects the sentence by applying a normalized
        Damerau Levenshtein. It is important to normalize the text
        because the word length should be taken into account!
        
        Input:
            sentence = free written text to correct for typos
                
        Output:
            new_sent = corrected free written text
        """
        new_sent = ''
        for word in sentence.split() : 
            if word in self.d_fix.keys(): 
                new_sent += self.d_fix[word] + ' '
            elif np.in1d(word, self.dictionary) == False:
                arr = normalized_damerau_levenshtein_distance_seqs(word, self.dictionary)
                if np.amin(arr) <= self.thresh:
                    result = np.where(arr == np.amin(arr))
                    self.d_fix[word] = self.dictionary[result][0]
                else :
                    self.d_fix[word] = word
                new_sent += self.d_fix[word] + ' '
            else :
                new_sent += word + ' '
        if 'ra' in new_sent: # confirmatie dat hij iets doet
            print('ra in sentence')
        return new_sent
    
    def getUniqueCorrections(self):
        return self.d_fix
    
def stemmingText(sentence, stemmer):
    """
    This function normalizes words with the kps package. 
    Stemming returns words to the base form. The base form
    is not required to be a valid word!
    
    Example: Troubling, Troubled and Trouble are all translated to 
        Troubl
    
    Input: 
        sentence = written text from an EHR record or another
            Natural Language type record (str)
        stemmer = the stemmer that is used to bring back
            words to the base form
    """  
    return ' '.join([stemmer.stem(x) for x in sentence.split(' ')])

def simpleCleaning(sentence, stem=False): # Keep in mind: this function removes numbers
    """
    Remove special characters that are not relevant to 
    the interpretation of the text
    
    Input:
        sentence = free written text from EHR record
        stem = normalize words (with a stemmer)
    Output :
        processed sentence (lemmatized depending on preference)
    """
    sticky_chars = r'([!#,.:";@\-\+\\/&=$\]\[<>\'^\*`â€™\(\)\d])'
    sentence = re.sub(sticky_chars, r' ', sentence)
    sentence = sentence.lower()
    if (stem):
        return stemmingText(sentence)
    else :
        return sentence
    
def removeAccent(text):
    """
    This function removes the accent of characters from the text.

    Variables:
        text = text to be processed
    """
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return text
    
def processArtefactsXML(entry):
    """
    Removes XML artefacts with a mapping function
    
    Input : 
        entry - Free written text entry from Electronic Health
            record (EHR)
    Output:
        entry - processed text field
    """
    correction_map ={'ã«' : 'e', 'ã¨' : 'e', 'ã¶': 'o', '\r' : ' ', '\n' : ' ', '\t': ' ', '·' : ' ', 
                     'ã©' : 'e', 'ã¯' : 'i', 'ãº':'u', 'ã³' : 'o', '\xa0' : ' '}
    for char in correction_map.keys():
        entry = entry.replace(char, correction_map[char])
    return entry

def score_binary(CL):
    """
    Calculates the dummy true en false positive rate for 
    a classifier that doesn't calculate probabilities 
    
    Input:
        CL = list of true label sorted on the 
            predictions label.
            The function sortedPredictionList can
            be used to generate such a list!
    Output:
        TPR = list with true positive rates 
        FPR = list with false positive rates 
        PRC = list with precision (PPV)
    """
    dummi = CL
    dummi = [2 if x==0 else x for x in dummi]
    dummi = [x -1 for x in dummi]
    CL.insert(0,0)
    dummi.insert(0,0)
    # Compute basic statistics:
    TP = pd.Series(CL).cumsum()
    FP = pd.Series(dummi).cumsum()
    P = sum(CL)
    N = sum(dummi)
    TPR = TP.divide(P) # sensitivity / hit rate / recall
    FPR = FP.divide(N)  # fall-out
    PRC = TP.divide(TP + FP) # precision
    return [TPR, FPR, PRC]

def binarize(value):
    """
    This function codifies the binary labels 'y' and 'n'
     to 1 and 0.
    """
    return int(value == 'y')

def func(value):
    return value[1][0]


def preset_CV10Folds(X_s):
    """
    Split the provided dataset (X_s) randomly 10 times for k-fold
    cross validation. 
    
    Input:
        X_s = pandas Series object with free written text 
            from different EHR entries. 
    Output:
        l_folds = list containing the indices for the train/test
            entries, required to contstruct the k-folds.
    """
    ss = ShuffleSplit(n_splits=10, test_size=0.5, random_state=SEED)
    l_folds = ss.split(X_s)
    return l_folds
    
def writePredictionsToFile(name, pred, true):
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

def optimalCutoff(pred, true, lbl, plot=False):
    """
    Determine the 'optimal' cutoff / threshold for classification.
    The cut-off is the balance between sensitivity and specificity
    
    Input:
        true = true label
        pred = prediction of classifier
        plot = draw the sensitivity / specificity plot ->
            whereby the cut-off is visualized as the intersection 
            between the two lines
    Output:
        cutoff = optimal cut-off (float)
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


def classifyOnLowerPrevalence(clf, X, y, positive_prev, lbl, color):
    """
    This function calculates the ROC-curve AUC on lower prevalence
    datasets. The ROC-AUC should NOT be affected by the different
    prevalence of RA.
    
    Reminder - same K-folds as plotted in the ROC-curve
    
    Input:
        clf = classifier (sklearn Pipeline object)
        X = list of free written text fields from 
            multiple EHR entries
        y = list of annotated labels from multiple EHR 
            entries
        positive_prev = prevalence of positive class (RA = True)
        lbl = name of classifier (string)
        color = specify color for the roc curve
    Output:
        plt = matplotlib pyplot featuring the Precision Recall curve
            of one classifier
        mean_auc = float of mean auc
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
        df_sub = df_sub.sample(frac=1, random_state=SEED) # shuffle
        estimator = clf.fit(X[train_ix], y[train_ix])
        probas_ = estimator.predict_proba(df_sub['XANTWOORD'])
        fpr, tpr, thresholds = metrics.roc_curve(df_sub['Outcome'], probas_[:, 1])
        tprs.append(interp(fpr_scale, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
    plt, mean_auc = plotSTD(tprs, aucs, color, lbl)
    plt.rcParams.update({'font.size': 20})
    plt.legend()
    return plt, mean_auc


def holdOutSplitPerformance(clf, lbl, X, y):
    """
    Calculates the performance of the classifier by
    performing 1 simple split rather than a k-fold split.
    
    Input:
        clf = classifier (sklearn Pipeline object)
        lbl = name of classifier
        X = list of free written text fields from 
            multiple EHR entries
        y = list of annotated labels from multiple EHR 
            entries
    """
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

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    [SKLEARN function]
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

def plotFeatureChiSquared(X_train_fold, y_train_fold, nr_features, **kwargs):
    """
    Draw a chi squared plot for the top n features
    
    Input:
        X_train_fold = array with text data (EHR entries) from
            trainingsset
        y_train_fold = labels of the trainingsset
        nr_features = specifies the nr of features to draw 
            (descending order)
        n_grams = chunk text on n_grams / motifs rather than
            on whitespace
        kwargs = arguments for feature vectorizer (TfidfVectorizer):
            ngram_range = specifies the range of the ngram features
    Output:
        plt = matplotlib pyplot showcasing the correlation for
            each of the most occurring features
    """
    tvec = TfidfVectorizer(max_features=100000,ngram_range=(1, 3))
    x_train_tfidf = tvec.fit_transform(X_train_fold)
    chi2score = chi2(x_train_tfidf, y_train_fold)[0]
    
    plt.figure(figsize=(15 ,10+ 6*(nr_features/20)))
    wscores = zip(tvec.get_feature_names(), chi2score)
    wchi2 = sorted(wscores, key=lambda x:x[1])
    topchi2 = list(zip(*wchi2[-nr_features:]))
    
    x = range(len(topchi2[1]))
    labels = topchi2[0]
    plt.barh(x ,topchi2[1], align='center', alpha=0.2)
    plt.plot(topchi2[1], x , '-o', markersize=5, alpha=0.8)
    plt.yticks(x, labels, fontsize=18)
    plt.xticks(fontsize=16)
    plt.xlabel('$\chi^2$', fontsize=18)
    plt.title('Chi-squared test of top ' + str(nr_features) + 
                  ' features', fontsize=20, fontweight='bold')
    return plt

def plotFeatureCorrelation(X_train_fold, y_train_fold, nr_features,**kwargs):
    """
    Draw a pearson correlation plot for the most occuring features.
    (Warning: this does not necesssarily draw the features with the 
      highest correlation like the FeatureImportance method!)
    
    Input:
        X_train_fold = array with text data (EHR entries) from
            trainingsset
        y_train_fold = labels of the trainingsset
        nr_features = specifies the nr of features to draw 
            (descending order)
        n_grams = chunk text on n_grams / motifs rather than
            on whitespace
    Output:
        plt = matplotlib pyplot showcasing the correlation for
            each of the most occurring features
    """
    if 'ngram_range' in kwargs.keys():
        count_vect = TfidfVectorizer(ngram_range=kwargs['ngram_range'])
    else :
        count_vect = TfidfVectorizer()
    X_train_tfidf = count_vect.fit_transform(X_train_fold) 
    plt.figure(figsize=(8,6))
    X_pd = pd.DataFrame(X_train_tfidf.toarray(), columns=count_vect.get_feature_names())
    feature_to_plot =list(X_pd.sum().sort_values(ascending=False).keys()[:nr_features])
    visualizer = FeatureCorrelation(labels=feature_to_plot, size=(750, 750), sort=True)
    visualizer.fit(X_pd[feature_to_plot], pd.Series(y_train_fold))
    ax = visualizer.ax
    ax.set_xlabel('Pearson Correlation', fontsize=18)
    ax.tick_params(labelsize=16)
    visualizer.finalize()
    plt.rcParams.update({'font.size': 55})
    plt.title('Pearson Correlation of top ' + str(nr_features) + 
              ' features', fontsize=20, fontweight='bold')
    return plt

def plotFeatureImportance(model, X_train_fold, y_train_fold, nr_features, top=True, **kwargs):
    """
    Draw a feature importance plot for the top n features.
    
    Feature importance is calculated with the leave-one-out method
    to assess the explained variance of said feature.
    
    In order to assess the most important features, the feature importance
    is calculated for every feature in the text. This isn't 
    visually pleasing however. Therefore we only draw the top n features
    (nr_features).
    
    Input:
        X_train_fold = array with text data (EHR entries) from
            trainingsset
        y_train_fold = labels of the trainingsset
        nr_features = specifies the nr of features to draw 
            (descending order)
        n_grams = chunk text on n_grams / motifs rather than
            on whitespace
        kwargs = arguments for feature vectorizer (TfidfVectorizer):
            ngram_range = specifies the range of the ngram features
            
    Output:
        plt = matplotlib pyplot showcasing the most important features for 
            the classifier
    """
    
    if 'ngram_range' in kwargs.keys():
        count_vect = TfidfVectorizer(ngram_range=kwargs['ngram_range'])
    else :
        count_vect = TfidfVectorizer()
    X_train_tfidf = count_vect.fit_transform(X_train_fold) 
    X_pd = pd.DataFrame(X_train_tfidf.toarray(), columns=count_vect.get_feature_names()) 
    feature_to_plot =list(X_pd.sum().sort_values(ascending=False).keys())
    fig = plt.figure(figsize=(10, 10))
    viz = FeatureImportances(model, labels=feature_to_plot, relative=False, absolute=True)
    viz.fit(X_pd[feature_to_plot], pd.Series(y_train_fold))
    plt.close(fig)
    if top:
        top_n_features = list(viz.features_)[-nr_features:] # nr_features
    else :
        top_n_features = list(viz.features_)[:nr_features]
    plt.figure(figsize=(8,6))
    print(len(feature_to_plot), len(y_train_fold))
    visualizer = FeatureImportances(model, labels=top_n_features,  # feature_to_plot
                                    size=(750, 750), relative=False, absolute=True)
    visualizer.fit(X_pd[top_n_features], pd.Series(y_train_fold))
    print(visualizer.feature_importances_)
    ax = visualizer.ax
    ax.set_xlabel('Feature Importance', fontsize=18)
    ax.tick_params(labelsize=16)
    visualizer.finalize()
    plt.rcParams.update({'font.size': 55})
    plt.title('Feature Importance of top ' + str(nr_features) + 
              ' features', fontsize=20, fontweight='bold')
    return plt

def plotLexicalDispersion(X, nr_features=20, **kwargs):
    """
    Draws a lexical dispersion plot which visualizes 
    the homogeneity across the corpus. 
    
    Also confirms wheter or not the data is randomized, 
    and visualizes the prevalence of features.
    
    Input:
        X = array with text data (EHR entries)
        nr_features = top n number of features to plot
        n_grams = chunksize for text processing :
            Note: chunksize refers to nr of words / not nr of 
                characters!
        kwargs = arguments for feature vectorizer (TfidfVectorizer):
            ngram_range = specifies the range of the ngram features
            stop_words = list of stopwords (e.g. in, over)
    """
    count = 0
    d = {}
    words = []
    for x in X:
        if kwargs['ngram_range'][1] != 1:
            l = [i for i in x.split(' ')]
            words.append([' '.join(l[i: i+(kwargs['ngram_range'][1])]) for i in range(len(l)) if len(l[i: i+(kwargs['ngram_range'][0])]) <= (kwargs['ngram_range'][1])])
            if len(words[-1]) == 0:
                del words[-1]
        else :
            words.append([i for i in x.split(' ')])
        count+=1
    d = np.array(words)
    if 'stop_words' in kwargs:
        count_vect = TfidfVectorizer(ngram_range=kwargs['ngram_range'], stop_words = kwargs['stop_words'])
    else :
        count_vect = TfidfVectorizer(ngram_range=kwargs['ngram_range'])
    
    X_train_tfidf = count_vect.fit_transform(X) 
    X_pd = pd.DataFrame(X_train_tfidf.toarray(), columns=count_vect.get_feature_names())
    feature_to_plot =list(X_pd.sum().sort_values(ascending=False).keys()[:nr_features]) 
    visualizer = DispersionPlot(feature_to_plot, size=(450, 450))
    ax = visualizer.ax
    ax.tick_params(labelsize=18)
    visualizer.fit(d)
    visualizer.poof()
    return

from itertools import compress 

def plotSampleDistribution(X, nr_features=50, stop_words = ['in']):
    """
    Draws a distribution of the top N words of any set (barchart)
    
    Input:
        X = array with text data (EHR entries)
        nr_features = number of features to display
    Output:
        plt = matplotlib.pyplot of the top n features 
    """
    words_to_count = [word.split(' ') for word in X]
    bool_list = [(item not in stop_words) for entry in words_to_count for item in entry]
    words_to_count = [item for entry in words_to_count for item in entry]
    words_to_count = list(compress(words_to_count, bool_list)) 
    
    counts = Counter(words_to_count) 

    labels =[ counts.most_common(nr_features)[x][0] for x in range(nr_features) ]
    values= [ counts.most_common(nr_features)[x][1] for x in range(nr_features) ]

    df = pd.DataFrame({'section':labels, 'frequency':values})
    ax = df.plot(kind='bar',  title ="Prevalence of Features", figsize=(16, 6), x='section', legend=True, fontsize=12, rot=90)
    plt.savefig('figures/feature_plot/top' + str(nr_features) + '_features_dist.png', bbox_inches='tight')
    return plt

def plotTrainTestDistribution(X_train, X_test, nr_features=50):
    """
    Draws a distribution of the top N words to assess 
    wheter the trainings/ test set are comparable!
    
    Input:
        X_train = array with text data (EHR entries) from
            trainingsset
        X_test = array with text data (EHR entries) from
            test set
        nr_features = specify the nr of features to plot
    Output:
        plt = matplotlib.pyplot of the top n features
    """
    words_to_count = [word.split(' ') for word in X_train]
    words_to_count = [item for entry in words_to_count for item in entry] # flatten
    counts_train = Counter(words_to_count) 
    
    train_labels =[ counts_train.most_common(nr_features)[x][0] for x in range(nr_features) ]
    train_values= [ counts_train.most_common(nr_features)[x][1] for x in range(nr_features) ]
    
    test_values = []
    
    words_to_count_test = [word.split(' ') for word in X_test]
    words_to_count_test = [item for entry in words_to_count_test for item in entry] # flatten
    counts_test = Counter(words_to_count_test) 
    
    for x in train_labels:
        test_values.append(counts_test.get(x))
    
    fig, ax = plt.subplots(figsize=(16,8))
    
    p1 = ax.bar([x + 0.2 for x in range(nr_features)], train_values, width=0.4, color='g', align='center')
    p2 = ax.bar([x - 0.2 for x in range(nr_features)], test_values, width=0.4, color='b', align='center')
    p3 = ax.bar(train_labels, [x - 0.2 for x in range(nr_features)], alpha=0, width=0.4, color='b', align='center')
    ax.legend((p1[0], p2[0]), ('Train', 'Test'))
    
    plt.xticks(rotation='vertical')
    plt.show()
    return

def exportTreeGraphViz(X, model, lbls, title, **kwargs):
    """
    Write the structure of the estimator to a .dot file. 
    This tree can be visualized in http://viz-js.com/
    
    Input:
        X = array with text data (EHR entries)
        nr_features = top n number of features to plot
        n_grams = chunksize for text processing :
                Note: chunksize refers to nr of words / not nr of 
                    characters!
        lbls = list of feature names
        model = tree-like classification model 
            Note: Decision Tree or subtree from Random Forest or
                Gradient Boosting
    """
    dot_data = tree.export_graphviz(model,
                feature_names= lbls, 
                class_names=['POSITIVE', 'NEGATIVE'],  
                filled=True, rounded=True, special_characters=True,
                proportion=True) 
    f = open("GraphViz/" + str(title) + ".dot", "w")
    f.write(dot_data)
    f.close()
    return

def calculateAUC(x, y):
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

def plotSTD(tprs, aucs, color, lbl, linestyle='-', lw=5, vis=1):
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
            label=lbl + r' mean kfold (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            alpha=.5, linestyle=linestyle, linewidth=lw)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.1)
        return plt, std_auc
    else :
        return
    
def plotPR(l_prec, aucs, color, lbl, linestyle='-', lw=5, std=False):
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
        mean_auc = calculateAUC(recall_scale, mean_precision)
        std_auc = np.std(aucs)
        std_precision = np.std(mean_precision, axis=0)
        if std:
            precision_upper = np.minimum(mean_precision + std_precision, 1)
            precision_lower = np.maximum(mean_precision - std_precision, 0)
            plt.fill_between(recall_scale, precision_lower, precision_upper, color=color, alpha=.1)
        plt.plot(recall_scale, mean_precision, color=color, alpha=0.6, 
                 label=lbl + r' mean kfold (AUPRC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 linestyle=linestyle, linewidth=lw)
        print(lbl + ' ' + str(mean_auc) +' (std : +/-' + str(std_auc) + ' )')
        return plt, mean_auc

def sampleSizeROC(clf, X, y, size, lbl, color):
    """
    This function calculates the ROC-curve AUC at a specified
    sample size. The ROC-AUC will likely respond to a
    major increase / decrease in data.
    
    Input:
        clf = classifier (sklearn Pipeline object)
        X = list of free written text fields from 
            multiple EHR entries
        y = list of annotated labels from multiple EHR 
            entries
        size = sample size cut-off
        lbl = name of classifier (string)
        color = specify color for the roc curve
    Output:
        plt = matplotlib pyplot featuring the ROC-curve
            of one classifier
        mean_auc = float of mean auc
    """
    fpr_scale = np.linspace(0, 1, 100)
    X = X[:size]
    y = y[:size]
    l_folds = preset_CV10Folds(X)
    tprs = []
    aucs = []
    for train_ix, test_ix in l_folds:
        y_test = y[test_ix]
        df = pd.DataFrame(data={'IX': test_ix, 'Outcome': y_test, 
                                'XANTWOORD' : X[test_ix]})
        estimator = clf.fit(X[train_ix], y[train_ix])
        probas_ = estimator.predict_proba(df['XANTWOORD'])
        fpr, tpr, thresholds = metrics.roc_curve(df['Outcome'], probas_[:, 1])
        tprs.append(interp(fpr_scale, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
    plt, std_aucs =plotSTD(tprs, aucs, color, lbl)
    plt.rcParams.update({'font.size': 20})
    plt.legend()
    return plt, aucs

def sampleSizePR(clf, X, y, size, lbl, color):
    """
    This function calculates the PR-curve AUC at a specified
    sample size. The PR-AUC will likely respond to a
    major increase / decrease in data.
    
    Input:
        clf = classifier (sklearn Pipeline object)
        X = list of free written text fields from 
            multiple EHR entries
        y = list of annotated labels from multiple EHR 
            entries
        size = sample size cut-off
        lbl = name of classifier (string)
        color = specify color for the roc curve
    Output:
        plt = matplotlib pyplot featuring the Precision Recall curve
            of one classifier
    """
    recall_scale = np.linspace(0, 1, 100)
    X = X[:size]
    y = y[:size]
    l_folds = preset_CV10Folds(X)
    l_prec = []
    aucs = []
    for train_ix, test_ix in l_folds:
        y_test = y[test_ix]
        df = pd.DataFrame(data={'IX': test_ix, 'Outcome': y_test, 
                                'XANTWOORD' : X[test_ix]})
        estimator = clf.fit(X[train_ix], y[train_ix])
        probas_ = estimator.predict_proba(df['XANTWOORD'])
        prec, tpr, thresholds = precision_recall_curve(df['Outcome'], probas_[:, 1])
        prec[0] = 0.0
        inter_prec = interp(recall_scale, prec, tpr)
        inter_prec[0] = 1.0 
        l_prec.append(inter_prec)
        auc = calculateAUC(recall_scale, inter_prec)
        aucs.append(auc)
    plt, mean_auc = plotPR(l_prec, aucs, color, lbl)
    #plt, mean_auc = bc.plotSTD(tprs, aucs, color, lbl)
    plt.rcParams.update({'font.size': 20})
    plt.legend()
    return plt, aucs

def entriesPatientMerge(pat_df, id_column, X_column, y_column=""):
    """
    Merges the entries into one entry per patient (according to the id_column)
    
    In other words, 
    This function merges all text (X_column) on the id_column (patient id)
    
    Input: 
        id_column = string indicating column with patient id
        X_column = string indicating column with free text field
        y_column = string indicating column with label (if it exists)
        
    Output: 
        dictionary with compressed data (1 row with all patient info)
    """
    field = ''
    for i in pat_df[X_column]:
        field += " " + i + " "
    if y_column!="":
        return {X_column: field, id_column : pat_df[id_column].iloc[0], y_column : pat_df[y_column].iloc[0]}
    else :
        return {X_column: field, id_column : pat_df[id_column].iloc[0]}

def mergeOnColumn(df, id_column, X_column, y_column=""):
    """
    This function creates a new dataframe, where all entries of an individual
    patient are merged into one row. Resulting in a patient based table.
    
    Input:
        df = entry based dataframe (multiple entries per patient)
        id_column = string indicating column with patient id
        X_column = string indicating column with free text field 
        y_column = string indicating column with label (if it exists)
        
    Ouput:
        df_ult = patient based dataframe (one summary per patient)
    """
    if y_column != "":
        df_ult = pd.DataFrame(columns=[X_column,  id_column, y_column])
    else : 
        df_ult = pd.DataFrame(columns=[X_column,  id_column])

    for pat in df[id_column].unique():
        pat_df = df[df[id_column]==pat]
        if y_column != "":
            df_ult = df_ult.append(entriesPatientMerge(pat_df, id_column, X_column, y_column), ignore_index=True)
        else : 
            df_ult = df_ult.append(entriesPatientMerge(pat_df, id_column, X_column), ignore_index=True)
    return df_ult