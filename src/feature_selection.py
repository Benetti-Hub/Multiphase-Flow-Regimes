'''Utility functions for the Feature Selection Notebook'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

import lightgbm as lgbm

def plot_ANOVA(imp_f_classif, save=True):
    '''
    Plot the ANOVA feature importance graph,
    If save is set to true, the image is saved
    in Plots/FeatureSelection/ANOVA.png
    
    Input:
        imp_f_classif: ANOVA importance dataframe
        
    Output:
        The plot of feature importance based on ANOVA
    '''
    
    fig, axes = plt.subplots(figsize=(35,10))  
    axes.set_title("ANOVA F-statistics",fontsize=30)
    plt.bar(range(imp_f_classif.shape[0]), imp_f_classif.F_score, align="center")
    plt.xticks(range(imp_f_classif.shape[0]), imp_f_classif['Features'], rotation='vertical', fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlim([-1, imp_f_classif.shape[0]])
    plt.grid(True)
    
    plt.ylabel('F(λ)', fontsize=30)
    plt.xlabel('Feature', fontsize=30)
    if save:
        plt.savefig(f'Plots/FeatureSelection/ANOVA.png', dpi=fig.dpi, bbox_inches='tight')
    
    return plt.show()

def generate_SFFSinfo(X, y, l, cv=5, balance_method=None):
    '''
    This function will generate additional info for the
    SFFS. In particular, it will collect F1-macro averaged
    score and the mean accuracy for each feature subset.
    
    Input:
        X: the features
        y: the targets
        l: list of selected features
        cv: number of cross validation folds
        balance_method: (optional) oversampling method chosen
    
    Output:
        A dataframe containing the collected metrics
    '''
    info_di = {}
    
    cv_info = np.zeros((cv, 2))
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    #Each feature selected by SFFS
    for i, features in enumerate(l, start=1): 
        X_step = X[features].values
        #Cross validation for each step
        for j, (train_idx, valid_idx) in enumerate(skf.split(X_step, y)):

            X_train, y_train = X_step[train_idx], y[train_idx]
            X_valid, y_valid = X_step[valid_idx], y[valid_idx]

            #Resample if required
            if balance_method:
                X_train, y_train = balance_method.fit_resample(X_train, y_train)

            model = lgbm.LGBMClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)

            cv_info[j, 0] = accuracy_score(y_valid, y_pred)
            cv_info[j, 1] = f1_score(y_valid, y_pred, average='macro')
                
        info_di[i] = {
            'feature_names'  : features,            
            'mean_acc' : np.mean(cv_info[:, 0]),
            'std_acc'  : np.std(cv_info[:, 0]),
            'mean_f1'  : np.mean(cv_info[:, 1]),
            'std_f1'   : np.std(cv_info[:, 1]),
        }   
        
    return  pd.DataFrame.from_dict(info_di).T

def plot_SFFS(scores, save=True):
    
    '''
    This function plot the results of SFFS.
    If save is set to true, the image is saved
    in Plots/FeatureSelection/SFFS.png
    
    Input:
        scores: the dataframe with SFFS results
        
    Output:
        The plot of SFFS results
    '''
    
    fig = plt.figure(figsize=(8, 6))
    plt.errorbar(scores.index, scores['mean_acc'], 
                 yerr=scores['std_acc'], label='Accuracy', linewidth=2)
    plt.errorbar(scores.index, scores['mean_f1'], 
             yerr=scores['std_f1'], label='F1_score', linewidth=2)

    plt.legend(loc='upper left')
    plt.ylabel('Metric value')
    plt.xlabel('Features used')
    
    plt.grid(True)
    
    if save:
        plt.savefig(f'Plots/FeatureSelection/SFFS.png', 
                    dpi=fig.dpi, bbox_inches='tight')
        
    return plt.show()
