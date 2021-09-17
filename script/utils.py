import numpy as np
import pandas as pd
import math
import os
from pickle import load

from scipy.constants import g as gravity
from scipy import optimize

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

from tensorflow.keras.models import load_model as tf_load 
import tensorflow as tf
import torch

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def bronze_to_silver(df):

    '''
    This function will perform basic preprocessing on the input data.
    Mainly the scope of this function is to check if the neessary columns
    are in the dataset, and apply the dictionary to the labels
    '''
    df = df[['P', 'T', 'DenL', 'DenG', 'VisL', 'VisG', 'ST', 
             'ID', 'Roughness', 'Ang', 'Vsl', 'Vsg', 'Flow_label']]
    
    di = {'A': 0, 'DB':1, 'I':2, 'SW':3, 'SS':4, 'B':5,
          'Annular': 0, 'Dispersed':1, 
          'Intermittent':2, 'Stratified':3} 
    
    df = df.replace({"Flow_label": di})     
    return df.astype('float64')

def silver_to_gold(df_, balance_method=None, 
                   kept_columns=None, 
                   gaussian_noise=False):

    '''
    This function will transform silver data to gold. Here the possibility of
    selecting only a handful of features, oversampling minority classes and
    adding gaussian noise (std=0.01) to the input features is considered.
    
    Input:
    
        pd.DataFrame df_ :    dataframe to modify
        bool balance_method : oversampling method from imblearn library
        list kept_columns :   features to keep after the transformation
        bool gaussian_noise:  weather to add a gaussian noise 
        
    Output:
        
        pd.DataFrame df : modified dataset
    
    '''    
    
    #Copy the dataset (avoid changing the original)
    df = df_.copy()
    
    #Add gaussian noise:
    if gaussian_noise:
        df.iloc[:,:-3] = df.iloc[:,:-3] * np.random.normal(
                                          loc=1, 
                                          scale=0.005, 
                                          size=df.iloc[:,:-3].shape)
        
    #Add oversampling method:    
    if balance_method:
        X, y = df.iloc[:,:-1].values, df.iloc[:,[-1]].values.ravel()      
        X, y = balance_method.fit_resample(X, y)
        df = pd.DataFrame(data=np.column_stack((X, y)), columns=df.columns)
            
    #Add feature engineering
    df = generate_features(df) 
        
    #Select only relevant columns:
    if kept_columns:
        df = df[kept_columns]

    #Refactor the target column as int32
    df['Flow_label'] = df['Flow_label'].astype('int32')
        
    return df
    
    
def bronze_to_gold(df, balance_method=None, kept_columns=None, gaussian=False):
    
    '''
    This function will just call the bronze to silver and
    silver to gold in sequence.
    '''
    return silver_to_gold(bronze_to_silver(df), 
                          balance_method=balance_method, 
                          kept_columns=kept_columns,
                          gaussian_noise=gaussian)

def feature_combination(estimator, X, y, balance_classes=True, cv=5):
    '''
    Function generated to train a model using only a restriced number of features  
    Note that "balance_classes=True" is only valid for classification problems
    
    Input:
        object estimator      Sklearn ML model
        np.array X            Training features array
        np.array y            Training labels
        bool balance_classed  perform class balancing
        int cv                number of cross validation
        
    Output:
        list infos            list of accuracy, F1 score etc 
                              of the trained model
        
        Calculated metrics:
        accuracy, precision, recall, F1_score
    
    '''    
    
    infos = np.zeros((cv, 4))
    c_matrix = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype='int32')
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)   
    for [train_index, test_index], i in zip(skf.split(X, y), range(cv)):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if balance_classes:
            oversampling_model = SMOTE(random_state=0, k_neighbors=5)
            X_train, y_train = oversampling_model.fit_resample(X_train, y_train)
            
        estimator.fit(X_train, y_train)
        y_pred       = estimator.predict(X_test)
        infos[i, 0]  = accuracy_score(y_test, y_pred)
        infos[i, 1:] = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)[:3]
        c_matrix += confusion_matrix(y_test, y_pred)
    
    return np.mean(infos, axis=0), c_matrix
    
def print_performance(y_true, y_pred):
    '''
    This function will evaulate the performance of a model printing out the F1 score
    and the accuracy for all the available classes (as well as their mean)
    '''
    
    print('Mean Accuracy: ', accuracy_score(y_true, y_pred))
    print('Mean F1 score: ', precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=1)[2])
    
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    print('\n\nSingle class Accuracy: ', cm.diagonal())
    print('Single class F1 score: ', precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=1)[2])
    
    print('\nClassification Report: \n', classification_report(y_true, y_pred, zero_division=1))
    
    print('\nConfusion Matrix:\n', confusion_matrix(y_true, y_pred))

'''
Dataset processing.
The following functions will deal with the creation of additional features for the 
dataset, as well as the data cleaning (missing values etc)
'''

def generate_features(df):
    '''
    This function will generate some usefull quantities to
    help the ML model to generate a better overall solution
    note that these quantities are the one I deemed interesting,
    by no means they are all usefull.
    '''
    #Angles related transofrmation
    df['SinAng'] = np.sin(df['Ang']/360*(2*math.pi))
    df['CosAng'] = np.cos(df['Ang']/360*(2*math.pi))
    
    #Reynolds Numbers
    df['ReL'] = df['ID']*df['DenL']*df['Vsl']/df['VisL']
    df['ReG'] = df['ID']*df['DenG']*df['Vsg']/df['VisG']
    
    #Fanning
    df['FanningL'] = 16/df['ReL']
    df.loc[df['ReL']>2300, 'FanningL'] = 0.0625/((np.log((150.39/(df['ReL']**0.98865))-(152.66/(df['ReL']))))**2)
    
    df['FanningG'] = 16/df['ReG']
    df.loc[df['ReG']>2300, 'FanningG'] = 0.0625/((np.log((150.39/(df['ReG']**0.98865))-(152.66/(df['ReG']))))**2)   
    
    #Froude Number
    df['FrL'] = ((df['DenL']/((df['DenL']-df['DenG'])*gravity*df['ID']))**0.5)*df['Vsl']
    df['FrG'] = ((df['DenG']/((df['DenL']-df['DenG'])*gravity*df['ID']))**0.5)*df['Vsg']
    
    df['NL'] = df['FrL']*df['FanningL']**0.5
    
    #LM and Y
    df['X_LM'] = ((df['FanningL']/df['FanningG'])**0.5)*(df['FrL']/df['FrG'])
    df['X_LM_2'] = df['X_LM']**2
    df['Y']  = df['SinAng']/(2*df['FanningG']*(df['FrG'])**2)
    
    #Dimensionless numbers
    df['We'] = df['DenL']*df['Vsl']**2*df['ID']/df['ST']
    df['Eo'] = (df['DenL']-df['DenG'])*gravity*(df['ID']**2)/df['ST']
    
    df['HU'] = calculate_holdup(df['X_LM'].values, df['Y'].values)
    
    #Parameters for Maps:        
    df['K_G'] = df['FrG']*(df['ReL'])**0.5
    df['T_TB'] = (2*df['FanningL'])*df['FrL']**2
    
    #Remove Dimensional Quantities

    df.drop(columns=['P', 'T', 'DenL', 'DenG', 'ID', 
                     'VisL', 'VisG', 'ST',
                     'Vsl', 'Vsg', 'Roughness',
                     'FanningL','FanningG', 'CosAng', 'SinAng'], inplace=True, errors='ignore')

    #Put the Target as the last column    
    if 'Flow_label' in df.columns:
        df = df[[col for col in df.columns if col != 'Flow_label']+['Flow_label']]
    
    return df

def calculate_holdup(X, Y):    
    
    alpha = np.zeros(len(X))
    for i in range(len(X)):
        alpha[i] = optimize.brentq(
            lambda a, X=X[i], Y=Y[i]: (1+75*a)/((1-a)**2.5 *a) -X**2/a**3 -Y, 
            a=0.00001, b=0.99999, 
            args=(X[i], Y[i]))
        
    return alpha

def generate_meta(df):
    
    X_meta = df[['Ang', 'ID', 'DenL', 'DenG']].values
    
    return X_meta
    
    
def load_model(folder : str, model_type : str):
    '''
    Simple function to load a model type
    given a folder directory
    '''
    di = {"LightGBM"       : "LightGBM",
          "RandomForest"   : "RandomForest",
          "XGBoost"        : "XGBoost",
          "ExtraTrees"     : "ExtraTrees",
          "TabNet"         : "TabNet",
          "SVC"            : "SVC",
          "MLP"            : "TF_models",
          "StandardScaler" : "StandardScaler"}
    try:
        model_class = di[model_type]
    except:
        raise TypeError("Model type specified not found," 
                        f"possibile values are: \n {di.keys()}")
    
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)   
        if model_type=="MLP":
            with tf.device('/cpu:0'):
                return tf_load(f'{folder}/{di[model_type]}')
        elif model_type=="TabNet":
            return torch.load(f'{folder}/{di[model_type]}.pth')
        elif di[model_type] in file_path:
            return load(open(file_path, 'rb'))
                
    print(f"Model type: {di[model_type]} not found in directory.\n Files found: \n", os.listdir(folder))
    
    