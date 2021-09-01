'''Utility module for feature engineering'''
import warnings

import math
from scipy.constants import g as gravity
from scipy import optimize

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

#Dictionary for string-int labels
di = {'A': 0, 'DB':1, 'I':2, 'SW':3, 'SS':4, 'B':5,
      'Annular': 0, 'Dispersed':1,
      'Intermittent':2, 'Stratified':3}

def bronze_to_gold(df_,
                   balance_method=None,
                   kept_columns=None,
                   gaussian_noise=False):

    '''
    This function will transform the dataframe adding relevant features.
    The function provide the possibility of selecting a subset of features
    (kept_columns), oversampling minority classes (balance_method) and
    adding gaussian noise (gaussian_noise).

    Input:

        pd.DataFrame df_ :    starting raw dataframe
        bool balance_method : oversampling method from imblearn library
        list kept_columns :   features to keep after the transformation
        bool gaussian_noise:  weather to add a gaussian noise

    Output:

        pd.DataFrame df : modified dataframe
    '''

    #Copy the dataset (avoid changing the original)
    df = df_.copy()

    df = df[['DenL', 'DenG', 'VisL', 'VisG', 'ST',
             'ID', 'Roughness', 'Ang', 'Vsl', 'Vsg',
             'Flow_label']]

    df = df.replace({"Flow_label": di})

    #Add oversampling method:
    if balance_method:
        X, y = df.iloc[:,:-1].values, df.iloc[:,[-1]].values.ravel()
        X, y = balance_method.fit_resample(X, y)
        df = pd.DataFrame(data=np.column_stack((X, y)), columns=df.columns)

    #Add gaussian noise:
    if gaussian_noise:
        df.iloc[:,:-3] = df.iloc[:,:-3] * np.random.normal(
                                          loc=1,
                                          scale=0.01,
                                          size=df.iloc[:,:-3].shape)

    #Add features
    df = generate_features(df)

    #Select only relevant columns:
    if kept_columns:
        df = df[kept_columns]

    #Refactor the target column as int32
    df['Flow_label'] = df['Flow_label'].astype('int32')

    return df

def generate_features(df):
    '''
    The function will replace dimensional features with
    dimensionless quantities usefull in the determination of
    the flow regime. To obtain additional insights about the
    choosen dimensionless features, read the paper.

    Input:

        pd.DataFrame df : dataframe with dimensional features

    Output:

        pf.DataFrame df : dataframe with dimensionless features

    '''
    #Angles related transofrmation
    df['SinAng'] = np.sin(df['Ang']/360*(2*math.pi))
    df['CosAng'] = np.cos(df['Ang']/360*(2*math.pi))

    #Reynolds Numbers
    df['ReL'] = df['ID']*df['DenL']*df['Vsl']/df['VisL']
    df['ReG'] = df['ID']*df['DenG']*df['Vsg']/df['VisG']

    #Fanning
    df['FanningL'] = 16/df['ReL']
    df.loc[df['ReL']>2300, 'FanningL'] = 0.0625/((np.log((150.39/(df['ReL']**0.98865))
                                                  -(152.66/(df['ReL']))))**2)

    df['FanningG'] = 16/df['ReG']
    df.loc[df['ReG']>2300, 'FanningG'] = 0.0625/((np.log((150.39/(df['ReG']**0.98865))
                                                  -(152.66/(df['ReG']))))**2)

    #Froude Number
    df['FrL'] = ((df['DenL']/((df['DenL']-df['DenG'])*
                    gravity*df['ID']))**0.5)*df['Vsl']
    df['FrG'] = ((df['DenG']/((df['DenL']-df['DenG'])*
                    gravity*df['ID']))**0.5)*df['Vsg']

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
                     'FanningL','FanningG', 'CosAng', 'SinAng'], 
            inplace=True, errors='ignore')

    #Put the Target as the last column
    if 'Flow_label' in df.columns:
        df = df[[col for col in df.columns if col != 'Flow_label']+['Flow_label']]

    return df

def calculate_holdup(X, Y):

    """
    Simple function to calculate the liquid
    holdup in the selected pipe. This quantity
    can be used as a feature accoring to the
    Multiphase Flow Handbook.
    """

    alpha = np.zeros(len(X))
    for i, _ in enumerate(X):
        alpha[i] = optimize.brentq(
            lambda a, X=X[i], Y=Y[i]: (1+75*a)/((1-a)**2.5 *a) -X**2/a**3 -Y,
            a=0.00001, b=0.99999,
            args=(X[i], Y[i]))

    return alpha
