#From the same folder
from script import utils
from script import model_optuna as mo

#Basic python stuff
import os, shutil, time
import numpy as np
import pandas as pd
from pickle import dump, load

import torch
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#Tf for a save check
from tensorflow.keras.models import load_model as tf_load 
import tensorflow as tf

#Optimization
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE

#Database Lib
import urllib
import pyodbc

class StackedClassifier():
    '''
    Stacked Classifier algorithms based on
    Optuna library.
    
    The main objective of this class is to
    optimize a series of sklearn-like models
    by tuning their hyperparameters using bayesian
    inference. The optimized estimators are also
    stacked in order to generate a super-learner
    if possible.    
    '''
    def __init__(self, base_algos=[], 
                 cv=5, scaler=None, 
                 balance_method=SMOTE(),
                 database_loc='local'):    
        
        '''
        init function of the stacked classifier:
        
        Input:
            base_algos : list of sklearn-like estimators to
                         optimize,
            
            cv : number of cross validation folds 
                 for the optimization,
            
            balance_method : oversampling method from 
                             imblearn library
            
            database_loc : location of the database 
                           (local or Azure)
        '''
        #Dictionary that contains usefull info of the SC:
        self.sc_info = {
            'base_models'    : {x:[] for x in base_algos},
            'meta_model'     : [],
            'scaler'         : scaler,
            'balance_method' : balance_method,
            'cv'             : cv,
            'db_type'        : database_loc
        }     
        
        #Dataframe containing info about the trials
        self.logs = pd.DataFrame() 

    def optimize_base(self, df, n_trials, kept=None):

        '''
        This function is the core of the stacking classifier.
        Given a number of trials and the dataframe, it will search
        the hyperparameter space of the estimators using Optuna.
        '''
        self._check_algos()
        
        self.logs_info = [] #Info about the trial
        self.meta_info = {} #Meta info for metalearner
        
        #Generate the splits for training
        train, valid = self._get_folds(df, kept)
        
        #Define Optuna Sampler and Pruner
        sampler = optuna.samplers.TPESampler()
        pruner  = optuna.pruners.MedianPruner(n_startup_trials=1, 
                                              n_warmup_steps=0, 
                                              interval_steps=1)
        #Optimize the algos:
        for algo in self.sc_info['base_models'].keys():

            study = optuna.create_study(sampler=sampler, pruner=pruner,
                    direction="maximize", study_name=f'{algo} optimization',
                    storage=self.database_location(algo), load_if_exists=True)

            study.optimize(lambda trial: self._optimize_single(trial, train, valid,
                           algo), n_trials=n_trials, n_jobs=1)

        #Apply PostProcessing
        self._postprocessing()

    def _optimize_single(self, trial, train, valid, algo, _base=True):

        '''      
        Function to optimize a single estimator. It will
        perform any of the "in-fold" preprocessing steps
        that are required for this specific problem.
        '''
        trial_id = f"{algo}_{trial.number}"

        predictions = []
        cv_metrics = np.zeros((self.sc_info['cv'], 4))
        for i in range(self.sc_info['cv']):

            #Get the data for each fold
            X_train, y_train = train[i][0], train[i][1]
            X_valid, y_valid = valid[i][0], valid[i][1]

            #Retrive the model with the trial hyperparameters
            model, train_params = mo.optuna_wrapper(algo, trial, X_valid, y_valid)            
            model.fit(X_train, y_train, **train_params)

            #Cast the prediction and compare the results
            y_pred = model.predict(X_valid)
            y_pred = y_pred if y_pred.ndim==1 else y_pred.argmax(1)
            
            cv_metrics[i, 0]  = accuracy_score(y_valid, y_pred)
            cv_metrics[i, 1:] = precision_recall_fscore_support(y_valid,  y_pred,
                                            average='macro', zero_division=0)[:3]
            
            #Check if the trial should be pruned at each CV
            if _base:
                trial.report(np.mean(cv_metrics[:i+1, 0]), i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                #Extend meta predictions
                predictions.extend(y_pred)


        #Insert info about the estimator
        if _base:
            #Info dict about training
            self.logs_info.append({
                'ID'           : trial_id,
                'Algo'         : algo,
                'Accuracy'    : np.mean(cv_metrics[:,0]),
                'Accuracy_std': np.std(cv_metrics[:,0]),
                'F1_score'    : np.mean(cv_metrics[:,-1]),
                'F1_score_std': np.std(cv_metrics[:,-1])
            })

            #Info dict about meta-predictions
            self.meta_info[trial_id] = predictions

        #Return the mean accuracy
        return np.mean(cv_metrics[:,0])

    def _postprocessing(self):
        
        '''
        Postprocessing function to save relevant infomations
        about the current state of the best models. It will
        generate two parquet file contianing the logs (accuracy,
        f1 score, train params) of each trial and the meta info
        (the predicted out of folds classes)
        '''
        logs = pd.DataFrame(self.logs_info)
        meta = pd.DataFrame(self.meta_info)
        
        #Check if log file is already present:
        if 'logs.parquet' in os.listdir('results/info/'):
            logs = pd.read_parquet('results/info/logs.parquet').append(logs, ignore_index=True)
            meta = pd.read_parquet('results/info/meta.parquet').join(meta)
        
        logs.to_parquet('results/info/logs.parquet')
        meta.to_parquet('results/info/meta.parquet')
        
        self.select_best(logs, meta)
        
    def select_best(self, logs, meta):
        
        '''
        Function to select the best models in the stacked classifier
        It will select the meta learner columns to consider.
        '''
        
        #Select only relevant Algos for the classifier
        algos = self.sc_info['base_models'].keys()
        logs = logs[logs["Algo"].isin(algos)].reset_index(drop=True)
        
        bof_idx = logs.groupby(['Algo'])['Accuracy'].idxmax()
        bof_col = logs.iloc[bof_idx]['ID']

        self.X_meta = meta[bof_col].values
        self.logs = logs.iloc[bof_idx].sort_values(['Algo'], ascending=False)

    def _get_folds(self, df, kept):
        
        '''
        Standardize the folds for each estimators. This allow
        in fold special operations, as well as the consistency
        required by a stacking classifier.
        '''
        X, y = df.iloc[:,:-1].values, df.iloc[:,[-1]].values.ravel()
        train_dict, valid_dict = {}, {}

        skf = StratifiedKFold(n_splits=self.sc_info['cv'], shuffle=False)
        for i, [train_index, valid_index] in enumerate(skf.split(X, y)):

            df_train = df.iloc[train_index].copy()
            df_valid = df.iloc[valid_index].copy()    

            df_train = utils.silver_to_gold(df_train, self.sc_info['balance_method'], kept)
            df_valid = utils.silver_to_gold(df_valid, kept_columns=kept, gaussian_noise=False)

            X_train, y_train = df_train.iloc[:,:-1].values, df_train.iloc[:,[-1]].values.ravel()
            X_valid, y_valid = df_valid.iloc[:,:-1].values, df_valid.iloc[:,[-1]].values.ravel()

            if self.sc_info['scaler']:
                X_train = self.sc_info['scaler'].fit_transform(X_train)
                X_valid = self.sc_info['scaler'].transform(X_valid)

            train_dict[i] = (X_train, y_train)
            valid_dict[i] = (X_valid, y_valid)

        y_gt = []
        x_gt = []
        for values in valid_dict.values():
            x_gt.extend(values[0])
            y_gt.extend(values[1])
            
        self.y_meta = np.array(y_gt)
        self.x_grou = np.array(x_gt)
        
        return train_dict, valid_dict

    def _split_meta(self, X, y):
        
        '''
        Same function of _get_folds, without the in
        fold operations required by this god forsaken
        problem.
        '''
        train_dict, valid_dict = {}, {}
        skf = StratifiedKFold(n_splits=self.sc_info['cv'], shuffle=True, random_state=42)
        for i, [train_index, valid_index] in enumerate(skf.split(X, y)):
            
            train_dict[i] = X[train_index], y[train_index]
            valid_dict[i] = X[valid_index], y[valid_index]

        return train_dict, valid_dict

    def train_meta(self, meta_algo, n_trials=3):
        
        '''
        Train the meta learner using the out-of-fold
        predictions of base estimators.
        '''
        
        assert self.X_meta.shape, 'Please train base estimators before proceding'
        
        train, valid = self._split_meta(self.X_meta, self.y_meta[:len(self.X_meta)])

        study = optuna.create_study(direction="maximize", 
                                    study_name=f'{meta_algo} meta optimization')

        study.optimize(lambda trial: self._optimize_single(trial, train, valid,
                        meta_algo, _base=False), n_trials=n_trials)

        model = mo.get_model(model_type=meta_algo, params=study.best_trial.params)
        model.fit(self.X_meta, self.y_meta[:len(self.X_meta)])

        #Save the meta model
        self.sc_info['meta_model'] = model
        
    def load_base(self, folder='./results/Models'):
        '''
        Simple function to load the various pre-trained models
        '''
        if self.sc_info['scaler']:
            self.sc_info['scaler'] = utils.load_model(folder, "StandardScaler")
         
        models = self.sc_info['base_models']
        for algo in models.keys():                
            models[algo] = utils.load_model(folder, algo)

    def train_base(self, X_train, y_train, remove_old=True):
        
        '''
        Train the best estimators found in the 
        optimization step usign the whole dataset
        '''
        #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        if remove_old:
            self.make_clean()
        
        scaler=self.sc_info['scaler']
        if scaler:
            X_train = scaler.fit_transform(X_train)
            dump(scaler, open(f'./results/Models/Scaler.pkl', 'wb'))
            
        for algo, study in self.get_studies().items():
            
            model = mo.get_model(algo, study.best_params)
            model.fit(X_train, y_train, **mo.t_params(algo, X_train, y_train))
            
            if algo=="MLP":
                model.save('./results/Models/TF_models/')
            elif algo=="TabNet":
                torch.save(model, 'results/Models/TabNet.pth')
            else:
                dump(model, open(f'results/Models/{algo}.pkl', 'wb'))
                
            self.sc_info['base_models'][algo] = model


    def make_clean(self, folder='./results/Models'):
        
        '''
        Function to remove all the file in a specific
        folder. Usefull to remove the previous models,
        as well as old database during the re-iteration 
        of the pipeline. Remember, this function kinda
        has no mercy, don't feed it stupid path or it will
        remove importat files
        '''
        
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))       
    
    def _base_pred(self, X):
        
        '''
        Cast the predictions of the base estimators
        for a feature array X
        '''
        if self.sc_info['scaler']:
            X = self.sc_info['scaler'].transform(X)

        y_meta = np.zeros((X.shape[0], len(self.sc_info['base_models'])))
        for i, model in enumerate(self.sc_info['base_models'].values()):
            y_pred = model.predict(X)
            y_meta[:,i] = y_pred if y_pred.ndim==1 else y_pred.argmax(1)
        
        return y_meta
    
    def predict(self, X):
        
        '''
        Returns the predictions for an array X
        '''
        y_meta = self._base_pred(X)
        return (self.sc_info['meta_model'].predict(y_meta))
        
    def predict_proba(self, X):     
        '''
        Returns the predicted probabilities
        for an array X
        '''
        y_meta = self._base_pred(X)
        return (self.sc_info['meta_model'].predict_proba(y_meta))
    
    def _check_algos(self):
        
        '''
        Make sure that the models are contained in the
        optuna wrapper method
        '''
        accepted_models = ['XGBoost', 'LightGBM', 
                           'CatBoost', 'TabNet', 
                           'RandomForest', 'ExtraTrees', 
                           'SVC', 'MLP']    
        
        algos = self.sc_info['base_models'].keys()
        r = [x not in accepted_models for x in algos]
        
        if any(r):
            raise NameError("The following algorithms are "
                            f"not accepted: {np.array(list(algos))[r]} \n"
                            f"accepted algos: {accepted_models}")
            
    def database_location(self, algo):
        '''
        Returns the location of the database chosen
        for the optuna optimization. Currently the
        options are for a local sqlite database and
        a cloud based microsoft sql server.
        '''
        
        if self.sc_info['db_type']=='local':
            #Connect to local sqlite
            connect_str = f'sqlite:///results/db/{algo}.db'
            return connect_str
        
        else:
            #Connect to Azure Server:
            server = "optuna.database.windows.net"
            username = "optuna-admin"
            password = "SuperMalus96"
            driver = '{ODBC Driver 17 for SQL Server}'
            
            database = algo+'-db'
            odbc_str = ('DRIVER='+driver+';SERVER='+server+';PORT=1433;UID='
                        +username+';DATABASE='+ database + ';PWD='+ password)
            connect_str = ('mssql+pyodbc:///?odbc_connect=' 
                           + urllib.parse.quote_plus(odbc_str))

            return connect_str

    #Get Methods    
    def get_base(self):
        '''
        Returns the list of the base estiamtors
        '''
        return (self.sc_info['base_models'])
    
    def get_scaler(self):
        '''
        Returns the scaler algorithm
        '''
        return (self.sc_info['scaler'])

    def get_studies(self):
        '''
        Return the studies dictionary
        '''        
        study_dict = {}
        for algo in self.sc_info['base_models'].keys():
            study_dict[f'{algo}'] = optuna.load_study(
                study_name=f'{algo} optimization',
                storage=self.database_location(algo))
            
        return study_dict            
        
    def plot_history(self, study_name):
        '''
        Simple wrapper for optuna plot history of a single
        estimator
        '''
        
        self._check_algos()
        study = optuna.load_study(study_name=f'{study_name} optimization',
                                  storage=self.database_location(study_name))

        return optuna.visualization.plot_optimization_history(study)