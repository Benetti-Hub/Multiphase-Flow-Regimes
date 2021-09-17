import xgboost as xgb
import lightgbm as lgbm
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

    
def optuna_wrapper(model_type, trial, X_train, y_train):
    
    '''
    Wrapper for optuna optimization, supported methods
    are GBMs, RandomForests, Pytorch-TabNet
    '''    
    
    di = {'XGBoost':      get_xgb_params,
          'LightGBM':     get_lgbm_params,
          'TabNet':       get_tabnet_params,
          'RandomForest': get_rf_params,
          'ExtraTrees' :  get_erf_params,
          'SVC':          get_svc_params,
          'MLP':          get_mlp_params}
    
    if model_type not in di:
        raise NameError(f"Model type not supported: {model_type}")
    
    return di[model_type](trial), t_params(model_type, X_train, y_train)  
    
    
def t_params(model_type, X_train, y_train):
       
    di = {'XGBoost':       {'verbose' : 0},
          'LightGBM':      {'verbose' : 0},
          
          'TabNet':        {'max_epochs' : 2000, 
                            'batch_size' : 16384, 
                            'virtual_batch_size': 8192,
                            'eval_set':[(X_train, y_train)],
                            'patience':300,
                            'eval_metric' : ['logloss']},
          
          'RandomForest':  {},
          'ExtraTrees':    {},
          'SVC':           {},
          
          'MLP':           {'batch_size'      : 8192,
                            'epochs'          : 4000,
                            'verbose'         : 0,
                            'callbacks'       : [EarlyStopping(monitor='loss', 
                                                               patience=400)]}
         }
    
    return di[model_type]

            
def get_model(model_type, params={}):
    
    di = {'XGBoost':       xgb.XGBClassifier,
          'LightGBM':      lgbm.LGBMClassifier,
          'TabNet':        get_tabnet,
          'RandomForest':  RandomForestClassifier,
          'ExtraTrees':    ExtraTreesClassifier,
          'SVC':           SVC,
          'MLP':           get_mlp}
        
    model = di[model_type](**params)

    return model

def get_xgb_params(trial):
    
    params = {
        "num_class": 6,
        "eval_metric": "merror",
        'n_estimators' : trial.suggest_int('n_estimators', 50, 300, log=True),
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        'seed' : 42
    }

    if params["booster"] == "gbtree" or params["booster"] == "dart":
        params["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        params["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        params["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        
    if params["booster"] == "dart":
        params["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        params["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        params["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        params["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        
    model = xgb.XGBClassifier(**params)        
    return model

def get_lgbm_params(trial):    
    
    params={'random_state' : 42,
            'num_leaves' : trial.suggest_int('num_leaves', 2, 500),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.014,0.017,0.02, 0.1]),
            'n_estimators' : trial.suggest_int('n_estimators', 50, 400, log=True),
            'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
            'subsample_freq' : trial.suggest_categorical('subsample_freq', [0, 1, 2, 10, 100]),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
    }
    
    model = lgbm.LGBMClassifier(**params)
    return model

def get_tabnet_params(trial):
    
    n_da      = trial.suggest_int("n_da", 32, 64, step=8)
    
    params={
        'n_d' : n_da,
        'n_a' : n_da,
        'n_steps' : trial.suggest_int("n_steps", 2, 5),
        'n_shared' : trial.suggest_int("n_shared", 1, 3),
        'n_independent' : trial.suggest_int("n_independent", 1, 3),
        'gamma' : trial.suggest_float("gamma", 2., 2.),
        'lambda_sparse': trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
        'optimizer_params': dict(lr=trial.suggest_float("lr", 1e-3, 1e-1, log=True))      
    }
    
    params={
        'n_d' : 48,
        'n_a' : 48,
        'n_steps' : 4,
        'n_shared' : 1,
        'n_independent' : 3,
        'gamma' : 2,
        'lambda_sparse': 0.00010145400087070716,
        'optimizer_params': dict(lr=0.001920099666669823)      
    }
    
    model = TabNetClassifier(**params)
    return model  
    
def get_tabnet(**kwargs):
    
    return TabNetClassifier(n_a=kwargs['n_da'], n_d=kwargs['n_da'],
                             n_steps=kwargs['n_steps'], 
                             n_shared=kwargs['n_shared'],
                             n_independent=kwargs['n_independent'],
                             gamma=kwargs['gamma'],
                             lambda_sparse=kwargs['lambda_sparse'],
                             optimizer_fn=torch.optim.Adam,
                             optimizer_params=dict(lr=kwargs['lr'])
                           )

def get_rf_params(trial):
    
    params = {
        'n_estimators' : trial.suggest_int('n_estimators', 50, 300, log=True),
        'criterion' : trial.suggest_categorical("criterion", ["gini", "entropy"]),
        'max_depth' : trial.suggest_int('max_depth', 1, 10),
        'n_jobs' : -1
    }
    
    model = RandomForestClassifier(**params)
    return model

def get_erf_params(trial):
    
    params = {
        'n_estimators' : trial.suggest_int('n_estimators', 50, 300, log=True),
        'criterion' : trial.suggest_categorical("criterion", ["gini", "entropy"]),
        'max_depth' : trial.suggest_int('max_depth', 1, 10),
        'n_jobs' : -1,
        'random_state' : 0
    }
    
    model = ExtraTreesClassifier(**params)
    return model
    
def get_svc_params(trial):
    
    params = {
        "kernel" : trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
        "C" : trial.suggest_float("C", 1e-10, 1e3, log=True)        
    }
        
    if params['kernel']=='poly':
        params['degree'] = trial.suggest_int('degree', 1, 2)

    model = SVC(**params)
    return model

def get_mlp_params(trial):

    params = {
        'n_layers' : trial.suggest_int("n_layers", 1, 5),
    }
    for l in range(params['n_layers']):
        params[f'n_units_l{l}'] = trial.suggest_int(f"n_units_l{l}", 64, 256, step=64)
        params[f'n_dropout_l{l}'] = trial.suggest_float(f"n_dropout_l{l}", 0., 0.4, step=0.1)
        
    params["lr"] = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    return get_mlp(**params)
    
def get_mlp(**kwargs):
    
    # We define our MLP.
    model = Sequential()
    model.add(BatchNormalization())
    for i in range(kwargs['n_layers']):
        model.add(Dense(kwargs[f"n_units_l{i}"], activation="relu"))
        model.add(Dropout(rate=kwargs[f"n_dropout_l{i}"]))
    
    model.add(Dense(6, activation="softmax"))

    # We compile our model with a sampled learning rate.
    optimizer = "Adam"
    lr = kwargs["lr"]
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=getattr(tf.keras.optimizers, optimizer)(lr=lr),
        metrics=["accuracy"],
    )
    
    return model

