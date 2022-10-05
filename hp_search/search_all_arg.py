import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_features', type=int, default=30)
parser.add_argument('--drop_years', type=str, default='none')

args = parser.parse_args()
num_features = args.num_features
drop_years = args.drop_years

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from skopt import BayesSearchCV


# In[ ]:
data = pd.read_csv('../data/train.csv')

y = data['pcr']
X = data.drop(['pcr'], axis=1)

if drop_years == 'none':
    pass
elif drop_years != 'none':
    X['Year of Diagnosis'] = X['Year of Diagnosis'].astype(int)
    X = X[X['Year of Diagnosis'] > int(drop_years)]
    y = y[X.index]
X.drop('Year of Diagnosis', axis=1, inplace=True)

fi = pd.read_csv('xgb_shap_pre.csv', index_col=0)
cols = list(fi.index[:num_features])
X = X[cols]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)


# In[ ]:
input_shape = [X_train.shape[1]]

def build_model(n_hidden=1, n_neurons=100, dropout=0.4, activation = "relu", learning_rate=3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Activation(activation))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(loss="binary_crossentropy", metrics=['AUC'], optimizer=optimizer)
    return model

keras_clf = keras.wrappers.scikit_learn.KerasClassifier(build_model)

param_distribs = {
    "n_hidden": [1, 2, 3, 4],
    "n_neurons": [25, 50, 200, 500, 1000, 1500],
    "dropout": [0.2, 0.4, 0.6, 0.8],
    "activation": ["relu", "elu"],
    "learning_rate": [3e-5, 3e-4, 3e-3, 3e-2],
}

# In[ ]:
early_stopping = keras.callbacks.EarlyStopping(
    patience=15,
    min_delta=1e-6,
    restore_best_weights=True,)

rnd_search_cv = BayesSearchCV(keras_clf, param_distribs, n_iter=50, scoring='roc_auc', cv=5, verbose=2)

rnd_search_cv.fit(X_train, y_train, epochs=50, batch_size=512,
                  validation_data=(X_valid, y_valid),
                  callbacks=[early_stopping])


keras_results = pd.DataFrame(rnd_search_cv.cv_results_)

keras_results.sort_values(by='rank_test_score').to_csv('results_hps/results_keras_{}_{}.csv'.format(num_features, drop_years))

best_keras = rnd_search_cv.best_estimator_


# Number of trees in random forest
n_estimators = [500, 750, 1000, 1250, 1500]
# Number of features to consider at every split
max_features = ['auto','sqrt']
# Maximum number of levels in tree
max_depth = [20, 40, 60, 80, 100, 120]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 3, 4, 6]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 4, 6, 8]
# Method of selecting samples for training each tree
bootstrap = [True]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
rf_random = BayesSearchCV(rf, random_grid, n_iter = 100, cv = 5, verbose=2, scoring='roc_auc', random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X, y)
rf_results = pd.DataFrame(rf_random.cv_results_)
rf_results.sort_values(by='rank_test_score').to_csv('results_hps/results_rf_{}_{}.csv'.format(num_features, drop_years))

best_rf = rf_random.best_estimator_


clf_xgb = XGBClassifier(tree_method='gpu_hist', use_label_encoder=False)

param_dist = {'n_estimators': [50, 100, 200, 400],
              'learning_rate': [0.03, 0.05, 0.075, 0.1, 0.3, 0.5],
              'subsample': [0.2, 0.4, 0.6, 1.0],
              'max_depth': [4, 6, 12, 20],
              'colsample_bytree': [0.6, 0.8, 1.0],
              'min_child_weight': [2, 4, 6]
             }

clf = BayesSearchCV(clf_xgb, 
                         param_dist,
                         cv = 5,  
                         n_iter = 50, 
                         scoring = 'roc_auc', 
                         error_score = 0, 
                         verbose = 0, 
                         n_jobs = -1)
clf.fit(X, y)
xgb_results = pd.DataFrame(clf.cv_results_)
xgb_results.sort_values(by='rank_test_score').to_csv('results_hps/results_xgb_{}_{}.csv'.format(num_features, drop_years))

best_xgb = clf.best_estimator_

keras_auroc = keras_results['mean_test_score'].max()
rf_auroc = rf_results['mean_test_score'].max()
xgb_auroc = xgb_results['mean_test_score'].max()

results_df = pd.DataFrame({'Model': ['Keras', 'Random Forest', 'XGBoost'],
                            'AUROC': [keras_auroc, rf_auroc, xgb_auroc]})
results_df.to_csv('results/results_{}_{}.csv'.format(num_features, drop_years))