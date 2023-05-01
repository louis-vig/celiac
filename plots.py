from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import nhanes_data

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import shap

from train_models import plot_PRC, plot_ROC, RANDOM_SEED, test_model, get_stats

def prepare_data():
    key = [nhanes_data.TTG_KEY, nhanes_data.EMA_KEY]
    # key = [nhanes_data.CANCER]
    def modify(data):
        data[nhanes_data.EMA_KEY][(data[nhanes_data.EMA_KEY] != 1) & (data[nhanes_data.EMA_KEY] != 3)] = 0
        data[nhanes_data.TTG_KEY][(data[nhanes_data.TTG_KEY] != 1) & (data[nhanes_data.TTG_KEY] != 3) & (data[nhanes_data.TTG_KEY])] = 0

        # data[nhanes_data.CANCER][data[nhanes_data.CANCER] != 1] = 0
        data[nhanes_data.GENDER] = data[nhanes_data.GENDER] - 1
    X, y =  nhanes_data.get_xy_data(nhanes_data.BIOCHEM_SI + [nhanes_data.HEIGHT, nhanes_data.WEIGHT, nhanes_data.AGE, nhanes_data.GENDER, nhanes_data.RACE], key, modify=modify)
    # | (y[main.TTG_KEY] == 1) | ((y[main.TTG_KEY] == 3) & (y[main.EMA_KEY] == 3))
    y = np.where((y[nhanes_data.EMA_KEY] == 1), 1, 0)
    y = np.ravel(y)


    cases = np.count_nonzero(y)
    n = len(y)
    print(f"cases={cases} n={n}")
    return X, y


if __name__ == "__main__":
    X, y = prepare_data()
    cols = X.columns
    categoricals_names = [val for val in nhanes_data.RACE_CODES.values()] + [nhanes_data.GENDER]
    c_names = []
    n_names = []
    for c in cols:
        if c in categoricals_names:
            c_names.append(c)
        else:
            n_names.append(c)
    categoricals = [X.columns.get_loc(categorical) for categorical in c_names]
    print(n_names, c_names)
    scaler = ColumnTransformer([
        ('num', StandardScaler(), n_names),
        ('cat', Binarizer(), c_names)
    ])
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X)
    X.columns = n_names+c_names

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_SEED)

    cases = np.count_nonzero(ytest)
    n = len(ytest)
    print(f"test cases = {cases}, test n = {n}")

    with open("LR_model_race.mdl", mode='rb') as file:
        lr_model = pickle.load(file)
        lr_params = lr_model.best_params_
        lr_score = lr_model.best_score_
        lr_model = lr_model.best_estimator_
    with open("XGB_model_race.mdl", mode='rb') as file:
        xg_model = pickle.load(file)
        xg_params = xg_model.best_params_
        xg_score = xg_model.best_score_
        xg_model = xg_model.best_estimator_

    labels = {
        'LBXSNASI' : 'Sodium',
        'LBXSLDSI' : 'Lactate Dehydrog.',
        'LBXSKSI' : 'Potassium',
        'LBXSATSI' : 'ALT',
        'LBXSASSI' : 'AST',
        'BMXWT' : 'Weight',
        'LBDSGBSI' : 'Globulin',
        'LBXSGTSI' : 'GGT',
        'WHITE' : 'White',
        'RIDAGEYR' : 'Age',
    }

    cols = np.array(Xtest.columns)
    for i in range(len(cols)):
        if Xtest.columns[i] in labels:
            cols[i] = labels[Xtest.columns[i]]

    explain = shap.TreeExplainer(xg_model['model'])
    vals = explain.shap_values(Xtest)

    shap.summary_plot(vals, Xtest, max_display=8, feature_names=cols)


    lr_preds, lr_scores = test_model(lr_model, Xtest)
    get_stats(lr_preds, ytest)
    print(confusion_matrix(ytest, lr_preds))
    xg_preds, xg_scores = test_model(xg_model, Xtest)
    get_stats(xg_preds, ytest)
    print(confusion_matrix(ytest, xg_preds))
    

    fpr_lr, tpr_lr, _ = roc_curve(ytest, lr_scores[:,1])
    fpr_xg, tpr_xg, _ = roc_curve(ytest, xg_scores[:,1])
 

    print(f"LM ROC AUC: {auc(fpr_lr, tpr_lr)}")
    print(f"XG ROC AUC: {auc(fpr_xg, tpr_xg)}")
    plot_ROC([(fpr_lr, tpr_lr), (fpr_xg, tpr_xg)])
    plt.show()

    precision_lr, recall_lr, _ = precision_recall_curve(ytest, lr_scores[:,1])
    precision_xg, recall_xg, _ = precision_recall_curve(ytest, xg_scores[:,1])
    plot_PRC([(recall_lr, precision_lr), (recall_xg, precision_xg)], cases, n)
    plt.show()

    race = "w/ demo. info."
    no_race = "no demo. info."
    roc_data = pd.DataFrame(np.array([[race, no_race, race, no_race],
                             ['LR', 'LR', 'GBDT', 'GBDT'],
                             [0.6665343127522853, 0.606300046323849, 0.683237376745405, 0.6430547283435737]]).T)
    roc_data.columns = ['race', 'Model', 'ROC AUC']
    roc_data['ROC AUC'] = pd.to_numeric(roc_data['ROC AUC'])
    print(roc_data)
    ax = sns.barplot(data=roc_data, x='race', y='ROC AUC', hue='Model')
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    plt.ylim([0,1])
    plt.xlabel("")
    plt.title("Model Performance")
    plt.show()