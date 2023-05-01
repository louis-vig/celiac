import pickle
import sklearn.linear_model as lm
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import nhanes_data
import pandas as pd
import numpy as np
import xgboost as xgb
import imblearn as imb
import seaborn as sns

RANDOM_SEED=100

def prepare_data():
    key = [nhanes_data.TTG_KEY, nhanes_data.EMA_KEY]
    key = [nhanes_data.CANCER]
    def modify(data):
        # data[nhanes_data.EMA_KEY][(data[nhanes_data.EMA_KEY] != 1) & (data[nhanes_data.EMA_KEY] != 3)] = 0
        # data[nhanes_data.TTG_KEY][(data[nhanes_data.TTG_KEY] != 1) & (data[nhanes_data.TTG_KEY] != 3) & (data[nhanes_data.TTG_KEY])] = 0

        data[nhanes_data.CANCER][data[nhanes_data.CANCER] != 1] = 0

        # data[nhanes_data.GENDER] = data[nhanes_data.GENDER] - 1
    X, y =  nhanes_data.get_xy_data(nhanes_data.BIOCHEM_SI + [nhanes_data.HEIGHT, nhanes_data.WEIGHT], key, modify=modify)
    # | (y[main.TTG_KEY] == 1) | ((y[main.TTG_KEY] == 3) & (y[main.EMA_KEY] == 3))
    # y = np.where((y[nhanes_data.EMA_KEY] == 1), 1, 0)
    y = np.ravel(y)

    print(f"cases={np.count_nonzero(y)} n={len(y)}")
    return X, y

def get_stats(preds, y, pos_class=1, neg_class=0):
    sensitivity = float(np.count_nonzero(preds[y==pos_class])/len(preds[y==pos_class]))
    specificity = 1- float(np.count_nonzero(preds[y==neg_class])/len(preds[y==neg_class]))
    print(f"sensitivity {sensitivity}")
    print(f"specificity {specificity}")
    accuracy = float(np.sum(preds==y))/y.shape[0]
    print("accuracy: %f" %(accuracy))
    return sensitivity, specificity, accuracy

def make_lr_model(y, C=100, classes=[0,1]):
    # LOGISTIC REGRESSION
    # ----------------
    # max_iter = max num of iterations to conserve
    # class_weight = dict of weights of classes in loss func
    # C = regularization - inversely proportional to regularizer strength, default = 1
    class_weights = class_weight.compute_class_weight("balanced", classes=classes, y=y)
    class_weights = {i : num for i, num in enumerate(class_weights)}
    lr_cl = lm.LogisticRegression(class_weight=class_weights, max_iter=50000, C=C)

    return lr_cl

def make_xg_model(y, n_estimators = 10, max_depth=5, reg_lambda=1000, classes = [0,1]):
    class_weights = class_weight.compute_class_weight("balanced", classes=classes, y=y)
    xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators = n_estimators, max_depth=max_depth, reg_lambda=reg_lambda, scale_pos_weight=class_weights[1])

    return xg_cl

def test_model(model, Xtest):
    return model.predict(Xtest), model.predict_proba(Xtest)

def cross_val(model, X, y, cv=5, scoring='roc_auc', sampling_strategy=0.01):
    lr_steps = []
    if sampling_strategy == 0:
        lr_steps = [('model', model)]
    else:
        lr_steps = [('over', imb.over_sampling.SMOTE(sampling_strategy=sampling_strategy)), ('model', model)]
    lr_pipe = imb.pipeline.Pipeline(steps=lr_steps)
    return cross_val_score(lr_pipe, X, y, cv=cv, scoring=scoring)

def oversample_data(Xtrain, ytrain):
    return imb.over_sampling.SMOTE(sampling_strategy=0.01).fit_resample(Xtrain, ytrain)

def optimize_hyperparams(model, X, y, params, oversample=None, categoricals=None, verbose=1, scoring='roc_auc', save=None):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_SEED)
    class_weights = class_weight.compute_class_weight("balanced", classes=[0,1], y=y)
    steps = []
    if oversample == 'SMOTE':
        steps.append(('over', imb.over_sampling.SMOTE(sampling_strategy=0.01, random_state=RANDOM_SEED)))
        # steps.append(('under', imb.under_sampling.RandomUnderSampler(sampling_strategy=1, random_state=RANDOM_SEED)))
    elif oversample == 'SMOTENC':
        steps.append(('over', imb.over_sampling.SMOTENC(categorical_features=categoricals, sampling_strategy=0.01, random_state=RANDOM_SEED)))
        # steps.append(('under', imb.under_sampling.RandomUnderSampler(sampling_strategy=1, random_state=RANDOM_SEED)))
    steps.append(('model', model))
    pipe = imb.pipeline.Pipeline(steps = steps)
    xg_gs = GridSearchCV(pipe, param_grid=params, scoring=scoring, n_jobs=-1, cv=cv, verbose=verbose)
    xg_gs.fit(X, y)
    print(pd.DataFrame(xg_gs.cv_results_))
    print(f"Best params: {xg_gs.best_params_}")
    print(f"Best score: {xg_gs.best_score_}")

    if save:
        with open(f"{save}_model.mdl", mode="wb") as flo:
            pickle.dump(xg_gs, flo)
    return xg_gs

def plot_PCA(Xpca, y, labels=[]):
    plt.scatter(Xpca[:,0][y==0], Xpca[:,1][y==0], s=1)
    plt.scatter(Xpca[:,0][y==1], Xpca[:,1][y==1], s=5)
    plt.title("Principal Component Analysis")
    plt.legend(labels)
    plt.xlabel('PC1')
    plt.ylabel('PC2')

def plot_ROC(fpr_tpr, labels = ["Logistic model", "GBDT model", "Baseline"]):
    for curve in fpr_tpr:
        plt.plot(*curve)
    plt.title("ROC Curve (no demo. info)")
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(labels)
    plt.xlim((0,1))
    plt.ylim((0,1))

def plot_PRC(recall_prec,  cases, n, labels = ["Logistic model", "GBDT model", 'No skill']):
    for curve in recall_prec:
        plt.step(*curve, where='post')
    plt.plot([0,1], [cases/n, cases/n], 'k--')
    plt.title("Precision Recall Curve (no dem. info.)")
    plt.legend(labels)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])

def plot_LDA(X, y):
    lda = LinearDiscriminantAnalysis(n_components = 1).fit(X, y)
    Xlda = np.squeeze(lda.transform(X))

    LDA = pd.DataFrame(dict(data=Xlda, Type=map(lambda i : "Celiac (scaled)" if i == 1 else "Non-Celiac", y)))

    weights = class_weight.compute_sample_weight("balanced", y)
    weights = 149*y+1
    bins = np.linspace(-4, 4, 25)
    ax = plt.subplot()
    sns.histplot(x=Xlda[y==0], bins=bins, stat="frequency", kde=True, ax = ax, label="No CD")
    lines, labels = ax.get_legend_handles_labels()
    ax2 = ax.twinx()
    sns.histplot(x=Xlda[y==1], bins=bins, stat="frequency", kde=True, ax = ax2, color=sns.color_palette()[1], label="CD")
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.set_ylabel("Frequency (No CD)")
    ax2.set_ylabel("Frequency (CD)")
    ax2.set_ylim([0,35])
    ax2.legend(lines + lines2, labels + labels2)
    plt.xlim([-5, 5])
    ax.set_xlabel("LDA Component 1")
    ax2.set_title("Linear Discriminant Analysis")

def plot_PCA(X, y):
    pca = PCA(n_components=2).fit(X,y)
    Xpca = pca.transform(X)

    data = pd.DataFrame(dict(PC1=Xpca[:,0], PC2=Xpca[:,1], Type=map(lambda i : "CD" if i == 1 else "No CD", y), Size = 5*y+1))
    sns.scatterplot(x = Xpca[:,0][y==0], y = Xpca[:,1][y==0], s=3, linewidth=0)
    sns.scatterplot(x = Xpca[:,0][y==1], y = Xpca[:,1][y==1], s=10)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.ylim([-15, 15])
    plt.xlim([-10, 30])
    plt.title("Principal Component Analysis (w/ dem. info.)")
    plt.legend(title="Type", labels=["No CD", "CD"])
    plt.show()


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
    # Xtrain, ytrain = oversample_data(Xtrain, ytrain)
    # Xtest, ytest = oversample_data(Xtest, ytest)


    lr_model = make_lr_model(y)
    class_weights = class_weight.compute_class_weight("balanced", classes=[0,1], y=y)
    param_grid = {
        "model__class_weight": [{0:1, 1:10}, {0:1, 1:100}, {0: class_weights[0], 1: class_weights[1]}, {0:1, 1:1000}],
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
    # param_grid = {
    #     "model__class_weight": [{0:1, 1:10}],
    #     "model__C": [0.1]
    #     }
    lr_cvsearch = optimize_hyperparams(lr_model, Xtrain, ytrain, oversample='', params=param_grid, categoricals=categoricals, save="LR", scoring='roc_auc')
    lr_model = lr_cvsearch.best_estimator_
    lr_model = lr_model.fit(Xtrain, ytrain)


    xg_model = make_xg_model(y)
    class_weights = class_weight.compute_class_weight("balanced", classes=[0,1], y=y)
    param_grid = {
        "model__scale_pos_weight": [class_weights[1]/class_weights[0]],
        "model__max_depth": [3, 4, 5, 6],
        "model__n_estimators": [5, 10, 25, 50],
        "model__reg_lambda":[0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
    
    # param_grid = {
    #     "model__scale_pos_weight": [class_weights[1]/class_weights[0]],
    #     "model__max_depth": [3],
    #     "model__n_estimators": [50],
    #     "model__reg_lambda":[1000]
    #     }
    xg_cvsearch = optimize_hyperparams(xg_model, Xtrain, ytrain, oversample='', params=param_grid, categoricals=categoricals, save="XGB", scoring='roc_auc')
    xg_model = xg_cvsearch.best_estimator_
    xg_model = xg_model.fit(Xtrain, ytrain)

    xgb.plot_importance(xg_model['model'], max_num_features=20,importance_type='gain',xlabel='gain')
    plt.show()


    lr_preds, lr_scores = test_model(lr_model, Xtest)
    get_stats(lr_preds, ytest)
    print(confusion_matrix(ytest, lr_preds))
    xg_preds, xg_scores = test_model(xg_model, Xtest)
    get_stats(xg_preds, ytest)
    print(confusion_matrix(ytest, xg_preds))
    

    fpr_lr, tpr_lr, _ = roc_curve(ytest, lr_scores[:,1], sample_weight=class_weight.compute_sample_weight("balanced", ytest))
    fpr_xg, tpr_xg, _ = roc_curve(ytest, xg_scores[:,1], sample_weight=class_weight.compute_sample_weight("balanced", ytest))
 

    print(f"LM ROC AUC: {auc(fpr_lr, tpr_lr)}")
    print(f"XG ROC AUC: {auc(fpr_xg, tpr_xg)}")
    plot_ROC([(fpr_lr, tpr_lr), (fpr_xg, tpr_xg)])
    plt.show()

    cases = np.count_nonzero(ytest)
    n = len(ytest)
    precision_lr, recall_lr, _ = precision_recall_curve(ytest, lr_scores[:,1])
    precision_xg, recall_xg, _ = precision_recall_curve(ytest, xg_scores[:,1])
    plot_PRC([(recall_lr, precision_lr), (recall_xg, precision_xg)], cases, n)
    plt.show()


    # plot_LDA(X[n_names], y)
    # plt.show()

    # plot_PCA(X[n_names], y)




