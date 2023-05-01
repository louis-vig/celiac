import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import json
import xgboost
import sklearn

if __name__ == "__main__":
    with open("XGB_model.mdl", mode='rb') as file:
        data = pickle.load(file)
    print(data.best_params_)
    data = pd.DataFrame(data.cv_results_)
    data = data[["mean_test_score", "params"]]
    data = pd.concat([pd.json_normalize(data["params"]), data["mean_test_score"]], axis=1)
    

    # # LR
    # data["pos_class_weight"] = data["model__class_weight.1"]/data["model__class_weight.0"]
    # data = data.drop(["model__class_weight.1", "model__class_weight.0"], axis=1)
    # data["model__C"] = np.log10(data["model__C"])
    # data.sort_values(["model__C", "pos_class_weight"], kind='stable')
    
    # print(data)
    # print(*np.meshgrid(data["pos_class_weight"].drop_duplicates(), data["model__C"].drop_duplicates()))
    # sns.heatmap(data.pivot(index="model__C", columns="pos_class_weight", values="mean_test_score"))
    # plt.show()

    # ax = plt.axes(projection="3d", )
    # ax.plot_surface(*np.meshgrid(data["pos_class_weight"].drop_duplicates(), data["model__C"].drop_duplicates()), data.pivot(index="model__C", columns="pos_class_weight", values="mean_test_score"), cmap=sns.color_palette("Blues", as_cmap=True))
    # ax.set_xlabel("Positive Class Weight")
    # ax.set_ylabel("log C (Inv. L2 Reg. Strength)")
    # ax.set_zlabel("ROC AUC")
    # ax.set_title("LR Model (no demographic info)")
    # plt.show()
    
    # XGB
    data = data.drop("model__scale_pos_weight", axis=1)
    data = data[data["model__max_depth"]==3]
    data["model__reg_lambda"] = np.log10(data["model__reg_lambda"])

    # sns.heatmap(data.pivot(index="model__max_depth", columns="model__n_estimators", values="mean_test_score"))
    # plt.show()

    print()
    ax = plt.axes(projection="3d", )
    ax.plot_surface(*np.meshgrid(data["model__n_estimators"].drop_duplicates(), data["model__reg_lambda"].drop_duplicates()), data.pivot(index="model__reg_lambda", columns="model__n_estimators", values="mean_test_score"), cmap=sns.color_palette("Blues", as_cmap=True))
    ax.set_xlabel("Number of Estimators")
    ax.set_ylabel("log λ (L2 Reg. Strength)")
    ax.set_zlabel("ROC AUC")
    ax.set_title("GDBT Model (no demographic info)")
    plt.show()