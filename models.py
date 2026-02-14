

import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score
)

def Calculate_metrics(precisions, recalls, aucs, trade_counts):
    
    metricas = {
          "precision_mean": np.mean(precisions),
          "precision_std": np.std(precisions),
          "recall_mean": np.mean(recalls),
          "trades_mean": np.mean(trade_counts),
          "auc_mean": np.mean(aucs),
          "auc_std": np.std(aucs)
    }

    return metricas

def Evaluate_model(model, X_test, y_test, umbral):

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= umbral).astype(int)

    if preds.sum() > 0:  # evitar divisiones raras
        precisions = precision_score(y_test, preds)
        recalls = recall_score(y_test, preds)
    else:
        precisions = 0
        recalls = 0

    aucs = roc_auc_score(y_test, probs)
    trade_counts = preds.sum()

    return precisions, recalls, aucs, trade_counts



def Get_ML_Model(X, y, UMBRAL_LR, UMBRAL_RF):

    modelos = [LogisticRegression(
                    max_iter=1000
                ), 
               RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    random_state=42
                )]

    LR = modelos[0]
    RF = modelos[1]

    tscv = TimeSeriesSplit(n_splits=5)

    precisions_LR = []
    recalls_LR = []
    aucs_LR = []
    trade_counts_LR = []
    precisions_RF = []
    recalls_RF = []
    aucs_RF = []
    trade_counts_RF = []

    aux = 0

    for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # 1️⃣ Logistic Regression
                LR.fit(X_train, y_train)

                precisions, recalls, aucs, trade_counts = Evaluate_model(LR, X_test, y_test, UMBRAL_LR)
                
                precisions_LR.append(precisions)
                recalls_LR.append(recalls)
                aucs_LR.append(aucs)
                trade_counts_LR.append(trade_counts)

                LR_probs = LR.predict_proba(X_test)[:, 1]
                LR_mask = LR_probs >= UMBRAL_LR

                # 2️⃣ Random Forest solo sobre trades filtrados
                X_RF = X_test[LR_mask]
                y_RF = y_test[LR_mask]

                if len(X_RF) > 0:

                    RF.fit(X_train, y_train)

                    precisions, recalls, aucs, trade_counts = Evaluate_model(RF, X_RF, y_RF, UMBRAL_RF)
                    
                    precisions_RF.append(precisions)
                    recalls_RF.append(recalls)
                    aucs_RF.append(aucs)
                    trade_counts_RF.append(trade_counts)


    #calcular metricas
    metricas_LR = Calculate_metrics(precisions_LR, recalls_LR, aucs_LR, trade_counts_LR)
    metricas_RF = Calculate_metrics(precisions_RF, recalls_RF, aucs_RF, trade_counts_RF)

    return LR, RF, metricas_LR, metricas_RF

