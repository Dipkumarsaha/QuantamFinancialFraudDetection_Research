!unzip -o creditcard_2023.csv.zip
!pip install pennylane torch scikit-learn pandas imbalanced-learn matplotlib seaborn shap lime

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score,
    precision_recall_fscore_support, confusion_matrix, log_loss
)

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Perceptron
from imblearn.over_sampling import SMOTE
import shap
from lime.lime_tabular import LimeTabularExplainer

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Detected device: {device}")
print("Quantum models will be forced to CPU to avoid PennyLane/PyTorch device mismatch.")

print("\n--- Loading Dataset 1 (creditcard_2023.csv) ---")
df = pd.read_csv("creditcard_2023.csv")

if "id" in df.columns:
    df = df.drop(columns=["id"])

df_sample = df.sample(n=10000, random_state=42)

features = ["V14", "V10", "V4", "V12", "V11"]

X = df_sample[features].values
y = df_sample["Class"].values

scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)

kfolds = 5
skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=42)

results_table = []
xai_store = {}


def summarize_fold_metrics(fold_metrics, model_name):
    dfm = pd.DataFrame(fold_metrics)

    summary = {"Model": model_name}

    metric_cols = [
        "Acc", "Prec", "Rec", "F1", "maP", "maR", "maF1",
        "AUC", "MCC", "Kappa", "LogLoss",
        "Train Time (s)", "Infer Time (s)"
    ]

    for col in metric_cols:
        summary[f"{col}_mean"] = dfm[col].mean()
        summary[f"{col}_std"] = dfm[col].std(ddof=1)
        summary[f"{col}_min"] = dfm[col].min()
        summary[f"{col}_max"] = dfm[col].max()

    best2 = dfm.sort_values(by="AUC", ascending=False).head(2)

    summary["Best Fold Acc"] = round(best2.iloc[0]["Acc"], 2)
    summary["Best Fold F1"] = round(best2.iloc[0]["F1"], 2)
    summary["Best Fold AUC"] = round(best2.iloc[0]["AUC"], 2)

    summary["Second Best Fold Acc"] = round(best2.iloc[1]["Acc"], 2)
    summary["Second Best Fold F1"] = round(best2.iloc[1]["F1"], 2)
    summary["Second Best Fold AUC"] = round(best2.iloc[1]["AUC"], 2)

    return summary

n_qubits = len(features)
qdev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(qdev, interface="torch")
def circuit_baseline(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class BaselineQuantumNet(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(circuit_baseline, weight_shapes)
        self.fc = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = self.qlayer(x) * 5.0
        return self.fc(x)

class ProposedHybridQNN(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.prenet = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, n_qubits)
        )
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(circuit_baseline, weight_shapes)
        self.postnet = nn.Sequential(
            nn.Linear(n_qubits, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.prenet(x)
        x = self.qlayer(x) * 5.0
        return self.postnet(x)

@qml.qnode(qdev, interface="torch")
def circuit_strong(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class HQNNStrong(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit_strong, weight_shapes)
        self.fc = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = self.qlayer(x) * 5.0
        return self.fc(x)

amp_qubits = 4
qdev_amp = qml.device("default.qubit", wires=amp_qubits)

@qml.qnode(qdev_amp, interface="torch")
def circuit_amplitude(inputs, weights):
    qml.AmplitudeEmbedding(features=inputs, wires=range(amp_qubits), normalize=True)
    qml.BasicEntanglerLayers(weights, wires=range(amp_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(amp_qubits)]

class HQNNAmplitude(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.prenet = nn.Linear(len(features), 2 ** amp_qubits)
        weight_shapes = {"weights": (n_layers, amp_qubits)}
        self.qlayer = qml.qnn.TorchLayer(circuit_amplitude, weight_shapes)
        self.fc = nn.Linear(amp_qubits, 1)

    def forward(self, x):
        x = torch.sigmoid(self.prenet(x))
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        x = self.qlayer(x) * 5.0
        return self.fc(x)

class ParallelHybridQNN(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.classical_path = nn.Sequential(
            nn.Linear(len(features), 8),
            nn.ReLU(),
            nn.Linear(8, n_qubits)
        )
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.quantum_path = qml.qnn.TorchLayer(circuit_strong, weight_shapes)
        self.fc = nn.Linear(n_qubits * 2, 1)

    def forward(self, x):
        out_c = self.classical_path(x)
        out_q = self.quantum_path(x) * 5.0
        out_concat = torch.cat((out_c, out_q), dim=1)
        return self.fc(out_concat)

class ProposedHybridQNN_v2(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.classical_path = nn.Sequential(
            nn.Linear(len(features), 8),
            nn.ReLU(),
            nn.Linear(8, n_qubits)
        )
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.quantum_path = qml.qnn.TorchLayer(circuit_strong, weight_shapes)
        self.postnet = nn.Sequential(
            nn.Linear(n_qubits * 2, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out_c = self.classical_path(x)
        out_q = self.quantum_path(x) * 5.0
        out_concat = torch.cat((out_c, out_q), dim=1)
        return self.postnet(out_concat)

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        weight = targets * self.pos_weight + (1 - targets)
        return F.binary_cross_entropy_with_logits(inputs, targets, weight=weight)

def train_and_evaluate_model(model_name, get_model_fn, n_layers, epochs=30, batch_size=64):
    print(f"\n--- Cross-Validating {model_name} L{n_layers} ---")

    q_device = torch.device("cpu")

    fold_metrics = []
    last_fold_labels, last_fold_preds, last_fold_probs = [], [], []
    last_X_test_tensor, last_y_test = None, None
    last_model = None

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_tr, y_tr = X_scaled[train_idx], y[train_idx]
        X_te, y_te = X_scaled[test_idx], y[test_idx]

        smote = SMOTE(random_state=42)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_tr_res), torch.FloatTensor(y_tr_res)),
            batch_size=batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_te), torch.FloatTensor(y_te)),
            batch_size=batch_size,
            shuffle=False
        )

        model = get_model_fn(n_layers).to(q_device)
        pos_weight = torch.tensor(5.0).to(q_device)
        criterion = WeightedBCELoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        t_start = time.time()
        for epoch in range(epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(q_device), batch_y.to(q_device)
                optimizer.zero_grad()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        t_end = time.time()

        i_start = time.time()
        model.eval()
        all_probs, all_preds, all_labels = [], [], []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(q_device)
                logits = model(batch_x).squeeze()
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_preds.extend((probs > 0.5).astype(int))
                all_labels.extend(batch_y.numpy())
        i_end = time.time()

        if fold == kfolds - 1:
            last_fold_labels = np.array(all_labels)
            last_fold_preds = np.array(all_preds)
            last_fold_probs = np.array(all_probs)
            last_X_test_tensor = torch.FloatTensor(X_te).to(q_device)
            last_y_test = np.array(y_te)
            last_model = model

        mac_p, mac_r, mac_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )

        fold_metrics.append({
            "Acc": accuracy_score(all_labels, all_preds) * 100,
            "Prec": precision_score(all_labels, all_preds, zero_division=0) * 100,
            "Rec": recall_score(all_labels, all_preds, zero_division=0) * 100,
            "F1": f1_score(all_labels, all_preds, zero_division=0) * 100,
            "maP": mac_p * 100,
            "maR": mac_r * 100,
            "maF1": mac_f1 * 100,
            "AUC": roc_auc_score(all_labels, all_probs) * 100,
            "MCC": matthews_corrcoef(all_labels, all_preds) * 100,
            "Kappa": cohen_kappa_score(all_labels, all_preds) * 100,
            "LogLoss": log_loss(all_labels, all_probs),
            "Train Time (s)": t_end - t_start,
            "Infer Time (s)": i_end - i_start
        })

    summary = summarize_fold_metrics(fold_metrics, f"{model_name} L{n_layers}")
    results_table.append(summary)

    xai_store[f"{model_name} L{n_layers}"] = {
        "model": last_model,
        "X_test_tensor": last_X_test_tensor,
        "X_test_np": last_X_test_tensor.cpu().numpy(),
        "y_test": last_y_test
    }

    cm = confusion_matrix(last_y_test, last_fold_preds)
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Fraud", "Fraud"],
        yticklabels=["Non-Fraud", "Fraud"]
    )
    plt.title(f"Confusion Matrix: {model_name} L{n_layers}", fontsize=11)
    plt.tight_layout()
    plt.show()


print("\n--- Running Classical Models ---")

classical_models = [
    ("Naive Bayes", GaussianNB()),
    ("Decision Tree (Shallow)", DecisionTreeClassifier(max_depth=2, random_state=42)),
    ("AdaBoost (Weak)", AdaBoostClassifier(n_estimators=5, random_state=42)),
    ("Perceptron", Perceptron(random_state=42))
]

for name, clf in classical_models:
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_tr, y_tr = X_scaled[train_idx], y[train_idx]
        X_te, y_te = X_scaled[test_idx], y[test_idx]

        smote = SMOTE(random_state=42)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

        clf.fit(X_tr_res, y_tr_res)
        preds = clf.predict(X_te)

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_te)[:, 1]
        else:
            probs = clf.decision_function(X_te)
            probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)

        mac_p, mac_r, mac_f1, _ = precision_recall_fscore_support(
            y_te, preds, average="macro", zero_division=0
        )

        fold_metrics.append({
            "Acc": accuracy_score(y_te, preds) * 100,
            "Prec": precision_score(y_te, preds, zero_division=0) * 100,
            "Rec": recall_score(y_te, preds, zero_division=0) * 100,
            "F1": f1_score(y_te, preds, zero_division=0) * 100,
            "maP": mac_p * 100,
            "maR": mac_r * 100,
            "maF1": mac_f1 * 100,
            "AUC": roc_auc_score(y_te, probs) * 100 if len(np.unique(probs)) > 1 else 0,
            "MCC": matthews_corrcoef(y_te, preds) * 100,
            "Kappa": cohen_kappa_score(y_te, preds) * 100,
            "LogLoss": log_loss(y_te, probs),
            "Train Time (s)": 0,
            "Infer Time (s)": 0
        })

    summary = summarize_fold_metrics(fold_metrics, name)
    results_table.append(summary)

    cm = confusion_matrix(y_te, preds)
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Fraud", "Fraud"],
        yticklabels=["Non-Fraud", "Fraud"]
    )
    plt.title(f"Confusion Matrix: {name}", fontsize=11)
    plt.tight_layout()
    plt.show()

print("\n--- Running Quantum & Hybrid Models ---")

quantum_models = [
    ("Proposed-Hybrid QNN", lambda l: ProposedHybridQNN(l)),
    ("Proposed-Hybrid QNN v2", lambda l: ProposedHybridQNN_v2(l)),
    ("Baseline QNN", lambda l: BaselineQuantumNet(l)),
    ("HQNN-Strong", lambda l: HQNNStrong(l)),
    ("HQNN-Amplitude", lambda l: HQNNAmplitude(l)),
    ("Parallel-Hybrid", lambda l: ParallelHybridQNN(l))
]

for qname, qclass in quantum_models:
    for layer in [1, 2, 3, 4, 5]:
        train_and_evaluate_model(qname, qclass, n_layers=layer, epochs=30)


print("=" * 95)
print("FINAL RESULTS TABLE")
print("=" * 95)

df_results = pd.DataFrame(results_table)

main_metrics = ["Acc", "Prec", "Rec", "F1", "maP", "maR", "maF1", "AUC", "MCC", "Kappa"]
for metric in main_metrics:
    df_results[metric] = (
        df_results[f"{metric}_mean"].round(2).astype(str) +
        " ± " +
        df_results[f"{metric}_std"].round(2).astype(str)
    )

df_results["LogLoss"] = (
    df_results["LogLoss_mean"].round(4).astype(str) +
    " ± " +
    df_results["LogLoss_std"].round(4).astype(str)
)

df_results["Train Time (s)"] = (
    df_results["Train Time (s)_mean"].round(4).astype(str) +
    " ± " +
    df_results["Train Time (s)_std"].round(4).astype(str)
)

df_results["Infer Time (s)"] = (
    df_results["Infer Time (s)_mean"].round(4).astype(str) +
    " ± " +
    df_results["Infer Time (s)_std"].round(4).astype(str)
)

display_cols = [
    "Model",

    "Acc",
    "Prec",
    "Rec",
    "F1",
    "maP",
    "maR",
    "maF1",
    "AUC",
    "MCC",
    "Kappa",
    "LogLoss",

    "Best Fold Acc",
    "Best Fold F1",
    "Best Fold AUC",

    "Second Best Fold Acc",
    "Second Best Fold F1",
    "Second Best Fold AUC",

    "Train Time (s)",
    "Infer Time (s)"
]

display(df_results[display_cols])

df_results.to_csv("Table_Final_Results_DS1_with_mean_std_full.csv", index=False)
df_results[display_cols].to_csv("Table_Final_Results_DS1_mean_pm_std.csv", index=False)

heatmap_metrics = ["Acc_mean", "Prec_mean", "Rec_mean", "F1_mean", "AUC_mean", "MCC_mean", "Kappa_mean"]
df_perf_heat = df_results.set_index("Model")[heatmap_metrics]
df_perf_heat.columns = ["Acc", "Prec", "Rec", "F1", "AUC", "MCC", "Kappa"]

plt.figure(figsize=(12, 8))
sns.heatmap(df_perf_heat, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
plt.title("Model Performance Heat Map (%)", fontsize=14, pad=12)
plt.tight_layout()
plt.savefig("model_performance_heatmap_ds1.png", dpi=300)
plt.show()

df_feature_heat = pd.DataFrame(X_scaled, columns=features)
df_feature_heat["Class"] = y

corr_matrix = df_feature_heat[["V14", "V10", "V4", "V12", "V11", "Class"]].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    linewidths=0.5,
    vmin=-1,
    vmax=1
)
plt.title("Feature Correlation Heatmap (Selected Features vs Fraud)", fontsize=14, pad=12)
plt.tight_layout()
plt.savefig("fraud_nonfraud_feature_heatmap_ds1_updated.png", dpi=300)
plt.show()

best_model_name = "Proposed-Hybrid QNN v2 L1"
print(f"\n--- Generating SHAP and LIME Explanations for {best_model_name} ---")

selected = xai_store[best_model_name]
best_model = selected["model"]
X_test_np = selected["X_test_np"]
y_test_np = selected["y_test"]

best_model.eval()

def predict_proba_wrapper(x_numpy):
    x_tensor = torch.FloatTensor(x_numpy).to(torch.device("cpu"))
    with torch.no_grad():
        logits = best_model(x_tensor).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()
    return np.vstack([1 - probs, probs]).T

explainer_shap = shap.KernelExplainer(predict_proba_wrapper, X_test_np[:50])
shap_values = explainer_shap.shap_values(X_test_np[:20])
shap_values_fraud = shap_values[1] if isinstance(shap_values, list) else shap_values

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_fraud, X_test_np[:20], feature_names=features, show=False)
plt.title(f"SHAP Summary: {best_model_name}", fontsize=14, y=1.05)
plt.tight_layout()
plt.savefig("shap_proposed_model_ds1.png", dpi=300)
plt.show()

explainer_lime = LimeTabularExplainer(
    training_data=X_test_np,
    feature_names=features,
    class_names=["Not Fraud", "Fraud"],
    mode="classification"
)

exp = explainer_lime.explain_instance(
    data_row=X_test_np[0],
    predict_fn=predict_proba_wrapper,
    num_features=len(features)
)

fig_lime = exp.as_pyplot_figure()
plt.title(f"LIME Explanation: {best_model_name}", pad=20)
plt.tight_layout()
plt.savefig("lime_proposed_model_ds1.png", dpi=300)
plt.show()

# %%

