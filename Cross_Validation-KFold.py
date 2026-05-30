kfolds = 5
skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=42)

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