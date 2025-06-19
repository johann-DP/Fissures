def rfe_selection(X, y, n_features_to_select=20):
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import RFE
    estimator = LinearRegression()
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=0.1)
    selector.fit(X, y)
    return X.columns[selector.support_].tolist()


def stability_selection(X, y, n_iter=50, sample_fraction=0.75, threshold=0.5):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LassoCV
    n_samples = X.shape[0]
    counts = pd.Series(0, index=X.columns)
    for i in range(n_iter):
        sample_idx = np.random.choice(n_samples, int(n_samples * sample_fraction), replace=False)
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
        lasso = LassoCV(cv=5, random_state=42, max_iter=200000)
        lasso.fit(X_sample, y_sample)
        selected = X.columns[lasso.coef_ != 0]
        counts[selected] += 1
    freq = counts / n_iter
    return freq[freq >= threshold].index.tolist()


def select_features_combined(X, y, n_features_to_select=20, n_iter=50, sample_fraction=0.75, stability_threshold=0.5):
    features_rfe = set(rfe_selection(X, y, n_features_to_select=n_features_to_select))
    features_stab = set(stability_selection(X, y, n_iter=n_iter, sample_fraction=sample_fraction, threshold=stability_threshold))
    if len(features_rfe) >= len(features_stab):
        return list(features_rfe)
    else:
        return list(features_stab)


def random_search_selection(X, y, n_search=10):
    import numpy as np
    best_params = None
    best_score = -np.inf
    param_candidates = []
    for i in range(n_search):
        n_feat = np.random.choice([20, 30, 40, 50])
        sample_fraction = np.random.uniform(0.75, 0.95)
        stability_threshold = np.random.uniform(0.4, 0.7)
        param_candidates.append((n_feat, sample_fraction, stability_threshold))
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for params in param_candidates:
        n_feat, sf, thresh = params
        sel = select_features_combined(X, y, n_features_to_select=n_feat, n_iter=30, sample_fraction=sf, stability_threshold=thresh)
        if len(sel) == 0:
            score = -np.inf
        else:
            X_sel = X[sel]
            model = LinearRegression()
            y_pred = cross_val_predict(model, X_sel, y, cv=cv, n_jobs=-1)
            score = r2_score(y, y_pred)
        if score > best_score:
            best_score = score
            best_params = params
    from rich.console import Console
    Console().print(f"[bold green]Recherche aléatoire sélectionnée:[/bold green] n_features={best_params[0]}, sample_fraction={best_params[1]:.2f}, threshold={best_params[2]:.2f} (R²={best_score:.3f})")
    return best_params
