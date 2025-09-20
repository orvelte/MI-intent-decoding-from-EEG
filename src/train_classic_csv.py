#LOSO baseline reusing the CSV loader
import argparse
import numpy as np
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from mne.decoding import CSP

from data_csv import load_subject_i

SUBJECTS = list(range(1, 10))  # 1..9

def run_loso(data_dir: str, n_csp=4, fs=250, tmin=0.5, tmax=3.5, l2_shrink="auto"):
    accs, f1s = [], []
    cms = []
    info_ref = None

    for test_i in SUBJECTS:
        X_train_list, y_train_list = [], []
        for i in SUBJECTS:
            if i == test_i:
                continue
            Xs, ys, info = load_subject_i(data_dir, i, fs=fs, tmin=tmin, tmax=tmax)
            if info_ref is None: info_ref = info
            if len(ys) == 0: 
                continue
            X_train_list.append(Xs); y_train_list.append(ys)

        X_test, y_test, _ = load_subject_i(data_dir, test_i, fs=fs, tmin=tmin, tmax=tmax)

        if not X_train_list or len(y_test) == 0:
            print(f"[S{test_i:02d}] skipped (no trials detected).")
            continue

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        clf = Pipeline([
            ("csp", CSP(n_components=n_csp, reg=None, log=True, norm_trace=False)),
            ("scaler", StandardScaler()),
            ("lda", LDA(solver="lsqr", shrinkage=l2_shrink))
        ])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred)

        accs.append(acc); f1s.append(f1m); cms.append(cm)
        print(f"[S{test_i:02d}] n_train={len(y_train):4d}  n_test={len(y_test):3d}  Acc={acc:.3f}  F1m={f1m:.3f}")

    if accs:
        print("\n=== LOSO Summary (Left vs Right) ===")
        print(f"Median Acc: {np.median(accs):.3f}  | Mean Acc: {np.mean(accs):.3f}")
        print(f"Median F1 : {np.median(f1s):.3f}  | Mean F1 : {np.mean(f1s):.3f}")
    else:
        print("No subjects produced valid trials. Check CSV schema and loader assumptions.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--n_csp", type=int, default=4)
    ap.add_argument("--fs", type=int, default=250)
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=3.5)
    ap.add_argument("--l2_shrink", type=str, default="auto")
    args = ap.parse_args()
    run_loso(**vars(args))

