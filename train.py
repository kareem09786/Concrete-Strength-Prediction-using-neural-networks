"""Predict concrete compressive strength (MPa) from mix proportions and age.

Trains a small neural network (Keras) and compares it with two scikit-learn
baselines on the same held-out test set.

Usage:  python train.py
"""
import json
import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
DATA_PATH = "data/concrete_compressive_strength.csv"
RESULTS_DIR = "results"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    target = df.columns[-1]
    n_before = len(df)
    # The raw dataset contains exact duplicate rows. Dropping them before the
    # split stops identical samples landing in both train and test.
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Rows: {n_before} -> {len(df)} after dropping {n_before - len(df)} duplicates")
    return df.drop(columns=target), df[target]


def metrics(y_true, y_pred):
    return {
        "MAE_MPa": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_MPa": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_model(n_features):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X, y = load_data()

    # 70% train / 15% validation / 15% test
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED)
    print(f"Train {len(X_train)} | Validation {len(X_val)} | Test {len(X_test)}")

    # Fit the scaler on training data only, so no test information leaks in.
    scaler = StandardScaler().fit(X_train)
    Xtr, Xva, Xte = (scaler.transform(a) for a in (X_train, X_val, X_test))

    results = {}

    # Baselines
    lin = LinearRegression().fit(Xtr, y_train)
    results["Linear regression"] = metrics(y_test, lin.predict(Xte))
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED).fit(Xtr, y_train)
    results["Random forest"] = metrics(y_test, rf.predict(Xte))

    # Neural network (early stopping on the validation set)
    model = build_model(Xtr.shape[1])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True)
    history = model.fit(Xtr, y_train, validation_data=(Xva, y_val),
                        epochs=500, batch_size=32, callbacks=[stop], verbose=0)
    nn_pred = model.predict(Xte, verbose=0).ravel()
    results["Neural network (Keras)"] = metrics(y_test, nn_pred)
    results["Neural network (Keras)"]["epochs_trained"] = len(history.history["loss"])

    print("\nHeld-out test set results")
    print(pd.DataFrame(results).T.round(3).to_string())

    with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
        json.dump({"split": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
                   "seed": SEED, "results": results}, f, indent=2)

    # Plots
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE (MPa²)"); plt.yscale("log"); plt.legend()
    plt.title("Neural network training curve"); plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/training_curve.png", dpi=150); plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, nn_pred, s=14, alpha=0.7)
    lims = [min(y_test.min(), nn_pred.min()), max(y_test.max(), nn_pred.max())]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("Measured strength (MPa)"); plt.ylabel("Predicted strength (MPa)")
    plt.title("Neural network: predicted vs measured (test set)"); plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/predicted_vs_actual.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
