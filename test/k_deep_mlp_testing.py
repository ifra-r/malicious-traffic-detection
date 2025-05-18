import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_and_test.deep_mlp import MLPModel,evaluate
import pickle
from utils.metrics import accuracy, precision, recall

def load_data():
    data = pd.read_csv('../data/cleaned/se.csv')
    X = data.iloc[:, :-1].values.astype(np.float32)
    y = data.iloc[:, -1].values.reshape(-1, 1).astype(np.float32)
    return X, y

def load_model(path="../results/k_deep_mlp.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def main():
    X, y = load_data()
    model = load_model()
    print("Evaluating loaded model on full dataset:")
    evaluate(model, X, y)

if __name__ == "__main__":
    main()
