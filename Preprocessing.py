import pandas as pd
import numpy as np
import math
import random

"""
tdl
set na to 0, i think the only columns that have na are float64s
we can train with it first to see if we need to do that
split up binnedInc
turn Geography to state
change dtype for new columns (the two binned inc ones)
"""

states = {
    "Maine": 1,
    "New Hampshire": 2,
    "Vermont": 3,
    "Massachusetts": 4,
    "Rhode Island": 5,
    "Connecticut": 6,
    "New York": 7,
    "New Jersey": 8,
    "Delaware": 9,
    "Maryland": 10,
    "District of Columbia": 11,
    "Pennsylvania": 12,
    "West Virginia": 13,
    "Virginia": 14,
    "North Carolina": 15,
    "South Carolina": 16,
    "Georgia": 17,
    "Florida": 18,
    "Alabama": 19,
    "Mississippi": 20,
    "Tennessee": 21,
    "Arkansas": 22,
    "Louisiana": 23,
    "Kentucky": 24,
    "Ohio": 25,
    "Indiana": 26,
    "Michigan": 27,
    "Illinois": 28,
    "Wisconsin": 29,
    "Minnesota": 30,
    "Iowa": 31,
    "Missouri": 32, 
    "Texas": 33,
    "Oklahoma": 34,
    "Kansas": 35,
    "Nebraska": 36,
    "South Dakota": 37,
    "North Dakota": 38,
    "Montana": 39,
    "Wyoming": 40,
    "Colorado": 41,
    "New Mexico": 42,
    "Arizona": 43,
    "Utah": 44,
    "Idaho": 45,
    "Washington": 46,
    "Oregon": 47,
    "Nevada": 48,
    "California": 49,
    "Alaska": 50,
    "Hawaii": 51,
}

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # set nan to 0
    df = df.fillna(0)

    # split binnedInc ([num, num))
    # print(df["binnedInc"])
    df[["lowerBinnedInc", "upperBinnedInc"]] = df.binnedInc.str.split(",", expand=True)
    df["lowerBinnedInc"] = df["lowerBinnedInc"].map(lambda x: x.lstrip("[("))
    df["upperBinnedInc"] = df["upperBinnedInc"].map(lambda x: x.rstrip(")]"))
    df = df.drop(columns=["binnedInc"])
    # print(df["lowerBinnedInc"])
    # print(df["upperBinnedInc"])

    # change dtype for binned inc columns``
    df["lowerBinnedInc"] = df["lowerBinnedInc"].astype(np.float64)
    df["upperBinnedInc"] = df["upperBinnedInc"].astype(np.float64)
    # print(df["lowerBinnedInc"])
    # print(df["upperBinnedInc"])

    df = df.drop(columns=["Geography"])
    
    # # turn Geography to state
    # df[["county", "state"]] = df.Geography.str.split(",", expand=True)
    # df = df.drop(columns=["county", "Geography"])
    # # print(df["state"])

    # # number states
    # df["state"] = df["state"].map(lambda x: x.strip())
    # df["state"] = df["state"].map(lambda x: states[x])
    # # print(df["state"])

    # # change dtype of states
    # df["state"] = df["state"].astype(np.int64)

    return df

def separate_labels(df: pd.DataFrame, column: str):
    labels = df[column]
    df = df.drop(columns=[column])
    y = labels.to_numpy()
    return df.to_numpy(), np.reshape(y, (y.shape[0], 1))

def train_val_split(x: np.ndarray, y: np.ndarray, pct_val: float):
    # print(f"train_val_split: x shape: {x.shape}")

    size = x.shape[0]
    count = math.ceil(size * pct_val)
    # print(f"train_val_split: count: {count}")

    val_x_dim = list(x.shape)
    val_x_dim[0] = count
    val_x = np.zeros(val_x_dim)
    # print(f"train_val_split: val: {val}")
    val_y = np.zeros((count, 1))

    for i in range(count):
        j = random.randint(0, size-1) # index to transfer
        # print(f"train_val_split: j: {j}")
        val_x[i] = x[j]
        val_y[i] = y[j]
        x = np.append(x[:j], x[j+1:], axis=0)
        y = np.append(y[:j], y[j+1:], axis=0)
        size -= 1
    return x, y, val_x, val_y

def train_test_val_split(x: np.ndarray, y: np.ndarray, pct_val: float):
    # print(f"train_val_split: x shape: {x.shape}")
    if pct_val > .5:
        print(f"train_test_val_split - pct_val cannot be over .5")
        return None

    size = x.shape[0]
    count = math.ceil(size * pct_val)
    # print(f"train_val_split: count: {count}")

    val_x_dim = list(x.shape)
    val_x_dim[0] = count
    val_x = np.zeros(val_x_dim)
    # print(f"train_val_split: val: {val}")
    val_y = np.zeros((count, 1))
    test_x_dim = list(x.shape)
    test_x_dim[0] = count
    test_x = np.zeros(test_x_dim)
    test_y = np.zeros((count, 1))

    for i in range(count):
        j = random.randint(0, size-1) # index to transfer
        # print(f"train_val_split: j: {j}")
        val_x[i] = x[j]
        val_y[i] = y[j]
        x = np.append(x[:j], x[j+1:], axis=0)
        y = np.append(y[:j], y[j+1:], axis=0)
        size -= 1
    for i in range(count):
        j = random.randint(0, size-1) # index to transfer
        # print(f"train_val_split: j: {j}")
        test_x[i] = x[j]
        test_y[i] = y[j]
        x = np.append(x[:j], x[j+1:], axis=0)
        y = np.append(y[:j], y[j+1:], axis=0)
        size -= 1
    return x, y, test_x, test_y, val_x, val_y


if __name__ == "__main__":
    train = pd.read_csv("train.csv", na_values=[0.0])
    print(train.isnull().sum())
    train = preprocess(train)
    print(train)
    print(train.dtypes)
    print(states)
    print(train.isnull().sum())
    x, y = separate_labels(train, "TARGET_deathRate")
    print(x)
    print(y)
    arr = train.to_numpy()
    print("MAIN arr: ", arr)
    x = np.array([1,2,3,4,5,6,7,8,9,10])
    y = np.array([11,12,13,14,15,16,17,18,19,20])
    x, y, val_x, val_y = train_val_split(x, y, .2)
    print(f"MAIN, x: {x}")
    print(f"MAIN, y: {y}")
    print(f"MAIN val x: {val_x}")
    print(f"MAIN, val y:{val_y}")
    x = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])
    y = np.array([[2], [4], [6], [8], [10]])
    x, y, val_x, val_y = train_val_split(x, y, .2)
    print(f"MAIN, x: {x}")
    print(f"MAIN, y: {y}")
    print(f"MAIN val x: {val_x}")
    print(f"MAIN, val y: {val_y}")
    x = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18]])
    y = np.array([[2],[4],[6],[8],[10],[12],[14],[16],[18]])
    x, y, test_x, test_y, val_x, val_y = train_test_val_split(x, y, .2)
    print(f"\n\nMAIN TRAIN TEST VAL SPLIT")
    print(f"MAIN, x: {x}")
    print(f"MAIN, y: {y}")
    print(f"MAIN test_x: {test_x}")
    print(f"MAIN test_y: {test_y}")
    print(f"MAIN val x: {val_x}")
    print(f"MAIN, val y: {val_y}")
        
