import pandas as pd

"""
tdl
set na to 0, i think the only columns that have na are float64s
we can train with it first to see if we need to do that
split up binnedInc
turn Geography to state
change dtype for new columns (the two binned inc ones)
"""

if __name__ == "__main__":
    train = pd.read_csv("train.csv", na_values=[0.0])
