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

    # drop geo, prob don't need it
    # and the columns that have a gaps
    df = df.drop(columns=["Geography", "PctPrivateCoverageAlone", "PctEmployed16_Over", "PctSomeCol18_24"])

    # remove where MedianAge over 100
    df = df.drop(df[df["MedianAge"]>100].index)
    
    """
    df.mean() to find mean
    .max() and .mean()
    for (columnName, columnData) in stu_df.iteritems():
        print('Column Name : ', columnName)
        print('Column Contents : ', columnData.values)
    """

    
    
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

def separate_labels_ids(df: pd.DataFrame, column: str):
    labels = df[column]
    ids = df["id"]
    x = df.drop(columns=[column, "id"])
    y = labels.to_numpy()
    return ids, x.to_numpy(), np.reshape(y, (y.shape[0], 1))


def separate_labels_ids_test(df: pd.DataFrame):
    ids = df["id"]
    x = df.drop(columns=["id"])
    return ids, x.to_numpy()


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


def analyze(df: pd.DataFrame):
    print("ANALYZE")
    for col in df:
        column = df[col]
        # print(column)
        print(f"name: {col}")
        print(f"mean: {column.mean()}")
        print(f"max: {column.max()}")
        print(f"min: {column.min()}")
        print(column.quantile(q=[0.25, 0.5, 0.75], interpolation="nearest"))
        print()

def remove_outliers(df: pd.DataFrame):
    print("REMOVE OUTLIERS")
    for col in df:
        q1, q3 = df[col].quantile(q=[0.25, 0.75], interpolation="nearest").to_numpy()
        iqr = q3-q1
        upper = q3+(1.5*iqr)
        lower = q1-(1.5*iqr)
        # print(f"remove_outlier: col: {col} q1 {q1}, lower {lower}, upper {upper}, q3 {q3}")
        # print(f"remove_outliers: col {col}, bigger than q3: {df[(df[col]>(upper))]}")
        # print(f"remove_outleirs: col {col}, smaller than q1: {df[(df[col]<(lower))]}")
        df = df.drop(df[(df[col]>(upper)) | (df[col]<(lower))].index)
    return df


if __name__ == "__main__":
    train = pd.read_csv("train.csv", na_values=[0.0])
    print(train.isnull().sum())
    train = preprocess(train)
    analyze(train)

    # dict = {"A": [3,3,3,100], "B": [2,2,2,2], "C": [100,3,3,3]}
    # df = pd.DataFrame(dict)
    # print(df)
    # df = remove_outliers(df)
    # print(df)
    # print(train)
    # train = remove_outliers(train)
    # print(train)


    # print(train)
    # print(train.dtypes)
    # print(states)
    # print(train.isnull().sum())
    # x, y = separate_labels(train, "TARGET_deathRate")
    # print(x)
    # print(y)
    # arr = train.to_numpy()
    # print("MAIN arr: ", arr)
    # x = np.array([1,2,3,4,5,6,7,8,9,10])
    # y = np.array([11,12,13,14,15,16,17,18,19,20])
    # x, y, val_x, val_y = train_val_split(x, y, .2)
    # print(f"MAIN, x: {x}")
    # print(f"MAIN, y: {y}")
    # print(f"MAIN val x: {val_x}")
    # print(f"MAIN, val y:{val_y}")
    # x = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])
    # y = np.array([[2], [4], [6], [8], [10]])
    # x, y, val_x, val_y = train_val_split(x, y, .2)
    # print(f"MAIN, x: {x}")
    # print(f"MAIN, y: {y}")
    # print(f"MAIN val x: {val_x}")
    # print(f"MAIN, val y: {val_y}")
    # x = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18]])
    # y = np.array([[2],[4],[6],[8],[10],[12],[14],[16],[18]])
    # x, y, test_x, test_y, val_x, val_y = train_test_val_split(x, y, .2)
    # print(f"\n\nMAIN TRAIN TEST VAL SPLIT")
    # print(f"MAIN, x: {x}")
    # print(f"MAIN, y: {y}")
    # print(f"MAIN test_x: {test_x}")
    # print(f"MAIN test_y: {test_y}")
    # print(f"MAIN val x: {val_x}")
    # print(f"MAIN, val y: {val_y}")
        
# what is outlier


"""
https://articles.outlier.org/calculate-outlier-formula
Anything above Q3   +   1.5   x   IQR is an outlier
Anything below Q1   -   1.5   x   IQR is an outlier

https://www.statology.org/pandas-quartiles/
df['points'].quantile([0.25, 0.5, 0.75])
df.quantile(q=[0.25, 0.5, 0.75], axis=0, numeric_only=True)

col.quantile(q=[0.25, 0.5, 0.75]).to_numpy() and can index from there

df.drop(i) to drop an index
df.drop(df[df.A>3].index)
df.drop(df[(df.A>3) | (df.B>3)].index)
dict = {"A": [1,2,3,4], "B": [2,3,4,5], "C": [3,4,5,6]}
ANALYZE
name: id
mean: 1527.0004103405827
max: 3046
min: 2
0.25     763
0.50    1529
0.75    2287
Name: id, dtype: int64

name: avgAnnCount
mean: 614.9048698645876
max: 38150.0
min: 6.0
0.25     78.0
0.50    173.0
0.75    519.0
Name: avgAnnCount, dtype: float64

AVG ANN COUNT

name: avgDeathsPerYear
mean: 189.74723020106688
max: 14010
min: 3
0.25     28
0.50     63
0.75    150
Name: avgDeathsPerYear, dtype: int64

name: TARGET_deathRate
mean: 178.7689782519491
max: 362.8
min: 59.7
0.25    161.1
0.50    178.1
0.75    195.3
Name: TARGET_deathRate, dtype: float64

name: incidenceRate
mean: 448.5303465927369
max: 1206.9
min: 201.3
0.25    421.400000
0.50    453.549422
0.75    480.500000
Name: incidenceRate, dtype: float64

name: medIncome
mean: 47084.06770619614
max: 125635
min: 23047
0.25    38887
0.50    45179
0.75    52387
Name: medIncome, dtype: int64

name: popEst2015
mean: 104707.51579811244
max: 10170292
min: 827
0.25    12007
0.50    27103
0.75    70408
Name: popEst2015, dtype: int64

name: povertyPercent
mean: 16.85116947066065
max: 47.4
min: 3.2
0.25    12.2
0.50    15.9
0.75    20.3
Name: povertyPercent, dtype: float64

name: studyPerCap
mean: 160.99935856284245
max: 9762.308998
min: 0.0
0.25     0.000000
0.50     0.000000
0.75    86.582585
Name: studyPerCap, dtype: float64

name: MedianAge
mean: 44.9967583093968
max: 624.0
min: 22.3
0.25    37.7
0.50    40.9
0.75    43.9
Name: MedianAge, dtype: float64

REMOVE ALL MEDIAN AGES OVER 100 - only 23 of them

name: MedianAgeMale
mean: 39.501559294214196
max: 64.7
min: 22.4
0.25    36.3
0.50    39.5
0.75    42.4
Name: MedianAgeMale, dtype: float64

name: MedianAgeFemale
mean: 42.08305293393517
max: 65.7
min: 22.3
0.25    39.1
0.50    42.3
0.75    45.2
Name: MedianAgeFemale, dtype: float64

name: AvgHouseholdSize
mean: 2.4781821091505947
max: 3.97
min: 0.0221
0.25    2.37
0.50    2.50
0.75    2.63
Name: AvgHouseholdSize, dtype: float64

name: PercentMarried
mean: 51.78075502667214
max: 72.5
min: 23.1
0.25    47.8
0.50    52.4
0.75    56.3
Name: PercentMarried, dtype: float64

name: PctNoHS18_24
mean: 18.183299138284774
max: 62.7
min: 0.0
0.25    12.8
0.50    17.2
0.75    22.6
Name: PctNoHS18_24, dtype: float64

name: PctHS18_24
mean: 34.969675830939686
max: 72.5
min: 0.0
0.25    29.2
0.50    34.7
0.75    40.7
Name: PctHS18_24, dtype: float64

name: PctBachDeg18_24
mean: 6.175420599097251
max: 51.8
min: 0.0
0.25    3.1
0.50    5.4
0.75    8.3
Name: PctBachDeg18_24, dtype: float64

name: PctHS25_Over
mean: 34.74649158801806
max: 54.8
min: 7.5
0.25    30.2
0.50    35.1
0.75    39.7
Name: PctHS25_Over, dtype: float64

name: PctBachDeg25_Over
mean: 13.364833812064013
max: 42.2
min: 2.5
0.25     9.4
0.50    12.4
0.75    16.1
Name: PctBachDeg25_Over, dtype: float64

name: PctUnemployed16_Over
mean: 7.834509643003692
max: 29.4
min: 0.4
0.25    5.6
0.50    7.6
0.75    9.7
Name: PctUnemployed16_Over, dtype: float64

name: PctPrivateCoverage
mean: 64.35917111202298
max: 92.3
min: 22.3
0.25    57.4
0.50    65.1
0.75    72.0
Name: PctPrivateCoverage, dtype: float64

name: PctEmpPrivCoverage
mean: 41.22605662700041
max: 70.7
min: 13.5
0.25    34.5
0.50    41.2
0.75    47.7
Name: PctEmpPrivCoverage, dtype: float64

name: PctPublicCoverage
mean: 36.216003282724664
max: 65.1
min: 11.2
0.25    30.8
0.50    36.3
0.75    41.4
Name: PctPublicCoverage, dtype: float64

name: PctPublicCoverageAlone
mean: 19.23807960607304
max: 43.3
min: 2.6
0.25    14.9
0.50    18.9
0.75    23.0
Name: PctPublicCoverageAlone, dtype: float64

name: PctWhite
mean: 83.77324630393517
max: 100.0
min: 10.1991551
0.25    77.399902
0.50    90.258007
0.75    95.465300
Name: PctWhite, dtype: float64

name: PctBlack
mean: 8.979035746535494
max: 84.86602358
min: 0.0
0.25     0.605981
0.50     2.202844
0.75    10.191814
Name: PctBlack, dtype: float64

name: PctAsian
mean: 1.2600936053947476
max: 42.61942454
min: 0.0
0.25    0.257097
0.50    0.561912
0.75    1.238787
Name: PctAsian, dtype: float64

name: PctOtherRace
mean: 1.9846275649450145
max: 41.93025142
min: 0.0
0.25    0.294262
0.50    0.824313
0.75    2.131790
Name: PctOtherRace, dtype: float64

name: PctMarriedHouseholds
mean: 51.22308457943373
max: 71.70305677
min: 22.99248989
0.25    47.813926
0.50    51.659418
0.75    55.425240
Name: PctMarriedHouseholds, dtype: float64

name: BirthRate
mean: 5.643639048900288
max: 21.32616487
min: 0.0
0.25    4.524920
0.50    5.392419
0.75    6.483678
Name: BirthRate, dtype: float64

name: lowerBinnedInc
mean: 43788.95670906853
max: 61494.5
min: 22640.0
0.25    37413.8
0.50    42724.4
0.75    51046.4
Name: lowerBinnedInc, dtype: float64

name: upperBinnedInc
mean: 53994.630898645875
max: 125635.0
min: 34218.1
0.25    40362.7
0.50    45201.0
0.75    54545.6
Name: upperBinnedInc, dtype: float64
"""

# https://rstudio-pubs-static.s3.amazonaws.com/1006494_e95db6241a13484f904b4379558ba0d0.html
# what is this