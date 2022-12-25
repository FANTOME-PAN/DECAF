from typing import Any, Tuple
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import numpy as np
import pandas as pd


# It will apply a perturbation at each node provided in perturb.
def gen_data_nonlinear(
        G: Any,
        base_mean: float = 0,
        base_var: float = 0.3,
        mean: float = 0,
        var: float = 1,
        SIZE: int = 10000,
        err_type: str = "normal",
        perturb: list = [],
        sigmoid: bool = True,
        expon: float = 1.1,
) -> pd.DataFrame:
    list_edges = G.edges()
    list_vertex = G.nodes()

    order = []
    for ts in nx.algorithms.dag.topological_sort(G):
        order.append(ts)

    g = []
    for v in list_vertex:
        if v in perturb:
            g.append(np.random.normal(mean, var, SIZE))
            print("perturbing ", v, "with mean var = ", mean, var)
        else:
            if err_type == "gumbel":
                g.append(np.random.gumbel(base_mean, base_var, SIZE))
            else:
                g.append(np.random.normal(base_mean, base_var, SIZE))

    for o in order:
        for edge in list_edges:
            if o == edge[1]:  # if there is an edge into this node
                if sigmoid:
                    g[edge[1]] += 1 / 1 + np.exp(-g[edge[0]])
                else:
                    g[edge[1]] += g[edge[0]] ** 2
    g = np.swapaxes(g, 0, 1)

    return pd.DataFrame(g, columns=list(map(str, list_vertex)))


def load_adult_ex() -> pd.DataFrame:
    """Load the Adult dataset in a pandas dataframe"""

    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    train_df = pd.read_csv(path, names=names, index_col=False)
    test_df = pd.read_csv(test_path, names=names, index_col=False)[1:]
    df = pd.concat([train_df, test_df])
    df = df.applymap(lambda x: x.strip() if type(x) is str else x)

    for col in df:
        if df[col].dtype == "object":
            df = df[df[col] != "?"]

    df["income"].replace({'<=50K.': '<=50K', '>50K.': '>50K'}, inplace=True)

    """Preprocess adult data set."""

    replace = [
        [
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked",
        ],
        [
            "Bachelors",
            "Some-college",
            "11th",
            "HS-grad",
            "Prof-school",
            "Assoc-acdm",
            "Assoc-voc",
            "9th",
            "7th-8th",
            "12th",
            "Masters",
            "1st-4th",
            "10th",
            "Doctorate",
            "5th-6th",
            "Preschool",
        ],
        [
            "Married-civ-spouse",
            "Divorced",
            "Never-married",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
            "Married-AF-spouse",
        ],
        [
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving",
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces",
        ],
        [
            "Wife",
            "Own-child",
            "Husband",
            "Not-in-family",
            "Other-relative",
            "Unmarried",
        ],
        ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
        ["Female", "Male"],
        [
            "United-States",
            "Cambodia",
            "England",
            "Puerto-Rico",
            "Canada",
            "Germany",
            "Outlying-US(Guam-USVI-etc)",
            "India",
            "Japan",
            "Greece",
            "South",
            "China",
            "Cuba",
            "Iran",
            "Honduras",
            "Philippines",
            "Italy",
            "Poland",
            "Jamaica",
            "Vietnam",
            "Mexico",
            "Portugal",
            "Ireland",
            "France",
            "Dominican-Republic",
            "Laos",
            "Ecuador",
            "Taiwan",
            "Haiti",
            "Columbia",
            "Hungary",
            "Guatemala",
            "Nicaragua",
            "Scotland",
            "Thailand",
            "Yugoslavia",
            "El-Salvador",
            "Trinadad&Tobago",
            "Peru",
            "Hong",
            "Holand-Netherlands",
        ],
        [">50K", "<=50K"],
    ]

    for row in replace:
        df = df.replace(row, range(len(row)))

    df = pd.DataFrame(MinMaxScaler().fit_transform(df),
                      index=df.index, columns=df.columns)

    msk = np.array([0, 1, 3, 5, 6, 7, 8, 9, 12, 13, 14])
    data, index, cols = df.values, df.index, df.columns
    data = data[:, msk]
    cols = cols[msk].to_list()
    cols[-1] = 'label'
    # X = df[:, :14].astype(np.uint32)
    # y = df[:, 14].astype(np.uint8)
    #
    # return X, y
    return pd.DataFrame(data, index=index, columns=cols)


def load_adult() -> pd.DataFrame:
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    names = [
        "age",  # 0 0
        "workclass",  # 1 1
        "fnlwgt",
        "education",  # 3 2
        "education-num",
        "marital-status",  # 5 3
        "occupation",  # 6 4
        "relationship",  # 7 5
        "race",  # 8 6
        "sex",  # 9 7
        "capital-gain",
        "capital-loss",
        "hours-per-week",  # 12 8
        "native-country",  # 13 9
        "label",  # 14 10
    ]
    df = pd.read_csv(path, names=names, index_col=False)
    df = df.applymap(lambda x: x.strip() if type(x) is str else x)

    for col in df:
        if df[col].dtype == "object":
            df = df[df[col] != "?"]

    replace = [
        [
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked",
        ],
        [
            "Bachelors",
            "Some-college",
            "11th",
            "HS-grad",
            "Prof-school",
            "Assoc-acdm",
            "Assoc-voc",
            "9th",
            "7th-8th",
            "12th",
            "Masters",
            "1st-4th",
            "10th",
            "Doctorate",
            "5th-6th",
            "Preschool",
        ],
        [
            "Married-civ-spouse",
            "Divorced",
            "Never-married",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
            "Married-AF-spouse",
        ],
        [
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving",
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces",
        ],
        [
            "Wife",
            "Own-child",
            "Husband",
            "Not-in-family",
            "Other-relative",
            "Unmarried",
        ],
        ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
        ["Female", "Male"],
        [
            "United-States",
            "Cambodia",
            "England",
            "Puerto-Rico",
            "Canada",
            "Germany",
            "Outlying-US(Guam-USVI-etc)",
            "India",
            "Japan",
            "Greece",
            "South",
            "China",
            "Cuba",
            "Iran",
            "Honduras",
            "Philippines",
            "Italy",
            "Poland",
            "Jamaica",
            "Vietnam",
            "Mexico",
            "Portugal",
            "Ireland",
            "France",
            "Dominican-Republic",
            "Laos",
            "Ecuador",
            "Taiwan",
            "Haiti",
            "Columbia",
            "Hungary",
            "Guatemala",
            "Nicaragua",
            "Scotland",
            "Thailand",
            "Yugoslavia",
            "El-Salvador",
            "Trinadad&Tobago",
            "Peru",
            "Hong",
            "Holand-Netherlands",
        ],
        [">50K", "<=50K"],
    ]

    for row in replace:
        df = df.replace(row, range(len(row)))

    msk = np.array([0, 1, 3, 5, 6, 7, 8, 9, 12, 13, 14])
    data, index, cols = df.values, df.index, df.columns
    data = data[:, msk]
    cols = cols[msk].to_list()
    # X = df[:, :14].astype(np.uint32)
    # y = df[:, 14].astype(np.uint8)
    #
    # return X, y
    return pd.DataFrame(MinMaxScaler().fit_transform(data), index=index, columns=cols)


'''
dag and dag_seed of adult data set
tk2id = dict(zip(['age', 'wc', 'edu', 'ms', 'occ', 'rs', 'race', 'sex', 'hpw', 'nc', 'label'], range(11)))

col_name    num_cls # -1 means continuous, 1 means binary
age         -1
wc          7
edu         16
ms          7
occ         14
rs          6
race        5
sex         1
hpw         -1
nc          41
label       1
feature_types = [-1, 7, 16, 7, 14, 6, 5, 1, -1, 41, 1]

dag_seed = [[0, 4], [0, 8], [0, 10], [0, 3], [0, 1], [0, 2], [0, 5], 
            [1, 10], 
            [2, 4], [2, 8], [2, 10], [2, 1], [2, 5],
            [3, 4], [3, 8], [3, 10], [3, 1], [3, 5], [3, 2], 
            [4, 10], 
            [5, 10], 
            [6, 4], [6, 10], [6, 8], [6, 2], [6, 3], 
            [7, 4], [7, 3], [7, 10], [7, 8], [7, 1], [7, 2], [7, 5], 
            [8, 10], 
            [9, 3], [9, 8], [9, 2], [9, 1], [9, 10], [9, 5]]

dag = [
    ['age', 'occ'],
    ['age', 'hpw'],
    ['age', 'label'],
    ['age', 'ms'],
    ['age', 'wc'],
    ['age', 'edu'],
    ['age', 'rs'],

    ['wc', 'label'],

    ['edu', 'occ'],
    ['edu', 'hpw'],
    ['edu', 'label'],
    ['edu', 'wc'],
    ['edu', 'rs'],

    ['ms', 'occ'],
    ['ms', 'hpw'],
    ['ms', 'label'],
    ['ms', 'wc'],
    ['ms', 'rs'],
    ['ms', 'edu'],

    ['occ', 'label'],

    ['rs', 'label'],

    ['race', 'occ'],
    ['race', 'label'],
    ['race', 'hpw'],
    ['race', 'edu'],
    ['race', 'ms'],

    ['sex', 'occ'],
    ['sex', 'ms'],
    ['sex', 'label'],
    ['sex', 'hpw'],
    ['sex', 'wc'],
    ['sex', 'edu'],
    ['sex', 'rs'],

    ['hpw', 'label'],

    ['nc', 'ms'],
    ['nc', 'hpw'],
    ['nc', 'edu'],
    ['nc', 'wc'],
    ['nc', 'label'],
    ['nc', 'rs'],
]

'''
