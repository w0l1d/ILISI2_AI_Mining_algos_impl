import pandas as pd
import numpy as np
import numbers

from IPython.core.display_functions import display


def normalize_data(df: pd.DataFrame, test_date: pd.DataFrame = None, ignore_cols: list = []):
    ignore_cols.append('class')
    ignore_cols.append('id')
    numeric_cols = df.select_dtypes(include='number').columns
    numeric_cols = [col for col in numeric_cols if col not in ignore_cols]

    for col in numeric_cols:
        vals = df[col]
        mn = min(vals)
        mx = max(vals)
        if 0 <= mn and mx <= 1:
            continue
        denom = mx - mn

        df[col] = df.apply(lambda row: (row[col] - mn) / denom, axis=1)
        if test_date is not None:
            test_date[col] = test_date.apply(lambda row: (row[col] - mn) / denom, axis=1)

    non_numeric_cols = df.select_dtypes(include='object').columns.tolist()
    non_numeric_cols = [col for col in non_numeric_cols if col not in ignore_cols]

    for col in non_numeric_cols:
        uniq_vals = df[col]
        uniq_vals = uniq_vals.unique()
        print(f'normilize order of "{col}" :: {uniq_vals}')
        vals_map = {}
        for val in range(len(uniq_vals)):
            vals_map[uniq_vals[val]] = [0] * len(uniq_vals)
            vals_map[uniq_vals[val]][val] = 1

        df[col] = df.apply(lambda row: vals_map[row[col]], axis=1)
        if test_date is not None:
            test_date[col] = test_date.apply(lambda row: vals_map[row[col]], axis=1)


def jaccard_coefficient(l1, l2):
    print(f'jaccard_coefficient :: {l1} -- {l2}')
    # a is the number combinations of 1 and 0
    a = len([1 for i, j in zip(l1, l2) if i == 1 and j == 0])
    # b is the number combinations of 0 and 1
    b = len([1 for i, j in zip(l1, l2) if i == 0 and j == 1])
    # c is the number combinations of 1 and 1
    c = len([1 for i, j in zip(l1, l2) if i == j == 1])

    return (a + b) / (a + b + c)


def appariement_coefficient(l1, l2):
    # a is the number combinations of 1 and 0
    a = len([1 for i, j in zip(l1, l2) if i == 1 and j == 0])
    # b is the number combinations of 0 and 1
    b = len([1 for i, j in zip(l1, l2) if i == 0 and j == 1])
    # c is the number combinations of 1 and 1
    c = len([1 for i, j in zip(l1, l2) if i == j == 1])
    # d is the number combinations of 0 and 0
    d = len([1 for i, j in zip(l1, l2) if i == j == 0])

    return (a + b) / (a + b + c + d)


def distance(row_a: pd.Series, row_b: pd.Series, columns: list, options: dict = {}) -> float:
    """
        Calculates the distance between two rows
        :param row: the row to calculate the distance from
        :param test_data: the row to calculate the distance to
        :param ignore_columns: columns to ignore in the calculation
        :param options: options for the distance calculation
        :               options['class'] = the class column name
        :               options['col'] = the distance calculation method for the column
        :               options['col'] = 'jaccard' for jaccard coefficient
        :
        :                   1  0
        :               1   a  b
        :               0   c  d
        :           symitrique attribut :     distance_jaccard = (b + c) / (a + b + c)
        :       non symetrique attribut : distance_appariement = (b + c) / (a + b + c + d)  **** (default)
        :
        :return: the distance between the two rows
    """
    # calculate distance between row and test_data
    rest_dist = {}
    for col in columns:
        if col == 'class' or col == 'id':
            continue
        if type(row_a[col]) == list:
            if col in options and options[col] == 'jaccard':
                rest_dist[col] = jaccard_coefficient(row_a[col], row_b[col])
            else:
                rest_dist[col] = appariement_coefficient(row_a[col], row_b[col])
        elif isinstance(row_a[col], numbers.Number):
            rest_dist[col] = abs(row_a[col] - row_b[col])
        else:
            print(f'error :: {col} is not a number or a list : ', type(row_a[col]), row_a[col], row_b[col])

    dist = sum(rest_dist.values()) / len(rest_dist)
    print(f'distance of each column ::  {rest_dist} / {len(rest_dist)} -- result = {dist}')
    return dist


def calcul_distances(df: pd.DataFrame, labels: list):
    # calculate distance
    Z = np.zeros((len(labels), len(labels)))
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            Z[i, j] = distance(df.iloc[i], df.iloc[j], columns=df.columns)
            Z[j, i] = Z[i, j]
    return Z


def display_dist_matrix(labels: list, matrix: np.ndarray):
    # create a pandas dataframe from the numpy matrix with row and column labels
    df = pd.DataFrame(matrix, index=labels, columns=labels)

    # display the dataframe with labels for rows and columns
    styled_df = df.style.set_caption('My Matrix').set_table_styles(
        [{'selector': 'th', 'props': [('font-size', '14px')]}]).set_properties(
        **{'text-align': 'center', 'font-size': '12px'}).set_table_attributes('border="1"')

    display(styled_df)
