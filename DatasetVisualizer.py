import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def gen_hist(df0, df1, col_name, logscale=True):
    """
    Generates a histogram for the values of a specific column of a given dataset.
    :param df0: The data frame column entries of all 0 label entries.
    :param df1: The data frame column entries of all 1 label entries.
    :param col_name: The name of the column.
    :param logscale: Whether the histogram value counts should be in logarithmic scale.
    """

    # Generate a list of all of the unique values in the column
    values = list(set(df0.unique()).union(df1.unique()))

    # For numerical and string values, we want to sort these unique values. If,
    # however, the current column represents a date, we want to sort by the flipped
    # value (sort by year first and month last).
    if 'DATE' in col_name:
        values = list(map(lambda x: '/'.join((x.split('/'))[::-1]), values))
        values.sort()
        values = list(map(lambda x: '/'.join((x.split('/'))[::-1]), values))
    else:
        values.sort()

    # If our column has a type str, we want to generate a discrete histogram (a plain
    # bar plot). We also want to generate a discrete histogram if the column type is
    # int, and we do not have too many values. In any other case, we will generate a
    # continuous histogram (with bins).
    column_is_string = (df0.dtypes == 'object')
    column_is_int = (df0.dtypes == 'int64')
    MAX_VALUES = 55
    not_too_many_values = (len(values) < MAX_VALUES)
    if column_is_string or (column_is_int and not_too_many_values):
        hist_type = 'discrete'
    else:
        hist_type = 'continuous'

    # Split our values to bins
    bins, bins_str = gen_bins(values, hist_type)

    # Count number of entries in each bin
    counts0, counts1 = count_entries_per_bin(df0, df1, bins, hist_type)

    # Generate actual bar plot
    xs = range(len(bins))
    plt.figure()
    plt.bar(xs, counts1, color='r')
    plt.bar(xs, counts0, bottom=counts1, color='g')
    if logscale:
        plt.yscale('log')
    plt.title(col_name + ' Distribution')

    # Truncate the bins_strs if they are too long and set the x ticks
    MAX_LABEL_LEN = 20
    for i, bin_str in enumerate(bins_str):
        if len(bin_str) > MAX_LABEL_LEN:
            bin_str = bin_str[:(MAX_LABEL_LEN-1)]
        bins_str[i] = bin_str

    if len(bins) >= 9:
        plt.xticks(xs, bins_str, rotation='vertical')
    else:
        plt.xticks(xs, bins_str)

    # Show the plot such that the ticks are not cut off
    plt.tight_layout()
    plt.show()


def gen_bins(values, hist_type):
    """
    Generates bins for a histogram based on its type. If the histogram is discrete,
    the bins are simply the unique values. If the histogram is continuous, the bins
    are lists of evenly sized ranges of the original values range.
    :param values: The list of unique values the data frame column entries may get.
    :param hist_type: 'discrete' for a plain bar graph, and 'continuous' for
    aggregating multiple unique values into bins.
    :return: (bins, bins_str) the list of bins, defined as above, and a clean string
    representation of them.
    """

    if hist_type == 'discrete':
        bins = values
        bins_str = list(map(str, bins))
    elif hist_type == 'continuous':
        bins_count = 10
        min_val = values[0]
        max_val = values[-1]
        step = math.ceil((max_val - min_val) / bins_count)
        bins = [[math.floor(min_val) + n * step, math.floor(min_val) + (n + 1) * step] for n in range(bins_count - 1)]
        bins_str = list(map(lambda x: (str(x[0]) + ' - ' + str(x[1])), bins))

    return (bins, bins_str)


def count_entries_per_bin(df0, df1, bins, hist_type):
    """
    Counts the number of entries in each bin, for each label class.
    :param df0: The data frame column entries of all 0 label entries.
    :param df1: The data frame column entries of all 1 label entries.
    :param bins: The list of bins to compare against. If the histogram is discrete,
    each bin is a singular value to compare against. If the histogram is continuous,
    each bin is a list of the minimal and (exclusive) maximal value the bin contains.
    :param hist_type: 'discrete' for a plain bar graph, and 'continuous' for
    aggregating multiple unique values into bins.
    :return: (counts0, counts1) the list of number of entries per bin, per label.
    """

    counts0 = []
    counts1 = []
    for bin in bins:
        if hist_type == 'discrete':
            count0 = sum(df0 == bin)
            count1 = sum(df1 == bin)
        elif hist_type == 'continuous':
            count0 = sum((df0 >= bin[0]) & (df0 < bin[1]))
            count1 = sum((df1 >= bin[0]) & (df1 < bin[1]))
        # If we append a 0 count, the stacked bar plot will not work
        counts0.append(max(1, count0))
        counts1.append(max(1, count1))

    return (counts0, counts1)


# Full list of column names in the data set
COL_NAMES = [
    'LOAN_ID',       'ORIG_CHAN', 'SELLER_NAME',    'ORIG_INT_R',   'ORIG_UPB',
    'ORIG_LOAN_T',   'ORIG_DATE', 'FST_PAY_DATE',   'ORIG_LTV',     'ORIG_CLTV',
    'NUM_BOR',       'ORIG_DTIR', 'BOR_C_SCORE',    'FST_TIME_IND', 'LOAN_PURPOSE',
    'PROP_TYPE',     'NUM_UNITS', 'OCC_TYPE',       'PROP_STATE',   'ZIP_SHORT',
    'PRIM_INS_PERC', 'PROD_TYPE', 'CO_BOR_C_SCORE', 'INSUR_TYPE',   'RELOC_IND',
    'DEFAULT'
]

# Path to the data set file
DS_PATH = 'dataset/prep_unbiased/AcqPer_2013Q4.txt'

# Load data set and split it based on the labels
df = pd.read_csv(DS_PATH, names=COL_NAMES)
df0 = df[df.iloc[0:, -1] == 0].dropna()
df1 = df[df.iloc[0:, -1] == 1].dropna()

# Plot various figures based on the loaded data frames
for col_num in range(26):
    gen_hist(df0.iloc[:, col_num], df1.iloc[:, col_num], COL_NAMES[col_num])



