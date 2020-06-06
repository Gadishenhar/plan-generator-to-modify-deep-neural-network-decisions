import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_labels_hist(df0, df1):

    labels = [0, 1]
    label_counts = [df0.size, df1.size]

    plt.figure()
    plt.bar(labels, label_counts)

    plt.title('Labels Distribution')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(range(len(labels)), labels)

    plt.show()

def plot_labels_log_hist(df0, df1):

    labels = [0, 1]
    label_counts = [df0.size, df1.size]

    plt.figure()
    plt.bar(labels, label_counts)
    plt.yscale('log')

    plt.title('Labels Distribution')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(range(len(labels)), labels)

    plt.show()

def plot_column_hist(df, col_num):

    df0 = df[df.iloc[0:, -1] == 0]
    df1 = df[df.iloc[0:, -1] == 1]

    labels = list(df.iloc[:, col_num].unique())
    if (len(labels) > 100):
        print('There are',len(labels), 'labels on column', col_num, '.')
        return 0
    counts0 = []
    counts1 = []
    for label in labels:
        counts0.append(sum(df0.iloc[:, col_num] == label))
        counts1.append(sum(df1.iloc[:, col_num] == label))

    plt.figure()
    plt.bar(labels, counts1)
    plt.bar(labels, counts0, bottom=counts1)
    plt.yscale('log')

    plt.title('Col 1 Distribution')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(range(len(labels)), labels)

    plt.show()


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
df0 = df[df.iloc[0:, -1] == 0]
df1 = df[df.iloc[0:, -1] == 1]

# Plot various figures based on the loaded data frames
#plot_labels_hist(df0, df1)
#plot_labels_log_hist(df0, df1)
for col_num in range(26):
    plot_column_hist(df, col_num)

