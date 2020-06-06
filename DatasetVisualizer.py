import pandas as pd
import matplotlib as plt

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
