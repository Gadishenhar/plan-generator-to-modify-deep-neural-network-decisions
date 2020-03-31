import pandas as pd


def print_useful_info(df_col, col_num):
    # TODO Figure out why set() takes forever
    print('Column #', col_num)
    print('There are', len(df_col), 'total value')
    print('There are', len(set(df_col)), 'unique values')
    print('There are', df_col.isna().sum(), 'missing values')
    print('')


# Load data set
df = pd.read_table('./tmp_dataset.txt', '|')
# TODO Add column names to data frame / dataset file

# Column 0 - LOAN IDENTIFIER
# TODO Keep this as assertion to make sure there really aren't duplicate identifiers
# print(len(df.iloc[:, 0]))
# print(len(set(df.iloc[:, 0])))
print_useful_info(df.iloc[:, 0], 0)

# Column 1 - ORIGINATION CHANNEL
print_useful_info(df.iloc[:, 1], 1)
df.iloc[:, 1] = (df.iloc[:, 1]).replace('R', 1)  # R stands for Retail
df.iloc[:, 1] = (df.iloc[:, 1]).replace('C', 2)  # C stands for Correspondent
df.iloc[:, 1] = (df.iloc[:, 1]).replace('B', 3)  # B stands for Broker

# Column 2 - SELLER NAME
# TODO Consider using deterministic numbering to the sellers (sets are not)
# TODO Consider using values indicative of prevalence of sellers
print_useful_info(df.iloc[:, 2], 2)
sellers = set(df.iloc[:, 2])
for i, seller in enumerate(sellers):
    df.iloc[:, 2] = (df.iloc[:, 2]).replace(seller, i)

# Column 3 - ORIGINAL INTEREST RATE
print_useful_info(df.iloc[:, 3], 3)
# Nothing to do

# Column 4 - ORIGINAL UPB
print_useful_info(df.iloc[:, 4], 4)
# Nothing to do

# Column 5 - ORIGINAL LOAN TERM
print_useful_info(df.iloc[:, 5], 5)
# Nothing to do

# Column 6 - ORIGINATION DATE
print_useful_info(df.iloc[:, 6], 6)
# TODO remove this column!

# Column 7 - FIRST PAYMENT DATE
print_useful_info(df.iloc[:, 7], 7)
# TODO remove this column!

# Column 8 - ORIGINAL LOAN-TO-VALUE (LTV)
print_useful_info(df.iloc[:, 8], 8)
# Nothing to do

# Column 9 - ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)
print_useful_info(df.iloc[:, 9], 9)
# Nothing to do

# Column 10 - NUMBER OF BORROWERS
col_10_mean = (df.iloc[:, 10]).mean()
df.iloc[:, 10] = (df.iloc[:, 10]).fillna(col_10_mean)
print_useful_info(df.iloc[:, 10], 10)
# Nothing to do

# Column 11 - ORIGINAL DEBT TO INCOME RATIO
col_11_mean = (df.iloc[:, 11]).mean()
df.iloc[:, 11] = (df.iloc[:, 11]).fillna(col_11_mean)
print_useful_info(df.iloc[:, 11], 11)
# Nothing to do

# Column 12 - BORROWER CREDIT SCORE AT ORIGINATION
col_12_mean = (df.iloc[:, 12]).mean()
df.iloc[:, 12] = (df.iloc[:, 12]).fillna(col_12_mean)
print_useful_info(df.iloc[:, 12], 12)
# Nothing to do

# Column 13 - FIRST TIME HOME BUYER INDICATOR
print_useful_info(df.iloc[:, 13], 13)
df.iloc[:, 13] = (df.iloc[:, 13]).replace('N', 0)
df.iloc[:, 13] = (df.iloc[:, 13]).replace('Y', 1)
df.iloc[:, 13] = (df.iloc[:, 13]).replace('U', 2)

# Column 14 - LOAN PURPOSE
print_useful_info(df.iloc[:, 14], 14)
df.iloc[:, 14] = (df.iloc[:, 14]).replace('P', 0)
df.iloc[:, 14] = (df.iloc[:, 14]).replace('C', 1)
df.iloc[:, 14] = (df.iloc[:, 14]).replace('R', 2)
df.iloc[:, 14] = (df.iloc[:, 14]).replace('U', 3)

# Column 15 - PROPERTY TYPE
print_useful_info(df.iloc[:, 15], 15)
df.iloc[:, 15] = (df.iloc[:, 15]).replace('SF', 0)
df.iloc[:, 15] = (df.iloc[:, 15]).replace('CO', 1)
df.iloc[:, 15] = (df.iloc[:, 15]).replace('CP', 2)
df.iloc[:, 15] = (df.iloc[:, 15]).replace('MH', 3)
df.iloc[:, 15] = (df.iloc[:, 15]).replace('PU', 4)

# Column 16 - NUMBER OF UNITS
# TODO Consider if this is beneficial or hurts us
print_useful_info(df.iloc[:, 16], 16)
# Nothing to do

# Column 17 - OCCUPANCY TYPE
print_useful_info(df.iloc[:, 17], 17)
df.iloc[:, 17] = (df.iloc[:, 17]).replace('P', 0)
df.iloc[:, 17] = (df.iloc[:, 17]).replace('S', 1)
df.iloc[:, 17] = (df.iloc[:, 17]).replace('I', 2)
df.iloc[:, 17] = (df.iloc[:, 17]).replace('U', 3)

# Column 18 - PROPERTY STATE
print_useful_info(df.iloc[:, 18], 18)
states = set(df.iloc[:, 18])
for i, state in enumerate(states):
    df.iloc[:, 18] = (df.iloc[:, 18]).replace(state, i)

# Column 19 - ZIP CODE SHORT
print_useful_info(df.iloc[:, 19], 19)
# Nothing to do

# Column 20 - PRIMARY MORTGAGE INSURANCE PERCENT
# TODO Should default be 0 or mean()?
df.iloc[:, 20] = (df.iloc[:, 20]).fillna(0)
print_useful_info(df.iloc[:, 20], 20)

# Column 21 - PRODUCT TYPE
# TODO Check later if more values exist in other files
print_useful_info(df.iloc[:, 21], 21)
df.iloc[:, 21] = (df.iloc[:, 21]).replace('FRM', 0)

# Column 22 - CO-BORROWER CREDIT SCORE AT ORIGINATION
col_22_mean = (df.iloc[:, 22]).mean()
df.iloc[:, 22] = (df.iloc[:, 22]).fillna(col_22_mean)
print_useful_info(df.iloc[:, 22], 22)

# Column 23 - MORTGAGE INSURANCE TYPE
df.iloc[:, 23] = (df.iloc[:, 23]).fillna(0)
print_useful_info(df.iloc[:, 23], 23)

# Column 24 - MORTGAGE INSURANCE TYPE
print_useful_info(df.iloc[:, 24], 24)
df.iloc[:, 24] = (df.iloc[:, 24]).replace('N', 0)
df.iloc[:, 24] = (df.iloc[:, 24]).replace('Y', 1)

# Export to new file
df.to_csv('prep.txt')


