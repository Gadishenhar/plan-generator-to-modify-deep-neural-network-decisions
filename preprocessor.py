import pandas as pd
import sys


def print_useful_info(df_col, col_num):
    print('Column #', col_num)
    print('There are', len(df_col), 'total entries')
    print('There are', df_col.nunique(), 'unique values')
    print('There are', sum(df_col.isnull()), 'missing values')
    print('')

def main(acq_path, out_path):

    # Load data set
    df = pd.read_csv(acq_path, sep='|')

    # Set column names
    COL_NAMES = [
        'LOAN_ID',
        'ORIG_CHAN',
        'SELLER_NAME',
        'ORIG_INT_R',
        'ORIG_UPB',
        'ORIG_LOAN_T',
        'ORIG_DATE',
        'FST_PAY_DATE',
        'ORIG_LTV',
        'ORIG_CLTV',
        'NUM_BOR',
        'ORIG_DTIR',
        'BOR_C_SCORE',
        'FST_TIME_IND',
        'LOAN_PURPOSE',
        'PROP_TYPE',
        'NUM_UNITS',
        'OCC_TYPE',
        'PROP_STATE',
        'ZIP_SHORT',
        'PRIM_INS_PERC',
        'PROD_TYPE',
        'CO_BOR_C_SCORE',
        'INSUR_TYPE',
        'RELOC_IND'
    ]
    df.columns = COL_NAMES

    # Start by converting all entries to numerical values, fill missing values and remember which columns should be deleted
    to_be_deleted = []

    # Column 0 - LOAN IDENTIFIER
    if len(df.iloc[:, 0]) != (df.iloc[:, 0]).nunique():
        print('There are duplicate loan identifiers! Only saving last entry for each identifier.')
        df.drop_duplicates(subset=COL_NAMES[0], keep='last', inplace=True)
    print_useful_info(df.iloc[:, 0], 0)

    # We will not need this column, after we are done preprocessing
    to_be_deleted.append(COL_NAMES[0])

    # Column 1 - ORIGINATION CHANNEL
    print_useful_info(df.iloc[:, 1], 1)
    df.iloc[:, 1].replace('R', 1, inplace=True)
    df.iloc[:, 1].replace('C', 2, inplace=True)
    df.iloc[:, 1].replace('B', 3, inplace=True)

    # Column 2 - SELLER NAME
    # TODO Consider using values indicative of prevalence of sellers
    print_useful_info(df.iloc[:, 2], 2)
    sellers = set(df.iloc[:, 2])
    for i, seller in enumerate(sellers):
        df.iloc[:, 2].replace(seller, i, inplace=True)

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
    to_be_deleted.append(COL_NAMES[6])

    # Column 7 - FIRST PAYMENT DATE
    print_useful_info(df.iloc[:, 7], 7)
    to_be_deleted.append(COL_NAMES[7])

    # Column 8 - ORIGINAL LOAN-TO-VALUE (LTV)
    print_useful_info(df.iloc[:, 8], 8)
    # Nothing to do

    # Column 9 - ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)
    print_useful_info(df.iloc[:, 9], 9)
    # Nothing to do

    # Column 10 - NUMBER OF BORROWERS
    col_10_mean = df.iloc[:, 10].mean()
    df.iloc[:, 10].fillna(col_10_mean, inplace=True)
    print_useful_info(df.iloc[:, 10], 10)

    # Column 11 - ORIGINAL DEBT TO INCOME RATIO
    col_11_mean = df.iloc[:, 11].mean()
    df.iloc[:, 11].fillna(col_11_mean, inplace=True)
    print_useful_info(df.iloc[:, 11], 11)

    # Column 12 - BORROWER CREDIT SCORE AT ORIGINATION
    col_12_mean = df.iloc[:, 12].mean()
    df.iloc[:, 12].fillna(col_12_mean, inplace=True)
    print_useful_info(df.iloc[:, 12], 12)

    # Column 13 - FIRST TIME HOME BUYER INDICATOR
    print_useful_info(df.iloc[:, 13], 13)
    df.iloc[:, 13].replace('N', 0, inplace=True)
    df.iloc[:, 13].replace('Y', 1, inplace=True)
    df.iloc[:, 13].replace('U', 2, inplace=True)

    # Column 14 - LOAN PURPOSE
    print_useful_info(df.iloc[:, 14], 14)
    df.iloc[:, 14].replace('P', 0, inplace=True)
    df.iloc[:, 14].replace('C', 1, inplace=True)
    df.iloc[:, 14].replace('R', 2, inplace=True)
    df.iloc[:, 14].replace('U', 3, inplace=True)

    # Column 15 - PROPERTY TYPE
    print_useful_info(df.iloc[:, 15], 15)
    df.iloc[:, 15].replace('SF', 0, inplace=True)
    df.iloc[:, 15].replace('PU', 1, inplace=True)
    df.iloc[:, 15].replace('CO', 2, inplace=True)
    df.iloc[:, 15].replace('MH', 3, inplace=True)
    df.iloc[:, 15].replace('CP', 4, inplace=True)

    # TODO Consider if keeping this column is beneficial or hurts us
    # Column 16 - NUMBER OF UNITS
    print_useful_info(df.iloc[:, 16], 16)
    # Nothing to do

    # Column 17 - OCCUPANCY TYPE
    print_useful_info(df.iloc[:, 17], 17)
    df.iloc[:, 17].replace('P', 0, inplace=True)
    df.iloc[:, 17].replace('I', 1, inplace=True)
    df.iloc[:, 17].replace('S', 2, inplace=True)
    df.iloc[:, 17].replace('U', 3, inplace=True)

    # Column 18 - PROPERTY STATE
    # TODO Also sort by how common the values are?
    print_useful_info(df.iloc[:, 18], 18)
    states = set(df.iloc[:, 18])
    for i, state in enumerate(states):
        df.iloc[:, 18].replace(state, i, inplace=True)

    # Column 19 - ZIP CODE SHORT
    print_useful_info(df.iloc[:, 19], 19)
    # Nothing to do

    # Column 20 - PRIMARY MORTGAGE INSURANCE PERCENT
    # TODO Should default be 0 or mean()?
    df.iloc[:, 20].fillna(0, inplace=True)
    print_useful_info(df.iloc[:, 20], 20)

    # Column 21 - PRODUCT TYPE
    print_useful_info(df.iloc[:, 21], 21)
    df.iloc[:, 21].replace('FRM', 0, inplace=True)

    # It appears that there is only one possible value for this column. If that is the case, this column offers us nothing.
    if df.iloc[:, 21].nunique() > 1:
        print('There are more than one product type in this file! Consider keeping this column after all')
    else:
        to_be_deleted.append(COL_NAMES[21])

    # Column 22 - CO-BORROWER CREDIT SCORE AT ORIGINATION
    col_22_mean = (df.iloc[:, 22]).mean()
    df.iloc[:, 22].fillna(col_22_mean, inplace=True)
    print_useful_info(df.iloc[:, 22], 22)

    # Column 23 - MORTGAGE INSURANCE TYPE
    df.iloc[:, 23].fillna(0, inplace=True)
    print_useful_info(df.iloc[:, 23], 23)

    # Column 24 - RELOCATION MORTGAGE INDICATOR
    print_useful_info(df.iloc[:, 24], 24)
    df.iloc[:, 24].replace('N', 0, inplace=True)
    df.iloc[:, 24].replace('Y', 1, inplace=True)

    # Remove unnecessary columns
    print('Removing unnecessary columns...\n')
    df.drop(to_be_deleted, axis=1, inplace=True)

    # Normalization:
    print('Normalizing...\n')
    df = (df - df.mean()) / df.std()
    df = df / (df.max() - df.min())

    # Export to new file
    print('Exporting...')
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    acq_path = sys.argv[1]
    out_path = sys.argv[2]
    main(acq_path, out_path)