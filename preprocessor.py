import pandas as pd
import sys


def print_useful_info(acq_df_col, col_num):
    return 0
    #print('Column #', col_num)
    #print('There are', len(acq_df_col), 'total entries')
    #print('There are', acq_df_col.nunique(), 'unique values')
    #print('There are', sum(acq_df_col.isnull()), 'missing values')
    #print('')


def prep_acq_columns(df, COL_NAMES):

    to_be_deleted = []

    # Column 0 - LOAN IDENTIFIER
    if len(df.iloc[:, 0]) != (df.iloc[:, 0]).nunique():
        print('There are duplicate loan identifiers! Only saving last entry for each identifier.')
        df.drop_duplicates(subset=COL_NAMES[0], keep='last', inplace=True)
    print_useful_info(df.iloc[:, 0], 0)
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

    # If there is only one possible value in this column, it offers us nothing.
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

    # Drop any leftover entires that contains NaNs (should be a single entry, due to a single NaN in column 9)
    df.dropna(inplace=True)

    return df, to_be_deleted


def split_train_val_test(df, OUT_PATH, TRAIN_VAL_TEST_SPLIT):

    # To make sure all of our sets have the same probability distribution, and because there are so few '1' labels,
    # we will split the data frame by the label and split those individually
    zeros_df = df[df.iloc[:, -1] == 0]
    ones_df = df[df.iloc[:, -1] == 1]

    # Shuffle
    zeros_df = zeros_df.sample(frac=1).reset_index(drop=True)
    ones_df = ones_df.sample(frac=1).reset_index(drop=True)

    # Split the data set to train, validation and test sets
    ZEROS_LEN = len(zeros_df)
    ONES_LEN = len(ones_df)

    ZEROS_TRAIN_LEN = round(ZEROS_LEN * TRAIN_VAL_TEST_SPLIT[0])
    ONES_TRAIN_LEN = round(ONES_LEN * TRAIN_VAL_TEST_SPLIT[0])

    # The number of zeros should be only 9 times greater than the number of ones
    #ZEROS_TRAIN_LEN = min(ZEROS_TRAIN_LEN, 2*ONES_TRAIN_LEN)

    ZEROS_VAL_LEN = round(ZEROS_LEN * TRAIN_VAL_TEST_SPLIT[1])
    ONES_VAL_LEN = round(ONES_LEN * TRAIN_VAL_TEST_SPLIT[1])
    #ZEROS_VAL_LEN = min(ZEROS_VAL_LEN, 2 * ONES_VAL_LEN)

    ZEROS_TEST_LEN = ZEROS_LEN - ZEROS_TRAIN_LEN - ZEROS_VAL_LEN
    ONES_TEST_LEN = ONES_LEN - ONES_TRAIN_LEN - ONES_VAL_LEN
    #ZEROS_TEST_LEN = min(ZEROS_TEST_LEN, 2 * ONES_TEST_LEN)

    train = zeros_df.iloc[:ZEROS_TRAIN_LEN, :]
    train = train.append(ones_df.iloc[:ONES_TRAIN_LEN, :])
    train = train.sample(frac=1).reset_index(drop=True)

    PER0 = sum(train.iloc[:, -1] == 0) / len(train) * 100
    PER1 = 100 - PER0
    #print('The train data set contains ', PER0, 'zeros and ', PER1)

    val = zeros_df.iloc[ZEROS_TRAIN_LEN:(ZEROS_TRAIN_LEN + ZEROS_VAL_LEN), :]
    val = val.append(ones_df.iloc[ONES_TRAIN_LEN:(ONES_TRAIN_LEN + ONES_VAL_LEN), :])
    val = val.sample(frac=1).reset_index(drop=True)

    PER0 = sum(val.iloc[:, -1] == 0) / len(val) * 100
    PER1 = 100 - PER0
    #print('The validation data set contains ', PER0, 'zeros and ', PER1)

    test = zeros_df.iloc[-ZEROS_TEST_LEN:, :]
    test = test.append(ones_df.iloc[-ONES_TEST_LEN:, :])
    test = test.sample(frac=1).reset_index(drop=True)

    PER0 = sum(test.iloc[:, -1] == 0) / len(test) * 100
    PER1 = 100 - PER0
    #print('The test data set contains ', PER0, 'zeros and ', PER1)

    return (train, val, test)


def preprocess_single_pair(acq_path, per_path):

    # Load data set
    ACQ_COL_NAMES = [
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
    acq_df = pd.read_csv(acq_path, sep='|', names=ACQ_COL_NAMES)

    # First, convert all entries to numerical values, fill missing values and remember which columns should be deleted
    acq_df, to_be_deleted = prep_acq_columns(acq_df, ACQ_COL_NAMES)

    # Before we remove the identifier column, we need to process the performance data
    PER_COL_NAMES = [ACQ_COL_NAMES[0], 'DEFAULT']
    per_df = pd.read_csv(per_path, sep='|', usecols=[0, 15], names=PER_COL_NAMES)
    per_df.drop_duplicates(subset=ACQ_COL_NAMES[0], keep='last', inplace=True)

    # Replace all foreclosure dates (or their lack of) with 0 or 1 to represent whether a default took place
    per_df.fillna(0, inplace=True)
    per_df.loc[per_df[PER_COL_NAMES[1]] != 0, PER_COL_NAMES[1]] = 1
    per_df.iloc[:, -1] = per_df.iloc[:, -1].astype(int)

    # Merge with the acquisition data set, based on the load identifier
    #print('Merging with performance data...\n')
    df = pd.merge(acq_df, per_df, on=ACQ_COL_NAMES[0], how='inner')

    # Remove unnecessary columns
    #print('Removing unnecessary columns...\n')
    df.drop(to_be_deleted, axis=1, inplace=True)

    # Normalization (note this ignores the labels):
    #print('Normalizing...\n')
    features = df.iloc[:, :-1]
    df.iloc[:, :-1] = (features - features.mean()) / features.std()
    df.iloc[:, :-1] = (features - features.min()) / (features.max() - features.min())

    # TODO Delete this if new code version works
    # This normalization included our labels, which are now ruined. However, there are still only two values. We need to
    # convert them back to 0 and 1
    #df.iloc[:, -1:].replace(df.iloc[:, -1:].min(), 0)
    #df.iloc[:, -1:].replace(df.iloc[:, -1:].max(), 1)

    #print('Splitting to train, validation and test sets...\n')
    return split_train_val_test(df, OUT_PATH, TRAIN_VAL_TEST_SPLIT)


def main(acq_path, per_path, out_path, TRAIN_VAL_TEST_SPLIT):

    train, val, test = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    for year in range(2000, 2018+1):
        for quarter in range(1, 4+1):
            acquisition_file_name = acq_path + '/Acquisition_' +str(year) + 'Q' + str(quarter) + '.txt'
            performance_file_name = per_path + '/Performance_' +str(year) + 'Q' + str(quarter) + '.txt'
            print('Currently processing', year, 'quarter', quarter)
            new_train, new_val, new_test = preprocess_single_pair(acquisition_file_name, performance_file_name)

            #train = train.append(new_train)
            #val = val.append(new_val)
            #test = test.append(new_test)

            # Shuffle one last time, so that the entry year will not matter, and save the sets
            new_train.sample(frac=1).to_csv(OUT_PATH + 'train' + str(year) + 'Q' + str(quarter) + '.txt', index=False, header=False)
            new_val.sample(frac=1).to_csv(OUT_PATH + 'val' + str(year) + 'Q' + str(quarter) + '.txt', index=False, header=False)
            new_test.sample(frac=1).to_csv(OUT_PATH + 'test' + str(year) + 'Q' + str(quarter) + '.txt', index=False, header=False)


if __name__ == '__main__':
    ACQ_PATH = 'Dataset/acquisition'
    PER_PATH = 'Dataset/performance'
    OUT_PATH = 'Dataset/prep_unbiased/'
    TRAIN_VAL_TEST_SPLIT = [0.6, 0.2, 0.2]
    main(ACQ_PATH, PER_PATH, OUT_PATH, TRAIN_VAL_TEST_SPLIT)
