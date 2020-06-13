import pandas as pd
import os.path

# A full list of the columns in the complete merged dataset, before any columns are dropped
COL_NAMES = [
    'LOAN_ID',       'ORIG_CHAN', 'SELLER_NAME',    'ORIG_INT_R',   'ORIG_UPB',
    'ORIG_LOAN_T',   'ORIG_DATE', 'FST_PAY_DATE',   'ORIG_LTV',     'ORIG_CLTV',
    'NUM_BOR',       'ORIG_DTIR', 'BOR_C_SCORE',    'FST_TIME_IND', 'LOAN_PURPOSE',
    'PROP_TYPE',     'NUM_UNITS', 'OCC_TYPE',       'PROP_STATE',   'ZIP_SHORT',
    'PRIM_INS_PERC', 'PROD_TYPE', 'CO_BOR_C_SCORE', 'INSUR_TYPE',   'RELOC_IND',
    'DEFAULT'
]

CLEANED_COL_NAMES =  [
                     'ORIG_CHAN', 'SELLER_NAME',    'ORIG_INT_R',   'ORIG_UPB',
    'ORIG_LOAN_T',                                  'ORIG_LTV',     'ORIG_CLTV',
    'NUM_BOR',       'ORIG_DTIR', 'BOR_C_SCORE',    'FST_TIME_IND', 'LOAN_PURPOSE',
                     'NUM_UNITS', 'OCC_TYPE',       'PROP_STATE',   'ZIP_SHORT',
    'PRIM_INS_PERC', 'PROD_TYPE', 'CO_BOR_C_SCORE', 'INSUR_TYPE',   'RELOC_IND',
    'DEFAULT'
]


def print_useful_info(acq_df_col, col_num):
    return 0
    print('Processing column', col_num)
    print('Column #', col_num)
    print('There are', len(acq_df_col), 'total entries')
    print('There are', acq_df_col.nunique(), 'unique values')
    print('There are', sum(acq_df_col.isnull()), 'missing values')
    print('')


def handle_empty_fields(df):
    """
    Handles all of the empty fields by either filling them with the mean value of that column, or by dropping that
    line altogether.
    :param df: The data frame of the data set.
    :return: The data frame after all empty fields have been taken care of.
    """

    # Column 10 - NUMBER OF BORROWERS
    col_10_mean = df.iloc[:, 10].mean()
    df.iloc[:, 10].fillna(col_10_mean, inplace=True)

    # Column 11 - ORIGINAL DEBT TO INCOME RATIO
    col_11_mean = df.iloc[:, 11].mean()
    df.iloc[:, 11].fillna(col_11_mean, inplace=True)

    # Column 12 - BORROWER CREDIT SCORE AT ORIGINATION
    col_12_mean = df.iloc[:, 12].mean()
    df.iloc[:, 12].fillna(col_12_mean, inplace=True)

    # Column 20 - PRIMARY MORTGAGE INSURANCE PERCENT
    df.iloc[:, 20].fillna(0, inplace=True)

    # Column 22 - CO-BORROWER CREDIT SCORE AT ORIGINATION
    col_22_mean = (df.iloc[:, 22]).mean()
    df.iloc[:, 22].fillna(col_22_mean, inplace=True)

    # Column 23 - MORTGAGE INSURANCE TYPE
    df.iloc[:, 23].fillna(0, inplace=True)

    # Drop any leftover entires that contains NaNs
    df.dropna(inplace=True)

    return df


def prep_columns(df, seller_names, property_states):
    """
    Handles all empty fields, and tokenizes the entire data set.
    :param df: The data frame of the data set.
    :param seller_names: A list of all of the seller names. Will be tokenized based on the list's order.
    :param property_states: A list of all of the property states. Will be tokenized based on the list's order.
    :return: A cleaned and tokenized data frame.
    """

    handle_empty_fields(df)

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
    print_useful_info(df.iloc[:, 2], 2)
    for i, seller in enumerate(seller_names):
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
    print_useful_info(df.iloc[:, 10], 10)
    # Nothing to do

    # Column 11 - ORIGINAL DEBT TO INCOME RATIO
    print_useful_info(df.iloc[:, 11], 11)
    # Nothing to do

    # Column 12 - BORROWER CREDIT SCORE AT ORIGINATION
    print_useful_info(df.iloc[:, 12], 12)
    # Nothing to do

    # Column 13 - FIRST TIME HOME BUYER INDICATOR
    print_useful_info(df.iloc[:, 13], 13)
    df.iloc[:, 13].replace('N', 1, inplace=True)
    df.iloc[:, 13].replace('Y', 2, inplace=True)
    df.iloc[:, 13].replace('U', 3, inplace=True)

    # Column 14 - LOAN PURPOSE
    print_useful_info(df.iloc[:, 14], 14)
    df.iloc[:, 14].replace('P', 1, inplace=True)
    df.iloc[:, 14].replace('C', 2, inplace=True)
    df.iloc[:, 14].replace('R', 3, inplace=True)
    df.iloc[:, 14].replace('U', 4, inplace=True)

    # Column 15 - PROPERTY TYPE
    print_useful_info(df.iloc[:, 15], 15)
    df.iloc[:, 15].replace('SF', 1, inplace=True)
    df.iloc[:, 15].replace('PU', 2, inplace=True)
    df.iloc[:, 15].replace('CO', 3, inplace=True)
    df.iloc[:, 15].replace('MH', 4, inplace=True)
    df.iloc[:, 15].replace('CP', 5, inplace=True)

    # Column 16 - NUMBER OF UNITS
    print_useful_info(df.iloc[:, 16], 16)
    # Nothing to do

    # Column 17 - OCCUPANCY TYPE
    print_useful_info(df.iloc[:, 17], 17)
    df.iloc[:, 17].replace('P', 1, inplace=True)
    df.iloc[:, 17].replace('I', 2, inplace=True)
    df.iloc[:, 17].replace('S', 3, inplace=True)
    df.iloc[:, 17].replace('U', 4, inplace=True)

    # Column 18 - PROPERTY STATE
    print_useful_info(df.iloc[:, 18], 18)
    for i, state in enumerate(property_states):
        df.iloc[:, 18].replace(state, i, inplace=True)

    # Column 19 - ZIP CODE SHORT
    print_useful_info(df.iloc[:, 19], 19)
    # Nothing to do

    # Column 20 - PRIMARY MORTGAGE INSURANCE PERCENT
    print_useful_info(df.iloc[:, 20], 20)
    # Nothing to do

    # Column 21 - PRODUCT TYPE
    print_useful_info(df.iloc[:, 21], 21)
    df.iloc[:, 21].replace('FRM', 0, inplace=True)

    # If there is only one possible value in this column, it offers us nothing.
    if df.iloc[:, 21].nunique() > 1:
        print('There are more than one product type in this file! Consider keeping this column after all')
    else:
        to_be_deleted.append(COL_NAMES[21])

    # Column 22 - CO-BORROWER CREDIT SCORE AT ORIGINATION
    print_useful_info(df.iloc[:, 22], 22)
    # Nothing to do

    # Column 23 - MORTGAGE INSURANCE TYPE
    print_useful_info(df.iloc[:, 23], 23)
    # Nothing to do

    # Column 24 - RELOCATION MORTGAGE INDICATOR
    print_useful_info(df.iloc[:, 24], 24)
    df.iloc[:, 24].replace('N', 1, inplace=True)
    df.iloc[:, 24].replace('Y', 2, inplace=True)

    # Delete unneeded columns
    df.drop(to_be_deleted, axis=1, inplace=True)

    return df


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

    ZEROS_VAL_LEN = round(ZEROS_LEN * TRAIN_VAL_TEST_SPLIT[1])
    ONES_VAL_LEN = round(ONES_LEN * TRAIN_VAL_TEST_SPLIT[1])

    ZEROS_TEST_LEN = ZEROS_LEN - ZEROS_TRAIN_LEN - ZEROS_VAL_LEN
    ONES_TEST_LEN = ONES_LEN - ONES_TRAIN_LEN - ONES_VAL_LEN

    train = zeros_df.iloc[:ZEROS_TRAIN_LEN, :]
    train = train.append(ones_df.iloc[:ONES_TRAIN_LEN, :])
    train = train.sample(frac=1).reset_index(drop=True)

    PER0 = sum(train.iloc[:, -1] == 0) / len(train) * 100
    PER1 = 100 - PER0
    print('The train data set contains ', PER0, 'zeros and ', PER1, 'ones')

    val = zeros_df.iloc[ZEROS_TRAIN_LEN:(ZEROS_TRAIN_LEN + ZEROS_VAL_LEN), :]
    val = val.append(ones_df.iloc[ONES_TRAIN_LEN:(ONES_TRAIN_LEN + ONES_VAL_LEN), :])
    val = val.sample(frac=1).reset_index(drop=True)

    PER0 = sum(val.iloc[:, -1] == 0) / len(val) * 100
    PER1 = 100 - PER0
    print('The validation data set contains ', PER0, 'zeros and ', PER1, 'ones')

    test = zeros_df.iloc[-ZEROS_TEST_LEN:, :]
    test = test.append(ones_df.iloc[-ONES_TEST_LEN:, :])
    test = test.sample(frac=1).reset_index(drop=True)

    PER0 = sum(test.iloc[:, -1] == 0) / len(test) * 100
    PER1 = 100 - PER0
    print('The test data set contains ', PER0, 'zeros and ', PER1, 'ones')

    return (train, val, test)


def merge_acq_per_data(acq_path, per_path):

    # Before we remove the identifier column, we need to process the performance data
    PER_COL_NAMES = ['LOAN_ID', 'DEFAULT'] #Choose two columns from the performance data: the loan's identifier + "DEFAULT": the date in which a costumer's house was confiscated
    per_df = pd.read_csv(per_path, sep='|', usecols=[0, 15], names=PER_COL_NAMES) #load the aquisition features
    per_df.drop_duplicates(subset='LOAN_ID', keep='last', inplace=True) #removing all duplicate rows from aquisition

    # Replace all foreclosure dates (or their lack of) with 0 or 1 to represent whether a default took place
    per_df.fillna(0, inplace=True) #Filling all the NaNs with zeroes
    per_df.loc[per_df[PER_COL_NAMES[1]] != 0, PER_COL_NAMES[1]] = 1 #in any case there's a result other than 0 it means that the house indeed was confiscated so we write "1"
    per_df.iloc[:, -1] = per_df.iloc[:, -1].astype(int) #Converting to integers

    acq_df = pd.read_csv(acq_path, sep="|", names=COL_NAMES[:-1])

    # Merge with the acquisition data set, based on the load identifier
    #print('Merging with performance data...\n')
    df = pd.merge(acq_df, per_df, on='LOAN_ID', how='inner') #merge according to Loan indentifier
    return df


def bias_labels(df, zeros_to_ones_ratio):

    zeros_df = df[df.iloc[:, -1] == 0]
    ZEROS_LEN = len(zeros_df)
    ones_df = df[df.iloc[:, -1] == 1]
    ONES_LEN = len(ones_df)

    # Set new length for zeros, based on the ratio
    NEW_ZEROS_LEN = min(ZEROS_LEN, ONES_LEN * zeros_to_ones_ratio)
    zeros_df = zeros_df.sample(frac=NEW_ZEROS_LEN/ZEROS_LEN)

    df = zeros_df.append(ones_df).sample(frac=1)

    # Print some final stats
    print('Total len', len(df))
    print('Number of zeros', NEW_ZEROS_LEN),
    print('Percent of zeros', NEW_ZEROS_LEN / len(df) * 100)
    print('Number of ones', ONES_LEN),
    print('Percent of ones', ONES_LEN / len(df) * 100)
    print()

    # Append and shuffle the result
    return df


def main(acq_path, per_path, out_path, TRAIN_VAL_TEST_SPLIT):


    # Iteration 1: go over all of the acquisition files, and merge them with the appropriate performance data
    # If the merged data set files already exist, we can skip this step
    force_iter_1 = True
    if (not os.path.isfile(out_path + 'AcqPer_2018Q4.txt')) or force_iter_1:
        for year in range(2000, 2018+1):
            for quarter in range(1, 4+1):
                suffix = str(year) + 'Q' + str(quarter) + '.txt'
                acquisition_file_name = acq_path + 'Acquisition_' + suffix
                performance_file_name = per_path + 'Performance_' + suffix
                print('Merging data from the year', year, 'quarter', quarter, '...')
                merged_df = merge_acq_per_data(acquisition_file_name, performance_file_name)
                merged_df.to_csv(out_path + 'AcqPer_' + suffix, index=False, header=False)
"""
    # Iteration 2: go over all of the merged data sets and extract from them all of the unique values from the seller
    # name and property state columns
    force_skip_iter_2 = False
    if not force_skip_iter_2:
        print('Extracting unique sellers and property states...')
        seller_names = set()
        property_states = set()
        for year in range(2001, 2018+1):
            for quarter in range(1, 4+1):
                suffix = str(year) + 'Q' + str(quarter) + '.txt'
                file_name = out_path + 'AcqPer_' + suffix
                print('Processing year', year, 'quarter', quarter, '...')
                seller_names = seller_names.union(set(pd.read_csv(file_name, usecols=[2]).iloc[:, 0].unique()))
                property_states = property_states.union(set(pd.read_csv(file_name, usecols=[18]).iloc[:, 0].unique()))

        # Now that we have all possible values saved, convert the set to list, to assure deterministic order
        seller_names = list(seller_names)
        property_states = list(property_states)

    # Save the final lists (this was after running previous section
    seller_names = ['WELLS FARGO BANK,  NA', 'AMERIHOME MORTGAGE COMPANY, LLC', 'METLIFE HOME LOANS LLC', 'SANTANDER BANK, NATIONAL ASSOCIATION', 'PACIFIC UNION FINANCIAL, LLC', 'CASHCALL, INC.', 'PULTE MORTGAGE, L.L.C.', 'CMG MORTGAGE, INC', 'GMAC MORTGAGE, LLC', 'CAPITAL ONE, NATIONAL ASSOCIATION', 'USAA FEDERAL SAVINGS BANK', 'FIRST BANK DBA FIRST BANK MORTGAGE', 'LAKEVIEW LOAN SERVICING, LLC', 'FLAGSTAR BANK, FSB', 'PMT CREDIT RISK TRANSFER TRUST 2015-2', 'FDIC, RECEIVER, INDYMAC FEDERAL BANK FSB', 'CITIMORTGAGE, INC.', 'SUNTRUST MORTGAGE INC.', 'REGIONS BANK', 'HSBC BANK USA, NATIONAL ASSOCIATION', 'STONEGATE MORTGAGE CORPORATION', 'PMTT4', 'TRUIST BANK (FORMERLY SUNTRUST BANK)', 'CHICAGO MORTGAGE SOLUTIONS DBA INTERBANK MORTGAGE COMPANY', 'RBC MORTGAGE COMPANY', 'NYCB MORTGAGE COMPANY, LLC', 'FRANKLIN AMERICAN MORTGAGE COMPANY', 'THE BRANCH BANKING AND TRUST COMPANY', 'UNITED SHORE FINANCIAL SERVICES, LLC D/B/A UNITED WHOLESALE MORTGAGE', 'HOMEWARD RESIDENTIAL, INC.', 'NETBANK FUNDING SERVICES', 'COLORADO FEDERAL SAVINGS BANK', 'FREMONT BANK', 'PHH MORTGAGE CORPORATION (USAA FEDERAL SAVINGS BANK)', 'HOMEBRIDGE FINANCIAL SERVICES, INC.', 'SIERRA PACIFIC MORTGAGE COMPANY, INC.', 'FEDERAL HOME LOAN BANK OF CHICAGO', 'PROSPECT MORTGAGE, LLC', 'ASSOCIATED BANK, NA', 'PMT CREDIT RISK TRANSFER TRUST 2016-1', 'JPMORGAN CHASE BANK, NATIONAL ASSOCIATION', 'AMTRUST BANK', 'JPMORGAN CHASE BANK, NA', 'PRINCIPAL RESIDENTIAL MORTGAGE CAPITAL RESOURCES, LLC', 'GMAC MORTGAGE, LLC (USAA FEDERAL SAVINGS BANK)', 'U.S. BANK N.A.', 'BISHOPS GATE RESIDENTIAL MORTGAGE TRUST', 'GUILD MORTGAGE COMPANY', 'OTHER', 'EAGLE HOME MORTGAGE, LLC', 'WELLS FARGO CREDIT RISK TRANSFER SECURITIES TRUST 2015', 'EVERBANK', 'FAIRWAY INDEPENDENT MORTGAGE CORPORATION', 'ROUNDPOINT MORTGAGE COMPANY', 'THIRD FEDERAL SAVINGS AND LOAN', 'SUNTRUST BANK', 'NATIONSTAR MORTGAGE, LLC', 'PNC BANK, N.A.', 'METLIFE BANK, NA', 'J.P. MORGAN MADISON AVENUE SECURITIES TRUST, SERIES 2015-1', 'FLAGSTAR CAPITAL MARKETS CORPORATION', 'IMPAC MORTGAGE CORP.', 'UNITED SHORE FINANCIAL SERVICES, LLC DBA UNITED WHOLESALE MORTGAGE', 'LOANDEPOT.COM, LLC', 'ALLY BANK', 'QUICKEN LOANS INC.', 'THE HUNTINGTON NATIONAL BANK', 'CHICAGO MORTGAGE SOLUTIONS DBA INTERFIRST MORTGAGE COMPANY', 'WELLS FARGO BANK, N.A.', 'J.P. MORGAN MADISON AVENUE SECURITIES TRUST, SERIES 2014-1', 'DITECH FINANCIAL LLC', 'BANK OF AMERICA, N.A.', 'CHASE HOME FINANCE, LLC', 'CHASE HOME FINANCE', 'CHASE HOME FINANCE (CIE 1)', 'AMERISAVE MORTGAGE CORPORATION', 'MOVEMENT MORTGAGE, LLC', 'FIRST TENNESSEE BANK NATIONAL ASSOCIATION', 'FINANCE OF AMERICA MORTGAGE LLC', 'PENNYMAC CORP.', 'CHASE HOME FINANCE FRANKLIN AMERICAN MORTGAGE COMPANY', 'WITMER FUNDING, LLC', 'JP MORGAN CHASE BANK, NA', 'IRWIN MORTGAGE, CORPORATION', 'USAA DIRECT DELIVERY', 'CALIBER HOME LOANS, INC.', 'DOWNEY SAVINGS AND LOAN ASSOCIATION, F.A.', 'FLEET NATIONAL BANK', 'FREEDOM MORTGAGE CORP.', 'STEARNS LENDING, LLC', 'HARWOOD STREET FUNDING I, LLC', 'CITIZENS BANK, NATIONAL ASSOCIATION', 'NEW YORK COMMUNITY BANK', 'PHH MORTGAGE CORPORATION', 'FIFTH THIRD BANK', 'PROVIDENT FUNDING ASSOCIATES, L.P.']
    property_states = ['MA', 'GA', 'MS', 'IN', 'WI', 'KS', 'AK', 'IL', 'NH', 'PA', 'HI', 'NM', 'NE', 'IA', 'AL', 'CT', 'TN', 'ID', 'NJ', 'ME', 'MI', 'UT', 'GU', 'ND', 'AZ', 'RI', 'OK', 'NY', 'MN', 'VA', 'AR', 'NC', 'SD', 'DE', 'VT', 'SC', 'TX', 'CA', 'WA', 'CO', 'OR', 'WY', 'MT', 'FL', 'MO', 'DC', 'PR', 'WV', 'VI', 'LA', 'MD', 'NV', 'KY', 'OH']

    # Iteration 3: go over each data set file, replace all string values with numerical values, fill missing entries,
    # drop any remaining NaNs and remove some columns. Data is NOT normalized yet.
    # If the tokenized data set files already exist, we can skip this step
    force_iter_3 = True
    if (not os.path.isfile(out_path + 'TokAcqPer_2018Q4.txt')) or force_iter_3:
        for year in range(2001, 2018+1):
            for quarter in range(1, 4+1):
                suffix = str(year) + 'Q' + str(quarter) + '.txt'
                file_name = out_path + 'AcqPer_' + suffix
                print('Tokenizing year', year, 'quarter', quarter, '...')
                df = pd.read_csv(file_name, names=COL_NAMES)
                prep_columns(df, seller_names, property_states).to_csv(out_path + 'TokAcqPer_' + suffix, index=False, header=False)



    # The next step must be done manually (for now) outside of python:
    # * merge all of the text files to one (in cmd: copy TokAcqPer* TokMerged.txt)
    # * save the result of df.describe() to stats.txt (in cmd: df.describe().to_csv('stats.txt')
    # TODO Write code for it here


    # Iter 4: Even though we have a single tokenized file, we need to normalize each quarter separately.
    if os.path.isfile(out_path + 'stats.txt') and (not os.path.isfile(out_path + 'NormTokAcqPer_2018Q4.txt')):
        stats_df = pd.read_csv(out_path + 'stats.txt')
        for year in range(2001, 2018+1):
            for quarter in range(1, 4+1):
                suffix = str(year) + 'Q' + str(quarter) + '.txt'
                file_name = out_path + 'TokAcqPer_' + suffix
                print('Normalizing year', year, 'quarter', quarter, '...')
                df = pd.read_csv(file_name, names=CLEANED_COL_NAMES)
                # .values.squeeze() because stack overflow says so
                features_mean = stats_df.loc[1][1:-1].astype(float).values.squeeze()  # mean shows up in the second row of describe()
                features_std = stats_df.loc[2][1:-1].astype(float).values.squeeze()  # std is third row
                features_min = stats_df.loc[3][1:-1].astype(float).values.squeeze()  # min is fourth row
                features_max = stats_df.loc[7][1:-1].astype(float).values.squeeze()  # max is eighth row
                df.iloc[:, :-1] = (df.iloc[:, :-1] - features_mean) / features_std
                df.iloc[:, :-1] = (df.iloc[:, :-1] - features_mean) / (features_max - features_min)
                df.to_csv(out_path + 'NormTokAcqPer_' + suffix, index=False, header=False)

    # Iter 5: Next, we need to go over each normalized tokenized data set file, and split it to a train file, a
    # validation file and a test file, using the given split.
    if os.path.isfile(out_path + 'NormTokAcqPer_2018Q4.txt'):
        for year in range(2001, 2018+1):
            for quarter in range(1, 4+1):
                suffix = str(year) + 'Q' + str(quarter) + '.txt'
                file_name = out_path + 'NormTokAcqPer_' + suffix
                print('Splitting year', year, 'quarter', quarter, '...')
                df = pd.read_csv(file_name, names=CLEANED_COL_NAMES)
                train, val, test = split_train_val_test(df, OUT_PATH, TRAIN_VAL_TEST_SPLIT)
                train.to_csv(OUT_PATH + 'train' + suffix, index=False, header=False)
                val.to_csv(OUT_PATH + 'val' + suffix, index=False, header=False)
                test.to_csv(OUT_PATH + 'test' + suffix, index=False, header=False)

  

    # The next step must be done manually (for now) outside of python:
    # * merge all of the new files to three (train, val and test)
    # copy train2* train.txt
    # copy val2* val.txt
    # copy test2* test.txt
    # TODO Write code for it here


    # Next, we open the train, val and test files and bias the labels

    # First time, a 10%, 90% bias
    print('Biasing train with 1-9 ratio...')
    df = pd.read_csv(OUT_PATH + 'train.txt')
    ZEROS_LEN = len(df[df.iloc[:, -1] == 0])
    ONES_LEN = len(df[df.iloc[:, -1] == 1])
    TOT_LEN = ZEROS_LEN + ONES_LEN
    print('Total len', len(df))
    print('Number of zeros', ZEROS_LEN),
    print('Percent of zeros', ZEROS_LEN / TOT_LEN * 100)
    print('Number of ones', ONES_LEN),
    print('Percent of ones', ONES_LEN / TOT_LEN * 100)
    print()
    bias_labels(df, 9).to_csv(OUT_PATH + 'biased_10_90_train.txt', index=False, header=False)

    print('Biasing validation with 1-9 ratio...')
    df = pd.read_csv(OUT_PATH + 'val.txt')
    ZEROS_LEN = len(df[df.iloc[:, -1] == 0])
    ONES_LEN = len(df[df.iloc[:, -1] == 1])
    TOT_LEN = ZEROS_LEN + ONES_LEN
    print('Total len', len(df))
    print('Number of zeros', ZEROS_LEN),
    print('Percent of zeros', ZEROS_LEN / TOT_LEN * 100)
    print('Number of ones', ONES_LEN),
    print('Percent of ones', ONES_LEN / TOT_LEN * 100)
    print()
    bias_labels(df, 9).to_csv(OUT_PATH + 'biased_10_90_val.txt', index=False, header=False)

    print('Biasing test with 1-9 ratio...')
    df = pd.read_csv(OUT_PATH + 'test.txt')
    ZEROS_LEN = len(df[df.iloc[:, -1] == 0])
    ONES_LEN = len(df[df.iloc[:, -1] == 1])
    TOT_LEN = ZEROS_LEN + ONES_LEN
    print('Total len', len(df))
    print('Number of zeros', ZEROS_LEN),
    print('Percent of zeros', ZEROS_LEN / TOT_LEN * 100)
    print('Number of ones', ONES_LEN),
    print('Percent of ones', ONES_LEN / TOT_LEN * 100)
    print()
    bias_labels(df, 9).to_csv(OUT_PATH + 'biased_10_90_test.txt', index=False, header=False)


    # Then again, for 33-66 split
    print('Biasing train with 1-2 ratio...')
    df = pd.read_csv(OUT_PATH + 'train.txt')
    bias_labels(df, 2).to_csv(OUT_PATH + 'biased_33_66_train.txt', index=False, header=False)

    print('Biasing validation with 1-2 ratio...')
    df = pd.read_csv(OUT_PATH + 'val.txt')
    bias_labels(df, 2).to_csv(OUT_PATH + 'biased_33_66_val.txt', index=False, header=False)

    print('Biasing test with 1-2 ratio...')
    df = pd.read_csv(OUT_PATH + 'test.txt')
    bias_labels(df, 2).to_csv(OUT_PATH + 'biased_33_66_test.txt', index=False, header=False)
"""


if __name__ == '__main__':
    ACQ_PATH = 'Dataset/acquisition/'
    PER_PATH = 'Dataset/performance/'
    OUT_PATH = 'Dataset/prep_unbiased/'
    TRAIN_VAL_TEST_SPLIT = [0.6, 0.2, 0.2]
    main(ACQ_PATH, PER_PATH, OUT_PATH, TRAIN_VAL_TEST_SPLIT)
