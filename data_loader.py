import pandas as pd
from pathlib import Path
import logging

DATA_DIR = Path("./data")

def load_data(dataset_name):

    """Loads the specified dataset."""

    logging.info(f"Loading dataset: {dataset_name}")

    if dataset_name == "kaggle":
        file_path = DATA_DIR / "creditcard.csv"

        df = pd.read_csv(file_path)
        df = df.dropna() 

        logging.info(f"Loaded Kaggle data. Shape: {df.shape}")

        X = df.drop('Class', axis=1)
        y = df['Class']

        return X, y

    elif dataset_name == "ieee-cis":

        identity     = pd.read_csv(DATA_DIR / 'train_identity.csv', usecols=['TransactionID', 'DeviceType', 'DeviceInfo'])
        transactions = pd.read_csv(DATA_DIR / 'train_transaction.csv', usecols=['TransactionID','TransactionAmt', 'ProductCD', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'isFraud'])

        df = pd.merge(transactions, identity, on="TransactionID", how="left")
        df = df.drop('TransactionID', axis=1)

        classe     = df['isFraud']
        product_cd = df['ProductCD']
        card4      = df['card4']
        card6      = df['card6']
        deviceType = df['DeviceType']
        deviceInfo = df['DeviceInfo']
        m1         = df['M1']
        m2         = df['M2']
        m3         = df['M3']
        m4         = df['M4']
        m5         = df['M5']
        m6         = df['M6']
        m7         = df['M7']
        m8         = df['M8']
        m9         = df['M9']
        ta         = df['TransactionAmt']

        df.drop(['isFraud', 'ProductCD', 'card4', 'card6', 'TransactionAmt', 'DeviceType', 'DeviceInfo', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'], axis=1, inplace=True)
        df.insert(0, 'Class', classe)
        df.insert(1, 'PCD', product_cd)
        df.insert(2, 'card4', card4)
        df.insert(3, 'card6', card6)
        df.insert(4, 'DT', deviceType)
        df.insert(5, 'DI', deviceInfo)
        df.insert(6, 'M1', m1)
        df.insert(7, 'M2', m2)
        df.insert(8, 'M3', m3)
        df.insert(9, 'M4', m4)
        df.insert(10, 'M5', m5)
        df.insert(11, 'M6', m6)
        df.insert(12, 'M7', m7)
        df.insert(13, 'M8', m8)
        df.insert(14, 'M9', m9)
        df.insert(15, 'TA', ta)

        logging.info(f"Loaded and merged IEEE-CIS data. Shape: {df.shape}")

        X = df.drop('Class', axis=1)
        y = df['Class']

        return X, y

    elif dataset_name == "synthetic":
        file_path = DATA_DIR / "transactions_480m.csv"
        df = pd.read_csv(file_path)

        cols = ['Year', 'Month', 'Day', 'Time', 'Amount', 'Use Chip', 'Merchant Name', 'Merchant City', 'Merchant State', 'Zip', 'MCC', 'Errors?', 'Is Fraud?']
        df = df[cols]

        df['Class']  = df['Is Fraud?'].astype('category')
        cat_columns = df.select_dtypes(['category']).columns
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
        df.drop(['Is Fraud?'], axis=1, inplace=True)

        df['Time'] = pd.to_datetime(df['Time'], format="%H:%M").astype('int64')/ 10**9

        df['UC'] = df['Use Chip']
        df['MN'] = df['Merchant Name']
        df['MC'] = df['Merchant City']
        df['MS'] = df['Merchant State']
        df['errors'] = df['Errors?']
        df.drop(['Use Chip', 'Merchant Name', 'Merchant City', 'Merchant State', 'Errors?'], axis=1, inplace=True)

        logging.info(f"Loaded synthetic data. Shape: {df.shape}")
        X = df.drop('Class', axis=1)
        y = df['Class']
        return X, y

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
