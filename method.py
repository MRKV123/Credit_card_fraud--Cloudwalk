import pandas as pd
import numpy as np

def dt_engineering(df):

    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df["month"] = df['transaction_date'].dt.month
    df["day"] = df['transaction_date'].dt.day
    df["hour"] = df['transaction_date'].dt.hour
    df["min"] = df['transaction_date'].dt.minute
    df['sec'] = df['transaction_date'].dt.second
    df["day_name"]=df.transaction_date.dt.day_name()
    df = df.drop(columns='transaction_date', axis=1)

    return df

def apply_business_rules(df):
    # Existing rules
    df['score_in_transaction_amount'] = np.where(df['transaction_amount'] > 2000, 120, 271)
    df['score_hour'] = np.where(df['hour'] > 12, 316, 75)
    df['score_merchant'] = df.groupby('merchant_id')['transaction_amount'].transform(lambda x: (x.sum() * 0.05) if x.any() else 0)
    df['score_user'] = df.groupby('user_id')['transaction_amount'].transform(lambda x: (x.sum() * 0.1) if x.any() else 0)
    if 'device_category' in df.columns:
        score_mapping = {'1 device': 258, '2 devices': 36, '3 devices': 5, '4 devices': 25, '5 devices or more': 0}
        df['score_device_category'] = df['device_category'].map(score_mapping).fillna(0)
    else:
        df['score_device_category'] = 0

    # Calculate total score
    df['score_in_device_id'] = np.where(df['device_id'].isna(), 67, 0)
    df['total_score'] = df['score_in_transaction_amount'] + df['score_hour'] + df['score_in_device_id'] + df['score_merchant'] + df['score_user'] + df['score_device_category']

    return df

def filter_fraudulent_transactions(df):
    fraud_df = pd.read_csv(r'C:\Users\nando\OneDrive\Área de Trabalho\CLOUDWALK_API\frauds.csv')
    fraud_ids = fraud_df[fraud_df['has_cbk'] == True]['user_id'].unique().tolist()
    return fraud_ids

def status_update(df, fraudulent_users):
    df['status'] = 'approved'

    fraud_ids = filter_fraudulent_transactions(df) 
    df.loc[df['user_id'].isin(fraud_ids), 'status'] = 'denied'
    df.loc[df['total_score'] > 577, 'status'] = 'denied'
    return df
