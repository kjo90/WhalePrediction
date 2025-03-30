import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def load_data():
    """Load necessary datasets."""
    trx_data = pd.read_csv('../data/trx_data.csv')
    profile = pd.read_csv('../data/profile.csv')
    train_label = pd.read_csv('../data/train_label.csv')
    return trx_data, profile, train_label

def process_transaction_data(trx_data):
    """Process transaction data to extract features."""
    trx_data['transaction_time'] = pd.to_datetime(trx_data['transaction_time'])
    first_transaction = trx_data.groupby('user_id')['transaction_time'].min().reset_index(name='first_transaction')
    last_transaction = trx_data.groupby('user_id')['transaction_time'].max().reset_index(name='last_transaction')
    current_date = pd.to_datetime('today')
    recency_of_transaction = (current_date - last_transaction['last_transaction']).dt.days
    gtv_stats = trx_data.groupby('user_id').agg(
        gtv_count=('gtv', 'count'),
        gtv_max=('gtv', 'max'),
        gtv_sum=('gtv', 'sum'),
        gtv_mean=('gtv', 'mean'),
        gtv_std_dev=('gtv', 'std')
    ).reset_index()
    gtv_stats['gtv_std_dev'].fillna(0, inplace=True)
    user_asset_type = pd.get_dummies(trx_data[['user_id', 'asset_type']].drop_duplicates(), columns=['asset_type'], prefix='asset_type')
    user_asset_type = user_asset_type.groupby('user_id').sum().reset_index()

    final_trx_data = pd.merge(first_transaction, last_transaction, on='user_id')
    final_trx_data['recency_of_transaction'] = recency_of_transaction
    final_trx_data = pd.merge(final_trx_data, gtv_stats, on='user_id')
    final_trx_data = pd.merge(final_trx_data, user_asset_type, on='user_id')

    # Save processed transaction data to CSV
    processed_folder = '../processed'
    os.makedirs(processed_folder, exist_ok=True)
    final_trx_data.to_csv(os.path.join(processed_folder, 'final_trx_data.csv'), index=False)

    return final_trx_data

def preprocess_data():
    """Preprocess training data."""
    trx_data, profile, train_label = load_data()
    final_trx_data = process_transaction_data(trx_data)

    merged_data = pd.merge(train_label, profile, on='user_id')
    merged_data = pd.merge(merged_data, final_trx_data, on='user_id')
    mode_value = merged_data['gender_name'].mode()[0]
    merged_data['gender_name'].fillna(mode_value, inplace=True)
    encoder = LabelEncoder()
    merged_data['gender_name'] = encoder.fit_transform(merged_data['gender_name'])
    merged_data = merged_data.drop(['first_transaction','last_transaction',
                                    'marital_status','education_background',
                                    'income_level','occupation',
                                    'mobile_marketing_name','mobile_brand_name'],axis = 1)

    # Save processed training data to CSV
    processed_folder = '../processed'
    os.makedirs(processed_folder, exist_ok=True)
    merged_data.to_csv(os.path.join(processed_folder, 'merged_data.csv'), index=False)

    return merged_data

def preprocess_test_data():
    """Preprocess test data."""
    trx_data, profile, train_label = load_data()
    final_trx_data = process_transaction_data(trx_data)

    test_data = pd.merge(profile,final_trx_data, on='user_id')
    test_user_ids = test_data['user_id']
    train_user_ids = train_label['user_id']

    #Find the user_ids in test_data that are not in train_label
    user_ids_not_in_train = test_user_ids[~test_user_ids.isin(train_user_ids)]

    # Filter the test_data to include only these user_ids
    test_data = test_data[test_data['user_id'].isin(user_ids_not_in_train)]

    # Fill NaN in 'gender_name' with the most frequent value (mode).
    mode_value = test_data['gender_name'].mode()[0]
    test_data['gender_name'].fillna(mode_value, inplace=True)
    encoder = LabelEncoder()
    test_data['gender_name'] = encoder.fit_transform(test_data['gender_name'])

    #Drop the features to match the train dataset for prediction using the trained model
    test_data = test_data.drop(['first_transaction','last_transaction',
                                'marital_status','education_background',
                                'income_level','occupation',
                                'mobile_marketing_name','mobile_brand_name'],axis = 1)


    # Save processed training data to CSV
    processed_folder = '../processed'
    os.makedirs(processed_folder, exist_ok=True)
    # Save the test_data to csv
    test_data.to_csv(os.path.join(processed_folder, 'test_data.csv'), index=False)

    return test_data

if __name__ == "__main__":
    trx_data, profile, train_label = load_data()
    process_transaction_data(trx_data)
    preprocess_data()
    preprocess_test_data()