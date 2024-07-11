import pandas as pd

# now we load the data

def load_data():
    df = pd.read_csv('BTC-Daily.csv')
    start_date = '2014-07-17'
    end_date = '2020-12-29'
    new_start_date = "2020-12-30"
    return df, start_date, end_date, new_start_date

def filtered_data(df, start_date, end_date):
    filtered_df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date), ['date', 'open', 'high', 'low', 'close']]
    print(filtered_df)
    x = filtered_df.head()
    y = filtered_df.shape
    filtered_df.describe()
    filtered_df.info()
    return filtered_df, x, y

def updated_data(df, new_start_date):
    updated_df = df.loc[(df['date'] >= new_start_date), ['date', 'open', 'high', 'low', 'close']]
    print(updated_df)
    z = updated_df.head()
    w = updated_df.shape
    updated_df.describe()
    updated_df.info()
    return updated_df, z, w

def combined_old_new_data():
    df = pd.read_csv('BTC-Daily.csv')
    df['date'] = pd.to_datetime(df['date'])

# 17th July 2014 to 29th December 2022

    start_date = '2014-07-17'
    end_date = '2020-12-29'

    new_start_date = '2020-12-30'

# filtered data frame

    filtered_df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date), ['date', 'open', 'high', 'low', 'close']]
    updated_df = df.loc[(df['date'] >= new_start_date), ['date', 'open', 'high', 'low', 'close']]

    print(filtered_df)
    print(updated_df)

    x = filtered_df.head()
    y = filtered_df.shape

    z = updated_df.head()
    w = updated_df.shape

    print(x,y,z,w)

    filtered_df.describe()
    filtered_df.info()

    updated_df.describe()
    updated_df.info()

