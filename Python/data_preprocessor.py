#%%
import numpy as np
import pandas as pd
import os
import holidays
#%%
from sklearn.preprocessing import FunctionTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

#%%
def load_housing(test_size=.3, seed=7531):
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    df_train, df_test, y_train, y_test = train_test_split(X, y.values, test_size=test_size, random_state=seed)

    return df_train, df_test, y_train, y_test


wd_map = {1: "Montag", 2: "Dienstag", 3: "Mittwoch", 4: "Donnerstag", 5: "Freitag", 6: "Samstag", 7: "Sonntag"}
a = 1

mo_map = {
    1: "Januar",
    2: "Februar",
    3: "März",
    4: "April",
    5: "Mai",
    6: "Juni",
    7: "Juli",
    8: "August",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Dezember"
}

winter_time = [
    '2018-10-28 02:00:00', '2019-10-27 02:00:00', '2020-10-25 02:00:00', '2021-10-31 02:00:00', '2022-10-30 02:00:00',
    '2023-10-29 02:00:00'
]

#%%
def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

#%%
def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

#%%
def load_energy(path, add_vars=True):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])

    df = df[df.load > 0]
    df = df[~df.date_time.isin(winter_time)]

    if add_vars is True:
        df['hour'] = df['hour_int']
        df['weekday'] = df['weekday_int'].map(wd_map)
        df['month'] = df['month_int'].map(mo_map)
        df['hour_week_int'] = (df['weekday_int'] - 1) * 24 + df['hour_int']
        df['hour_month_int'] = (df['date'].dt.day - 1) * 24 + df['hour_int']
        df['monthday'] = df['date'].dt.day
        df['yearday'] = df['date'].dt.dayofyear
        df['year'] = df['date'].dt.year

        de_holidays = holidays.DE()
        df['holiday'] = df['date'].apply(lambda x: 1 if x in de_holidays else 0)

    return df

#%%
def prep_energy(df,
                time_trend=True,
                time_trend_sq=False,
                cat_features=False,
                fine_resolution=False,
                sin_cos_features=False,
                last_obs=False):

    df = df.copy()

    assert ~((cat_features is True) & (sin_cos_features is True)), "Can't be both categorical and sin_cos!"

    feat_str = ""
    if time_trend is True:
        feat_str += "Time Trend, "

        if time_trend_sq is True:
            feat_str += 'Time Trend squared, '

    if cat_features is True:
        feat_str += 'One-Hot Enc., '
    elif sin_cos_features is True:
        feat_str += 'Sin/Cos Enc., '
    else:
        feat_str += 'Integer Enc., '

    if fine_resolution is True:
        feat_str += 'Fine Res., '
    else:
        feat_str += 'Coarse Res., '

    if last_obs is True:
        feat_str += 'and last Obs.'
    else:
        feat_str = feat_str[:-2]

    fml = []
    if cat_features is True:
        if fine_resolution is True:
            df = pd.get_dummies(df, columns=['hour_int', 'hour_week_int', 'hour_month_int'])

            hour_list = [col for col in df.columns if 'hour_int_' in col]
            hourweek_list = [col for col in df.columns if "hour_week_" in col]
            hourmonth_list = [col for col in df.columns if "hour_month_" in col]

            hour_list.remove('hour_int_1')
            hourweek_list.remove('hour_week_int_0')
            hourmonth_list.remove('hour_month_int_0')

            # hourweek_list.remove('hour_int')

            fml = fml + hour_list + hourweek_list + hourmonth_list

        else:
            df = pd.get_dummies(df, columns=['hour_int', 'weekday', 'month'])
            # fml = fml + ['hour', 'weekday', 'month']
            hour_list = [col for col in df.columns if "hour_" in col]
            weekday_list = [col for col in df.columns if "weekday_" in col]
            # daymonth_list = [col for col in df.columns if 'monthday_' in col]
            month_list = [col for col in df.columns if "month_" in col]
            # year_list = [col for col in df.columns if 'year_' in col]
            # yearday_list = [col for col in df.columns if 'yearday_' in col]

            hour_rem_list = ['hour_stop_int', 'hour_int_0', 'hour_week_int', 'hour_month_int']
            weekday_rem_list = ['weekday_int', 'weekday_Dienstag']
            month_rem_list = ['month_int', 'month_April']

            for hr in hour_rem_list:
                if hr in hour_list:
                    hour_list.remove(hr)

            for wdr in weekday_rem_list:
                if wdr in weekday_list:
                    weekday_list.remove(wdr)

            for mr in month_rem_list:
                if mr in month_list:
                    month_list.remove(mr)

            # # hour_list.remove('hour_int')
            # hour_list.remove('hour_stop_int')
            # hour_list.remove('hour_int_0')
            # hour_list.remove('hour_week_int')
            # hour_list.remove('hour_month_int')
            # # hour_list.remove('day_year')

            # weekday_list.remove('weekday_int')
            # weekday_list.remove('weekday_Dienstag')

            # # daymonth_list.remove('monthday_1')

            # month_list.remove('month_int')
            # month_list.remove('month_April')

            # year_list.remove('year_2022')

            # yearday_list.remove('yearday_1')

            fml = fml + hour_list + weekday_list + month_list  #+ daymonth_list + year_list + yearday_list

    elif sin_cos_features is True:
        df['hour_sin'] = sin_transformer(24).fit_transform(df['hour_int'])
        df['hour_cos'] = cos_transformer(24).fit_transform(df['hour_int'])

        df['weekday_sin'] = sin_transformer(7).fit_transform(df['weekday_int'])

        df['weekday_cos'] = cos_transformer(7).fit_transform(df['weekday_int'])
        df['hour_week_sin'] = sin_transformer(24 * 7).fit_transform(df['hour_week_int'])
        df['hour_week_cos'] = cos_transformer(24 * 7).fit_transform(df['hour_week_int'])

        df['month_sin'] = sin_transformer(12).fit_transform(df['month_int'])
        df['month_cos'] = cos_transformer(12).fit_transform(df['month_int'])

        df['day_year_sin'] = sin_transformer(365).fit_transform(df['yearday'])
        df['day_year_cos'] = cos_transformer(365).fit_transform(df['yearday'])

        fml = fml + [
            'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos', 'hour_week_sin',
            'hour_week_cos', 'day_year_sin', 'day_year_cos'
        ]
    else:
        if fine_resolution is True:
            fml = fml + ['hour_int', 'hour_week_int', 'hour_month_int']
        else:
            fml = fml + ['hour_int', 'weekday_int', 'month_int', 'holiday']  #, 'yearday', 'year'] #monthday

    if time_trend is True:
        fml = fml + ['time_trend']
        if time_trend_sq is True:
            df['time_trend_sq'] = df['time_trend']**2
            fml = fml + ['time_trend_sq']

    if last_obs is True:
        df['load_t-1'] = df['load'].shift(1)
        df = df.dropna()
        fml = fml + ['load_t-1']

    # load_mean = df['load'].mean()
    # load_std = df['load'].std()

    # df['load'] = (df['load'] - load_mean) / load_std

    print(f"Total number of features: {len(fml)}")

    return df, fml, feat_str

#%%
def split_energy(df, start_t1, start_t2=None):

    df = df.set_index('date_time')
    n1 = np.where(df.index == start_t1)[0][0]

    if start_t2 is not None:
        n2 = np.where(df.index == start_t2)[0][0]
        return df.iloc[:n1], df.iloc[n1:n2], df.iloc[n2:]

    else:
        return df.iloc[:n1], df.iloc[n1:]
#%%
def select_features(data_train, data_test, feature_list):
    '''
    Extracts the needed features from the dataframe

    Parameters: 
    data = > the dataframe from which the features are extracted
    feature_list => the names of the features which should be extractes

    Returns:
    Dataframe with the extracted features
    '''
    X_train = data_train[feature_list]
    X_test = data_test[feature_list]

    return X_train, X_test

#%%
def select_target(data_train, data_test, target_list):
    '''
    Extracts the target from the training and testing dataframes

    Parameters:
    data_train (DataFrame): The training data.
    data_test (DataFrame): The testing data.
    target_list (list): The names of the target columns which should be extracted.

    Returns:
    Tuple: (y_train, y_test) - Target values for training and testing.
    '''
    y_train = data_train[target_list].squeeze()  # Ziel für Training
    y_test = data_test[target_list].squeeze()    # Ziel für Test

    return y_train, y_test
  
#%%
#def split_data(X, y, test_size=0.3, seed=7531):
    '''
    Splits the data into training and testing sets.

    Parameters:
    X (DataFrame): The feature data.
    y (DataFrame): The target data. Should be a single-column DataFrame.
    test_size (float): The proportion of the dataset to include in the test split.
    seed (int): Random seed for reproducibility.

    Returns:
    Tuple: (X_train, X_test, y_train, y_test).
    '''
    # If y is a single-column DataFrame, convert it to Series
#    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
#        y = y.squeeze()  # Convert to Series

    # Split the data
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

#    return X_train, X_test, y_train, y_test





#%%
#def split_data(X, y, test_size=.3, seed=7531):

    '''
    Splits the data into training and testing sets.

    Parameters:
    X (DataFrame): The feature data.
    y (Series): The target data.
    test_size (float): The proportion of the dataset to include in the test split.
    seed (int): Random seed for reproducibility.

    Returns:
    Tuple: (X_train, X_test, y_train, y_test).
    '''

 #   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

 #   return X_train, X_test, y_train, y_test

# %%
