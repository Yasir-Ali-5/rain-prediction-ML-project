import pandas as pd

def clean_data(df):
    # Clean Date Column
    df['Date'] = pd.to_datetime(df['Date'])
    df['date'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year

    #Drop Extra Columns 
    df.drop(["Date","Evaporation","Sunshine","WindGustDir","WindDir9am","WindDir3pm"],axis=1,inplace=True)

    # fill missing Values
    df['MinTemp'] = df['MinTemp'].fillna(float(int(df['MinTemp'].mean())))
    df['MaxTemp'] = df['MaxTemp'].fillna(float(int(df['MaxTemp'].mean())))
    df['Rainfall'] = df['Rainfall'].fillna(float(int(df['Rainfall'].mean())))
    df['Cloud9am'] = df['Cloud9am'].fillna(float(int(df['Cloud9am'].mean())))
    df['Cloud3pm'] = df['Cloud3pm'].fillna(float(int(df['Cloud3pm'].mean())))
    df['WindGustSpeed'] = df['WindGustSpeed'].fillna(float(int(df['WindGustSpeed'].mean())))
    df['Pressure9am'] = df['Pressure9am'].fillna(float(int(df['Pressure9am'].mean())))
    df['Pressure3pm'] = df['Pressure3pm'].fillna(float(int(df['Pressure3pm'].mean())))
    df['Humidity9am'] = df['Humidity9am'].fillna(float(int(df['Humidity9am'].mean())))
    df['Humidity3pm'] = df['Humidity3pm'].fillna(float(int(df['Humidity3pm'].mean())))

    # Drop Extra Nan rows
    df = df.dropna()

    # Return clean data
    return df