from sklearn.preprocessing import LabelEncoder

def encode_data(df,column_name):
    encoder = LabelEncoder()
    df[column_name] = encoder.fit_transform(df[column_name])
    