
def split_data(df):

    train_mask = (df['year'] >= 2007) & (df['year'] <= 2012)
    test_mask = (df['year'] >= 2013)
    X_train = df[train_mask].drop(columns=['RainTomorrow'])
    y_train = df[train_mask]['RainTomorrow']
    X_test = df[test_mask].drop(columns=['RainTomorrow'])
    y_test = df[test_mask]['RainTomorrow']
    return X_train, X_test, y_train, y_test