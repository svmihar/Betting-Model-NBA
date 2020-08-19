import pandas as pd
from sklearn.model_selection import train_test_split

data_name = "data_sidefeature.csv"

data_df = pd.read_csv(data_name)
del data_df['date']
train_df, test_df = train_test_split(
    data_df, test_size=0.1, random_state=69, shuffle=False
)


target_columns = "score"
feature_columns = list(train_df.columns)
feature_columns.remove(target_columns)

X_train, y_train = train_df[feature_columns], train_df[target_columns]
X_test, y_test = test_df[feature_columns], test_df[target_columns]
