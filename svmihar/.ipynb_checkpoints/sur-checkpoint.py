from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV
import pandas as pd
from xg import X_train, X_test, y_train,y_test
from mf import competition
import pdb

competition = competition.to_dataframe()
max_, min_ = max(competition['score']), min(competition['score'])
# pdb.set_trace()
reader = Reader(rating_scale=(min_, max_), line_format='user item rating ')
competition['date'] = pd.to_datetime(competition['date'])
data = Dataset.load_from_df(competition[['team_a', 'team_b', 'score']],reader)
param_grid = {
    "n_epochs": [5,100], 
    "lr_all": [2e-2, 5e-5], 
    "reg_all": [4e-1,4e-5], 
    "verbose": [True]
}
gs =  GridSearchCV(SVD, param_grid, measures=['mae'])

gs.fit(data)

print(gs.best_score['mae'])
print(gs.best_params['mae'])