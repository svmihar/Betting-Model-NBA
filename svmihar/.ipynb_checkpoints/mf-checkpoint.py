import turicreate as tc
import pdb
from sklearn.metrics import mean_absolute_error
data_name = 'data_sidefeature.csv'
train_name = 'train_sidefeature.csv'
val_name = 'val_sidefeature.csv'

competition = tc.SFrame.read_csv(data_name)
train, val = tc.recommender.util.random_split_by_user(competition, 'team_a', 'team_b')
"""
train.export_csv(train_name)
val.export_csv(val_name)
train = tc.SFrame.read_csv(train_name)
val = tc.SFrame.read_csv(val_name)

"""

if __name__ == '__main__': 
    model1 = tc.recommender.factorization_recommender.create(train, 'team_a', 'team_b', 'score')
    model2 = tc.recommender.ranking_factorization_recommender.create(train, 'team_a', 'team_b', 'score')
    tc.recommender.util.compare_models(train, [model1, model2], exclude_known_for_precision_recall=False)

    y_pred1 = model1.predict(val) 
    y_pred2 = model2.predict(val) 
    score1 = mean_absolute_error(y_pred1, val['score'])
    score2 = mean_absolute_error(y_pred2, val['score'])

    print(score1, score2)