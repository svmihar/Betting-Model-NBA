import turicreate as tc
from sklearn.metrics import mean_absolute_error
from data import train_df, test_df

train, val = tc.SFrame(train_df), tc.SFrame(test_df)


if __name__ == "__main__":
    model1 = tc.recommender.factorization_recommender.create(
        train, "team_a", "team_b", "score"
    )
    model2 = tc.recommender.ranking_factorization_recommender.create(
        train, "team_a", "team_b", "score"
    )
    tc.recommender.util.compare_models(
        train, [model1, model2], exclude_known_for_precision_recall=False
    )

    y_pred1 = model1.predict(val)
    y_pred2 = model2.predict(val)
    score1 = mean_absolute_error(y_pred1, val["score"])
    score2 = mean_absolute_error(y_pred2, val["score"])

