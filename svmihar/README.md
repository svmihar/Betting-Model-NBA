
# This is a work in progress

current best MAE is 14.75 with RandomForestRegression

Sports betting predictive modelling project

By: PerryGraham & alden-august-wang

The initial purpose of this project is to predict the Over-Under of an NBA game.

# Data collection:
Data downloaded from Kaggle from Nathan Lauga.
# Data cleaning:
* Added a point total column
* Exculded data before 2018 season
* Removed null values
* Sorted by date
* Combined data from team ranking history
* Encoded team IDs and Season
# Data insight & visuals:
* Opened cleaned data file with Tableau to explore the dataset
    + Median total points = 221
    + Upper hinge = 235, lower hinge = 168
# Feature engineering:
* At first I only used past 2 years of data, because I thought since teams change a lot it wouldnt makes sense to take longer periods of data, of teams that dont exist anymore. However, adding the last 10 years of data helped a lot in getting better results
* Then I tried using less features (only ones with high correlation with the target, this actually got worse results across all the model types
* I tried manually scaling all of the data to see if that would help get better result in any of the models
* The random forest resgression model currently yields the most accurate results out of all the regression models that I have tried
# Model fitting:

# Cross valiation:

# Results:


# svmihar's approach
## problem statement
given 2 distinct teams, what is the score?
f(x,y) = P(x|y)

or, to oversimplify, this is basically a recommendation engine, like **the movies / books dataset.**

## data
columns: `team_a, team_b, date, score`

## approach
[matrix factorization](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))
using apple's turicreate
    - using the [RankingFactorizationRecommender](https://apple.github.io/turicreate/docs/api/generated/turicreate.recommender.ranking_factorization_recommender.RankingFactorizationRecommender.html)
    - can be improved by normalizing points
    - is time relevant? is a certain time affects team performance?
    - **team_a vs team_b is different than team_b vs team_a!!**
    
[xgbregressor]()
using xgboost
[lightgbm]()
using lightgbm
[surprise](https://realpython.com/build-recommendation-engine-collaborative-filtering/)
- limited to [team_a, team_b, points, date]

[fastai collab]()
    [tabulardata]()
    
## result
(data, data_sidefeature).csv
- rankingfactorization: 16.832728665013466, 19.19.449039216423152

```python
Class                            : FactorizationRecommender

Schema
------
User ID                          : team_a
Item ID                          : team_b
Target                           : score
Additional observation features  : 2
User side features               : []
Item side features               : []

Statistics
----------
Number of observations           : 9537
Number of users                  : 30
Number of items                  : 30

Training summary
----------------
Training time                    : 2.5

Model Parameters
----------------
Model class                      : FactorizationRecommender
num_factors                      : 8
binary_target                    : 0
side_data_factorization          : 1
solver                           : auto
nmf                              : 0
max_iterations                   : 50

Regularization Settings
-----------------------
regularization                   : 0.0
regularization_type              : normal
linear_regularization            : 0.0

Optimization Settings
---------------------
init_random_sigma                : 0.01
sgd_convergence_interval         : 4
sgd_convergence_threshold        : 0.0
sgd_max_trial_iterations         : 5
sgd_sampling_block_size          : 131072
sgd_step_adjustment_interval     : 4
sgd_step_size                    : 0.0
sgd_trial_sample_minimum_size    : 10000
sgd_trial_sample_proportion      : 0.125
step_size_decrease_rate          : 0.75
additional_iterations_if_unhealthy : 5
adagrad_momentum_weighting       : 0.9
num_tempering_iterations         : 4
tempering_regularization_start_value : 0.0
track_exact_loss                 : 0

```
- xgboost: (Best is trial 107 with value: 17.248528174936922
```bash 
FrozenTrial(number=107, value=17.248528174936922, datetime_start=datetime.datetime(2020, 8, 17, 19, 18, 28, 396400), datetime_complete=datetime.datetime(2020, 8, 17, 19, 18, 28, 435759), params={'booster': 'gbtree', 'lambda': 2.5981140408909322e-08, 'alpha': 6.224825617319637e-07, 'max_depth': 9, 'eta': 0.38956787965984907, 'gamma': 0.02870411021482083, 'grow_policy': 'depthwise'}, distributions={'booster': CategoricalDistribution(choices=('gbtree', 'gblinear', 'dart')), 'lambda': LogUniformDistribution(high=10.0, low=1e-08), 'alpha': LogUniformDistribution(high=10.0, low=1e-08), 'max_depth': IntUniformDistribution(high=9, low=1, step=1), 'eta': LogUniformDistribution(high=1.0, low=1e-08), 'gamma': LogUniformDistribution(high=1.0, low=1e-08), 'grow_policy': CategoricalDistribution(choices=('depthwise', 'lossguide'))}, user_attrs={}, system_attrs={}, intermediate_values={}, trial_id=107, state=TrialState.COMPLETE)
```
    - using sidefeature1: 16.....
    - using sidefeature1,2: 
        

## future improvements
- add more features
    - win streak
    - player stats (with a player scoring function)
    - ??
