# Predicting Premier League Match Outcomes

Dataset: https://github.com/datasets/football-datasets/tree/main/datasets/premier-league

## 1. Overview

This project builds a machine learning model to predict Premier League match outcomes (Home Win, Draw, Away Win).  
The model is trained on historical seasons and evaluated on the most recent season (2025/26).

## 2. Dataset

The dataset contains 4,440 matches with team, match, and performance statistics.

**Target variable**  
- `FTR` — Full-Time Result (H, D, A)

**Main features**  
- Elo ratings (team strength)  
- Recent form (points last 5 matches)  
- Goals scored and conceded (recent matches)  
- Rest days before match

## 3. Exploratory Insights

- Home teams have a slight advantage.  
- Recent form is predictive of future results.  
- Away teams conceding more goals increases home win probability.  
- More rest days generally improves performance.

These findings guided feature selection.

## 4. Model

A Random Forest classifier was used due to the moderate dataset size and strong tabular performance.

**Training setup**  
- Train: first 11 seasons  
- Test: 2025/26 season  
- Initial estimators: 100

**Baseline performance**  
- Accuracy: 63%  
- Baseline accuracy (most frequent class): 45%

The model predicts home and away wins reasonably well but struggles with draws.

**Most important features**  
- Home/Away Elo ratings  
- Points in last 5 matches  
- Goals scored/conceded (last 5)  
- Rest days

## 5. Hyperparameter Tuning

`GridSearchCV` was used to optimize the Random Forest.

**Best parameters**  
- `n_estimators`: 200  
- `max_depth`: None  
- `max_features`: 0.5  
- `min_samples_split`: 10  
- `min_samples_leaf`: 4  
- `bootstrap`: True  
- `class_weight`: None

**Best CV score:** 0.668

**Test performance after tuning**  
- Accuracy: 0.64  
- Macro F1: 0.58

Hyperparameter tuning provided only a small and statistically insignificant improvement, suggesting that the model is close to its performance ceiling given the current feature set.

## 6. Future Improvements

**Feature engineering**  
- Injury data  
- Expected Goals (xG)  
- More advanced team strength metrics

**Potential applications**  
- Betting probability estimates  
- Fantasy football support  
- Broadcast analytics

## 7. Conclusion

The project demonstrates that machine learning can meaningfully predict Premier League outcomes. The baseline Random Forest model achieved 63% accuracy, outperforming a naive prediction strategy. After tuning, the model reached ~64% accuracy, which see only a marginal improvement.

This limited gain is expected because Random Forest is already a strong model for tabular data, and predictive performance in football is often constrained more by feature quality than by model complexity. With the current features, the model appears to have extracted most of the available signal.

Further improvements are therefore more likely to come from enhanced feature engineering (e.g., injuries, xG, or richer team strength metrics) rather than from additional model tuning or complexity.