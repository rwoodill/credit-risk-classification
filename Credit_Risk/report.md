# Credit Risk Analysis Report

## Overview of the Analysis

When lending companies lend money to borrowers, they are expecting the borrower to return or repay the lender. Credit Risk (the borrower not returning the asset or paying back the loan) can cause the lender to lose money. With this analysis we will use Machine Learning to analyze historical lending activity to build a model that can identify whether a borrower is a credit risk or not.

This analysis will use the Logistic Regression Algorithim to determine which loans are healthy (low-risk) or non-healthy (high-risk) based on loan status provided by a lending company.

Using the dataset provided, a Logistic Regression Model was created that generated an accuracy score of 95%. The model generates high-accuracy, but further analysis reveals that the model's recall value (0.91) for non-healthy loans is lower than the recall value (0.99) of healthy loans. This shows that the model will better predict the loan status as healthy. This may be due to the imbalance of the dataset, i.e. there are more healthy loans documented than non-healthy loans.

Taking a look at the value counts of the data reveals the imbalance of the data. The majority of the data is about healthy loans (0) and there is less data available about non-healthy loans (1).
```
y_var.value_counts()
0    75036
1     2500
Name: loan_status, dtype: int64
```
To generate a higher accuracy score, the data is oversampled by using RandomOverSampler to create a more balanced dataset.

```
y_random_model.value_counts()
0    56271
1    56271
Name: loan_status, dtype: int64
```
Using the provided dataset, a Logistic Regression model fit with oversampled data is created that generates an accuracy score of 99% (higher than the model fitted with imbalanced data). The model's non-healthy loans recall value increased to 0.99, indicating the model will perform better on both healthy and non-healthy loans.

## Results

* Machine Learning Model 1 (The Logistic Regression model with imbalanced data):
  * Predicted healthy loans 100% of the time
  * Predicted non-healthy loans 85% of the time
  * Is more likely to:
    * mistakenly classify a healthy loan as a non-healthy loan
    * mistakenly classify a non healthy loan as a healthy loan
  * Generated an accuracy score of 95%
  * Made a mistake 1% of the time when predicting healthy loans
  * Made a mistake 9% of the time with predicting non-healthy loans


* Machine Learning Model 2 (The Logistic Regression model with balanced data):
  * Has a much lower chance of making these mistakes:
    * mistaking a healthy loan for a non-healthy loan
    * mistaking a non-healthy loan for a healthy loan 
  * Made a mistake 1% of the time when predicting healthy loans
  * Made a mistake 1% of the time when predicting non-healthy loans
  * Generated an accuracy score of 99%

## Summary


