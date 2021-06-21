# Credit_Risk_Analysis
Using supervised machine learning to assess credit risk

## Overview
We are helping Jill learn different machine learning algorithms to apply to real world tasks. We are helping Jill with credit checks to see if we can create an accurate algorithm based on given data to classify applicants as being high or low risk for loans. The targeted outcome on the dataset is the loan stauts column, after preprocessing the data, we see that over 68,000 rows are high_risk and just over 340 rows are low_risk. With a major unbalanced dataset, we will have to run different resampling models along with ensembling. We will use 6 different models and determine which ones are the best to use and to check if any of these should be used.

We oversampled the data by using conducting naive random oversampling and SMOTE. The third technique was under-sampling by using the cluster centroids algorithm. The last resampling model we used was SMOTEENN, which combines both over and under-sampling.

The last two models were ensemble models to classify the data, we used a balanced random forest and an easy ensemble classifier. These two models are key because they will help in reducing bias. 
## Results

### Oversampling
#### Naive Random Oversampling
![credit_risk_naive_over](https://user-images.githubusercontent.com/79118630/122810465-64eec480-d29d-11eb-9b53-fa5fccf57744.png)


#### SMOTE
![credit_risk_SMOTE](https://user-images.githubusercontent.com/79118630/122810478-691ae200-d29d-11eb-8223-24f91044d0d4.png)


### Undersampling
#### Cluster Centroid
![credit_risk_under](https://user-images.githubusercontent.com/79118630/122810505-6fa95980-d29d-11eb-9c90-af2ca85995b5.png)


### Combo: Over and Under sampling - SMOTEENN
![credit_risk_SMOTEENN](https://user-images.githubusercontent.com/79118630/122810525-7506a400-d29d-11eb-9672-cc71e9f6f6e8.png)



### Ensemble
#### Balanced Random Forest
![credit_risk_BRF_ensemble](https://user-images.githubusercontent.com/79118630/122810540-7a63ee80-d29d-11eb-957d-2faa6c8f972a.png)

#### Easy Ensemble 
![credit_risk_EEC_adaboost](https://user-images.githubusercontent.com/79118630/122810567-8354c000-d29d-11eb-9ff8-34018c264cdc.png)


## Summary
