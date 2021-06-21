# Credit_Risk_Analysis
Using supervised machine learning to assess credit risk

## Overview
We are helping Jill learn different machine learning algorithms to apply to real world tasks. We are helping Jill with credit checks to see if we can create an accurate algorithm based on given data to classify applicants as being high or low risk for loans. The targeted outcome on the dataset is the loan stauts column, after preprocessing the data, we see that over 68,000 rows are high_risk and just over 340 rows are low_risk. With a major unbalanced dataset, we will have to run different resampling models along with ensembling. We will use 6 different models and determine which ones are the best to use and to check if any of these should be used.

We oversampled the data by using conducting naive random oversampling and SMOTE. The third technique was under-sampling by using the cluster centroids algorithm. The last resampling model we used was SMOTEENN, which combines both over and under-sampling.

The last two models were ensemble models to classify the data, we used a balanced random forest and an easy ensemble classifier. These two models are key because they will help in reducing bias. 
## Results
For all of our results, we have a confusion matrix, balanced accuracy score and a classification report.
- Confusion Matrix: shows the values of how many instances the machine learning algorithm was able to predict
- accuracy score: calculates how accurate the algorithm with the dataset
- classification report: brings everything together to display the matrix and accuracy through precision (pre), sensitivity (also called recall, shown by 'rec') and the F1 score.
  - Precision is also known as the positive predictve value (PPV), it is a measure of how realiable a positive classification is.
  - Sensitivity, for this case, would be a measure of out of how many people who were correctly classified, as being high risk, were correctly classified as high risk (the same for low)
  - F1 score is the also known as the harmonic mean, it is a summary statistic of precision and sensitivity. 
### Oversampling
#### Naive Random Oversampling
- Random oversampling selects random instances of the minority calss and adds them to the training set until both classes are balanced. 

![credit_risk_naive_over](https://user-images.githubusercontent.com/79118630/122810465-64eec480-d29d-11eb-9b53-fa5fccf57744.png)
- This yields a 64.03% accurate. For predicting high risk, we get very low results in these different columns, but we get a fairly high F1 score. When we use random over sampling, we get a much higher score for predicting low risk clients then compared to high risk clients. 

#### SMOTE
- Synthetic minority oversampling technique (SMOTE) is another oversampling model. With random oversampling, random instances were selected from the minority class and added. In SMOTE, new instances are created from the minority class. For an instance in the minority class, a number of close instances is chosen and new values are created. 
- Essentially the difference is that random oversampling uses already existing instances, SMOTE creates new instances

![credit_risk_SMOTE](https://user-images.githubusercontent.com/79118630/122810478-691ae200-d29d-11eb-8223-24f91044d0d4.png)
- We get a slighlty better accuracy score with 65.15%. Looking at sensitivity, high risk was greater when random and low risk sensitivity is greater in SMOTE.
- With SMOTE, we get a similar precision score, a better sensitivity and a higher F1 score of 0.81. This time, high risk has the same F1 score and low risk had an increase in score. Since we create new instances instead of repeating random instances, we find that we get slighlty better low risk metrics

### Undersampling
- Undersampling is what it sounds like, it is the opposite of oversampling as it reduces the instances of the majority class to match the size of the minority class. Oversampling duplicates or creates similar data to address the imbalance, undersampling uses only existing data. Undersampling should only be used when there is enough usable data, in our case we have plenty of data in both classes. 
#### Cluster Centroid
- Cluster centroid undersampling is slightly similar to SMOTE. The algorithm identifies clusters in the majority class, then generates synthetic data points called centroids that represent the clusters. This doesn't create new instances, it takes already existing clusters and centers them using a centroid, this is done to the majority class until it is the size of the minority. 

![credit_risk_under](https://user-images.githubusercontent.com/79118630/122810505-6fa95980-d29d-11eb-9c90-af2ca85995b5.png)
- Our use of undersampling gave us worse results. The accuracy score is 54.75% and we have lower sensitivity and F1 scores. The sensitivity for low risk is shockingly low 0.41. This means under 50% of clients who were low risk were accurately classified as being low risk. Less than half of the clients who were low risks for loans were not given loans, yet 68% of those high risk were accurately assessed as being high risk. 
- It is good that majority of people who are high risk are being classified as it, but if the company uses this algorithm, they would also be turning away many clients who should be getting loans.

### Combo: Over and Under sampling - SMOTEENN
- With SMOTE, the algorithm doesn't capture the overall distribution of data because the new data points can be heavily influenced by outliers. With undersampling, it involves losing data and can't be used when the dataset is too small. How we combat these issues is too combine the two with SMOTEENN. 
- SMOTEENN is SMOTE and Edited Nearestr Neighbors (ENN). The first step is to oversample the minority class by using SMOTE, then clean the resulting data with undersampling. If the two nearest neighbors of a data point belong to two different classes, then it is dropped. Thus combining SMOTE oversampling with undersampling.

![credit_risk_SMOTEENN](https://user-images.githubusercontent.com/79118630/122810525-7506a400-d29d-11eb-9672-cc71e9f6f6e8.png)
- We get a similar accuracy score as compared to SMOTE of 65.51%. The sensitivity score for assessing low risk is less than 0.6 (as it was greater than 0.6 for both oversample models) but for high risk clients is much higher. With a total F1 score of 0.71, this algorithm shows promise for analyzing the high and low risk clients.
- By combining the over and under algorithms, overall sensitivity is a little on the low side, but the overall algorithm is more accurate.

### Ensemble
- With Ensemble learning, its a process of combining multiple models, to help improve the accuracy, as well as decrease variance of the model, resulting in an increase in overall performance of the model.
#### Balanced Random Forest
- Instead of using one complex decision tree, we used a random forest algorithm which will sample the data and build several smaller and simpler decision trees. The individual trees are weak when compared to the overall dataset since each tree is made from samples of the data, but when put together, it can be a strong learner. 

![credit_risk_BRF_ensemble](https://user-images.githubusercontent.com/79118630/122810540-7a63ee80-d29d-11eb-957d-2faa6c8f972a.png)
- We get a much higher accruacy score then before, 77.27%. Our precision and F1 scores for high risk clients are still very low, as it has been throughout all the algorithms, but this one has been the highest its been with precision at 0.03 and F1 score at 0.06. These new scores help increase the overall F1 score to 0.93. 

#### Easy Ensemble - AdaBoost
- Like the random forest, boosting is a technique that takes weak models and combines them to make a strong prediction. What makes boosting unique is that each model learns from the errors of the previous model. For our easy ensemble model, we used Adaptive Boosting (AdaBoost)
- In AdaBoost, a model is trained and evaluated. After the errors are evaluated, another model is trained with errors from the previous model given extra weight. This weighting is to minimize similar errors in subsequent models, this process continues until the error rate is minimized.
 
![credit_risk_EEC_adaboost](https://user-images.githubusercontent.com/79118630/122810567-8354c000-d29d-11eb-9ff8-34018c264cdc.png)
- Our last algorithm yields the highest accuracy score with 93.17%. Our precision and F1 scores for high risk classifications also increased: precision is 0.09 and F1 is 0.16.
- Both high and low risk classification sensitivity scores were greater than 0.90, showing that this algorithm accurately classifies the clients. 
- Precision, sensitivity and F1 scores all yield very high results, along with a high accuracy score, this algorithm shows very strong predictive ability in classifying high and low risk clients. 

## Summary
- We used 6 machine learning algorithms in trying to find if there was a way to automate the process of assessing clients credit to classify their risk levels when applying for loans. With a major imbalance in the data, we needed to use different methods when training and testing the model to ensure less bias and higher accuracy. 
- We can confidently say that undersampling shoud not be used since the score returned were so low. Comparing the oversampling, they gave similar results but the SMOTE algorithm worked more accurately. If we compare the SMOTE and SMOTEENN results, they are similar and one could argue that SMOTEENN is better because it takes in undersampling. But I would say that SMOTE would be better to use because of the higher F1 score and higher overall sensitivity. 
- The two ensemble models however have much better scores and would recommend either of these rather than the resampling algorithms. Both have very high accuracy scores and have strong classification report scores. I would recommend the AdaBoosting technique, but one thing to look out for is overfitting the algorithm. Overfitting the model is avoided by splitting the data set into training and testing sets, but does not eliminate this problem. So I recommend the boosting method and to be cautius, and if being cautious is too much, then a random forest algorithm will also suffice.  
