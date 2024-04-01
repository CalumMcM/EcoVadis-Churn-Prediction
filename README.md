# EcoVadis-Churn-Prediction
 
## Data Analysis:
The following information was found from data analysis that was applied to the 'Assignment (churn prediction).xlsx' file

 ### Preliminary analysis:
    * Dataset size: (10,000 x 15)

    * Feature types:
        RowNumber             int64
        CustomerId            int64
        Surname              object
        CreditScore           int64
        Country              object
        Gender               object
        Age                   int64
        Tenure                int64
        CustomerFeedback     object
        Balance (EUR)       float64
        NumberOfProducts      int64
        HasCreditCard         int64
        IsActiveMember        int64
        EstimatedSalary     float64
        Exited                int64

### Data Cleaning:

* CustomerFeedback is the only column with NaN values (a total of 6982). This 
    missing data appears independent of whether or not the user has left the 
    system and so will be filled with empty speech marks to prevent NaN errors
    during model training. 

* There are no duplicated results in the customer feedback. 

### Notable Insights:

* The number of users in the database that have exited are 2,037 meaning that
  only 20.4% of all entries are for exited users, leaving us with a class
  imbalance. 

* From the group box plot it is clear that the most influential factors to
whether a user leaves the system or not is their age and their balance. Older
users are more likely to leave while those with a lower balance are more likely
to stay. 

    Graph: group_box_plot.png 

* Based on the description of numerical data in the database, the average
    customer has been with the company for 5 years and holds a balance of around
    76,000 euros with a credit score of 650. The majority of users have a credit
    card and use at least one product. 
    
* The expected salary a user has does not seem to affect whether or not they 
    are more likely to exit or not. It is only in the salary range >170,000 that 
    the ratio of those leaving trends consistently below 20%. 
   
    Graph: estimated_salary.png

* It is expected that the longer the customer is with the service the more
    they are devoted to staying and not exiting. This only seems to become
    the case at the 10 year mark with the ratio of staying vs leaving
    remaining similar from years 1-9. 
    
    Graph: tenure.png
    
* Customers are more likely to leave the server once their balance becomes >0.
        This is likely due to the fact that those without money in the account
        are "ghost" users who do not use the service (logic only works if product
        is free)
        
    Graph: binary_balance.png

* Looking at these "ghost" users, i.e., users who have a balance of 0 in their
  account, we see that they are actually more likely to leave the service if
  they are inactive than those who are. This suggests that active members with a
  balance of zero may go on to invest in the future and are currently trying to
  understand the best way to invest their money. 
        
    Graph:        zero_balance_active.png

## Model Training and Evaluation

Two models were trained on the data, a RandomForestClassifier and an XGBoost
classifier. Before training took place however, some final adjustments to the
dataset were made. 

### Pre-training:

Before data was passed to the model to train, the following fields were removed:
'RowNumber', 'CustomerId', 'Surname', and 'CustomerFeedback'. These fields were
removed as they held either personal information such as 'Surname' or they were
fields that would not help predictions as they would be unique to each user and
thus no pattern exist.

As well as removing these fields, certain fields had label encoding applied to
them. This was done to convert them from being of the String data type to an
Integer data type which the model will be able to better use as part of integer
encoding. 

Certain data fields such as 'Balance (EUR)' and 'EstimatedSalary' were
momentarily rounded to the nearest thousand to see if binning the values into
buckets would improve results, but it did not and so this pre-processing was
removed. 

The data was randomly split such that 70% made up the training data and 30% made
up the test data. A random state of 42 was used to ensure the random split would
be the same each time the code is ran. 

### Evaluation

Confusion matrices were produced for the results in order to visualise
displacement of predictions, while classification reports are also generated in
order to understand the precision, recall and f1-score of the model. The
following are the results:

Random Forest: 

              precision    recall  f1-score   support

           0       0.87      0.96      0.92      2379
           1       0.77      0.47      0.58       621

    accuracy                           0.86      3000
    macro avg      0.82      0.72      0.75      3000
    weighted avg   0.85      0.86      0.85      3000

XGBoost:

              precision    recall  f1-score   support

           0       0.89      0.95      0.92      2379
           1       0.74      0.53      0.61       621

    accuracy                           0.86      3000
    macro avg      0.81      0.74      0.77      3000
    weighted avg   0.85      0.86      0.85      3000

#### SMOTE

As the input possessed a class imbalance, SMOTE was applied in order to even out
the number of classes present in training data. This meant that where before the
class distribution was: {0 : 5584, 1 : 1416} it was now: {0 : 5584, 1 : 5584}.
SMOTE was only applied to the training sets and had the following impact on the
test set results: 

Random Forest with SMOTE:

              precision    recall  f1-score   support

           0       0.90      0.87      0.88      2379
           1       0.55      0.61      0.58       621

    accuracy                           0.81      3000
    macro avg      0.72      0.74      0.73      3000
    weighted avg   0.82      0.81      0.82      3000

XGBoost with SMOTE:

              precision    recall  f1-score   support

           0       0.90      0.87      0.88      2379
           1       0.56      0.63      0.59       621

    accuracy                           0.82      3000
    macro avg      0.73      0.75      0.74      3000
    weighted avg   0.83      0.82      0.82      3000


It is evident that the addition of SMOTE has not increased the results
significantly and in fact has reduced performance of the model on nearly all
metrics. This could be arising from the fact that SMOTE is creating noise as
part of its process in bulking the number of instances present in the smaller
dataset. This noise could be leading to a decision boundary which is further
from the real world equivalent and as a result the model struggles when only
presented with real data. 

#### Sentiment Analysis

Valuable data is held as part of the CutomerFeedback field which if it could be
converted into a form that the RandomForestClassifier or XGBoost model can
understand could potentially improve model performance. The reasoning behind
using sentiment analysis is that customers who leave happier feedback should in
theory be more likely to stay at the company than those who do not. 

This was achieved by returning to the DataLoader class and adding an extra
function which would apply sentiment analysis to a given column. VADER (Valence
Aware Dictionary and sEntiment Reasoner) was chosen as the analysis tool to
determine the sentiment of customer feedback reviews. No pre-processing is
applied to the customer feedback before it is passed into VADER as it does not
require standard NLP pre-processing techniques such as tokenization or stemming.
This lightweight solution is what has made it ideal for this use case. As the
inclusion of SMOTE only decreased the performance of the original model it was
not used in combination with the sentiment analysis. 

With sentiment analysis being applied to the CustomerFeedback column the results
are now:

Random Forest + Sentiment Analysis:

              precision    recall  f1-score   support

           0       0.87      0.96      0.92      2379
           1       0.77      0.46      0.57       621

    accuracy                           0.86      3000
    macro avg      0.82      0.71      0.74      3000
    weighted avg   0.85      0.86      0.84      3000

XGBoost + Sentiment Analysis

              precision    recall  f1-score   support

           0       0.88      0.95      0.91      2379
           1       0.72      0.52      0.60       621

    accuracy                           0.86      3000
    macro avg      0.80      0.73      0.76      3000
    weighted avg   0.85      0.86      0.85      3000