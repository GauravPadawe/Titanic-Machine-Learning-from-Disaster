# Titanic-Machine-Learning-from-Disaster
Implemented a basic Logistic Regression model and deployed it on Local Machine using Flask api.

### Table of Contents

#### 1. **Information**
    - Reason for Choosing this Dataset ?**
    - Source
    - Details
    - Questionnaire
    - Objective

#### 2. **Loading Dataset**
    - Importing packages
    - Reading Data
    - Shape of data
    - Dtype

#### 3. **Data Cleansing & EDA**
    - Checking Null values
    - Descriptive Statistics
    - Univariate Analysis
    - Multivariate Analysis
    - Label Encoding
    - Null values Imputation
    - Pearson Correlation

#### 5. **Modelling**
    - Splitting Data & Choosing Algorithms
    - Logistic Regression Implementation
    - Parameter Tuning
    - ROC-AUC
    - Choosing Final model
    - Predicting on test set
    - Model deployment details

#### 6. **Conclusion**

#### 7. **References**

#### 8. **What's next ?**<br><br>


### Reason for choosng this dataset ?

- The Reason behind choosing this model is my Personal Interest to explore various Domains out there.


- Is the one of the most profound data set for beginners.


### Source :

- https://www.kaggle.com/c/titanic/data


### Details :

- The data has been split into two groups: training set (train.csv) & test set (test.csv)

    - The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.

    - The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.


- We also include gender_submission.csv, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.



- Data Dictionary
    - survival : Survival 0 = No, 1 = Yes
    - pclass : Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd 
    - sex : Sex 
    - Age : Age in years
    - sibsp : # of siblings / spouses aboard the Titanic
    - parch : # of parents / children aboard the Titanic 
    - ticket : Ticket number
    - fare : Passenger fare
    - cabin : Cabin number
    - embarked : Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton


- Variable Notes
    - pclass: A proxy for socio-economic status (SES), 1st = Upper / 2nd = Middle / 3rd = Lower
    - age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
    - sibsp: The dataset defines family relations in this way, Sibling = brother, sister, stepbrother, stepsister While Spouse = husband, wife (mistresses and fiancés were ignored)
    - parch: The dataset defines family relations in this way, Parent = mother, father while Child = daughter, son, stepdaughter, stepson. Some children travelled only with a nanny, therefore parch=0 for them.
    
    
### Questionnaire :

- Can we figure how many Males / Females Survived ?


- Can we identify survival rate w.r.t Pclass ?


- How is our Target variable distributed ? is it Imbalanced ?
    

### **Objective :**

- The goal is to explore data, perform analysis and build a predictive model on a dataset using Logistic Regression as base estimator.


- Also, deploying model locally with the help of Flask.
