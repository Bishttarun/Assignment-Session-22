
# coding: utf-8

# Problem Statement
# 
# I decided to treat this as a classification problem by creating a new binary variable affair
# (did the woman have at least one affair?) and trying to predict the classification for each
# woman.
# Dataset
# The dataset I chose is the affairs dataset that comes with Statsmodels. It was derived
# from a survey of women in 1974 by Redbook magazine, in which married women were
# asked about their participation in extramarital affairs. More information about the study
# is available in a 1978 paper from the Journal of Political Economy.
# Description of Variables
# The dataset contains 6366 observations of 9 variables:
# rate_marriage: woman's rating of her marriage (1 = very poor, 5 = very good)
# age: woman's age
# yrs_married: number of years married
# children: number of children
# religious: woman's rating of how religious she is (1 = not religious, 4 = strongly religious)
# educ: level of education (9 = grade school, 12 = high school, 14 = some college, 16 =
# college graduate, 17 = some graduate school, 20 = advanced degree)
# 
# occupation: woman's occupation (1 = student, 2 = farming/semi-skilled/unskilled, 3 =
# "white collar", 4 = teacher/nurse/writer/technician/skilled, 5 = managerial/business, 6 =
# professional with advanced degree)
# occupation_husb: husband's occupation (same coding as above)
# affairs: time spent in extra-marital affairs
# Code to loading data and modules
# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# from patsy import dmatrices
# from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import train_test_split
# from sklearn import metrics
# from sklearn.cross_validation import cross_val_score
# dta = sm.datasets.fair.load_pandas().data
# 
# # add "affair" column: 1 represents having affairs, 0 represents not
# dta['affair'] = (dta.affairs > 0).astype(int)
# y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + \
# religious + educ + C(occupation) + C(occupation_husb)',
# dta, return_type="dataframe")
# 
# X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
# 'C(occupation)[T.3.0]':'occ_3',
# 'C(occupation)[T.4.0]':'occ_4',
# 'C(occupation)[T.5.0]':'occ_5',
# 'C(occupation)[T.6.0]':'occ_6',
# 'C(occupation_husb)[T.2.0]':'occ_husb_2',
# 'C(occupation_husb)[T.3.0]':'occ_husb_3',
# 'C(occupation_husb)[T.4.0]':'occ_husb_4',
# 'C(occupation_husb)[T.5.0]':'occ_husb_5',
# 'C(occupation_husb)[T.6.0]':'occ_husb_6'})
# y = np.ravel(y)
# 
# NOTE:​ ​The​ ​solution​ ​shared​ ​through​ ​Github​ ​should​ ​contain​ ​the​ ​source​ ​code​ ​used​ ​
# and​ ​the​ ​screenshot​ ​of​ ​the​ ​output.

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


# Load dataset

# In[2]:


dta = sm.datasets.fair.load_pandas().data


# # add "affair" column: 1 represents having affairs, 0 represents not.astype
# dta['affair'] = (dta.affair>0).astype.int
# 

# In[4]:


dta.groupby('affair').mean()


# data reveals that on average women who have affairs rate their marrige low

# In[5]:


dta.groupby('rate_marriage').mean()


# An increase in age, yrs_married, and children appears to correlate with a declining marriage rating.

# Data visualization

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


# histogram of education
dta.educ.hist()
plt.title('Histogram of Education')
plt.xlabel('Education Level')
plt.ylabel('Frequency')


# In[9]:


# hitogram of marriage rating
dta.rate_marriage.hist()
plt.title('Histogram of Marraige Rating')
plt.xlabel('Marraige rating')
plt.ylabel('Frequency')


# Let's take a look at the distribution of marriage ratings for those having affairs versus those not having affairs.

# In[10]:


# barplot of marriage rating grouped by affair (True or False)
pd.crosstab(dta.rate_marriage, dta.affair.astype(bool)).plot(kind='bar')
plt.title('Marriage Rating Distribution by Affair Status')
plt.xlabel('Marriage Rating')
plt.ylabel('Frequency')


# Let's use a stacked barplot to look at the percentage of women having affairs by number of years of marriage.

# In[11]:


affair_yrs_married = pd.crosstab(dta.yrs_married, dta.affair.astype(bool))
affair_yrs_married.div(affair_yrs_married.sum(1).astype(float), axis=0).plot(kind='bar'
, stacked=True)
plt.title('Affair Percentage by Years Married')
plt.xlabel('Years Married')
plt.ylabel('Percentage')


# Prepare Data for Logistic Regression

# In[12]:


# create dataframes with an intercept column and dummy variables for # occupation and occupation_husb
y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + religious + educ + C(occupation) + C(occupation_husb)',
dta, return_type="dataframe")
X.columns


# Renaming columns for dummy variables

# In[13]:


# fix column names of X
X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
'C(occupation)[T.3.0]':'occ_3',
'C(occupation)[T.4.0]':'occ_4',
'C(occupation)[T.5.0]':'occ_5',
'C(occupation)[T.6.0]':'occ_6',
'C(occupation_husb)[T.2.0]':'occ_husb_2',
'C(occupation_husb)[T.3.0]':'occ_husb_3',
'C(occupation_husb)[T.4.0]':'occ_husb_4',
'C(occupation_husb)[T.5.0]':'occ_husb_5',
'C(occupation_husb)[T.6.0]':'occ_husb_6'})


# In[14]:


# flatten y into a 1-D array
y = np.ravel(y)


# Logistic Regression

# In[15]:


# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X, y)
# check the accuracy on the training set
model.score(X, y)


# 73% accuracy seems good, but what's the null error rate?

# In[16]:


# what percentage had affairs?
y.mean()


# Only 32% of the women had affairs, which means that you could obtain 68% accuracy by always predicting "no". So we're doing better than the null error rate, but not by much.

# Examining coefficients

# In[17]:


# examine the coefficients
X.columns, np.transpose(model.coef_)


# Increases in marriage rating and religiousness correspond to a decrease in the likelihood of having an affair.For both the wife's occupation and the husband's occupation, the lowest likelihood of having an affair corresponds to the baseline occupation (student), since all of the dummy coefficients are positive.

# Model Evaluation Using a Validation Set

# In[ ]:


# Split the data into training and testing test


# In[18]:


# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)


# to predict class labels for the test set. We will also generate the class probabilities, just to take a look.

# In[19]:


# predict class labels for the test set
predicted = model2.predict(X_test)
predicted


# In[20]:


# generate class probabilities
probs = model2.predict_proba(X_test)
probs


# the classifier is predicting a 1 (having an affair) any time the probability in the second column is greater than 0.5.

# # generate evaluation metrics
# print(metrics.accuracy_score(y_test, predicted))
# print(metrics.roc_auc_score(y_test, probs[:, 1]))

# In[ ]:


The accuracy is 73%, which is the same as we experienced when training and predicting on the same data.
We can also see the confusion matrix and a classification report with other metrics.


# In[22]:


print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))


# Model Evaluation Using Cross-Validation

# In[24]:


# evaluate the model using 10-fold cross-validation
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
scores, scores.mean()


# Predicting the Probability of an Affair

# let's predict the probability of an affair for a random woman not present in the dataset. She's a 25-year-old teacher who graduated college, has been married for 3 years, has 1 child, rates herself as strongly religious, rates her marriage as fair, and her husband is a farmer.

# In[25]:


model.predict_proba(np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 3, 25, 3, 1, 4,
16]]))


# The predicted probability of an affair is 23%.
