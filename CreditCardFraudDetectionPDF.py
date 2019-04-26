#!/usr/bin/env python
# coding: utf-8

# # CREDIT CARD FRAUD DETECTION

# ## *IMPORTING LIBRARIES*

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ## *IMPORTING THE DATASET*

# In[5]:


creditcard_df = pd.read_csv("creditcard.csv")


# In[6]:


creditcard_df.head(15)


# In[7]:


creditcard_df.tail()


# In[8]:


creditcard_df.describe()


# ## *VISUALIZATION OF THE DATASET*

# In[9]:


fair_tran = creditcard_df[creditcard_df["Class"] == 0]


# In[10]:


fraud_tran = creditcard_df[creditcard_df["Class"] == 1]


# In[11]:


fraud_tran


# In[12]:


fair_tran


# In[13]:


print("Percentage of Fraud Transactions = ", (len(fraud_tran)/(len(fair_tran) + len(fraud_tran))) * 100, "%")


# In[14]:


print("Percentage of Fair Transactions = ", (len(fair_tran)/(len(fair_tran) + len(fraud_tran))) * 100, "%")


# In[15]:


sns.countplot(creditcard_df["Class"], label = "No. of Occurences")


# In[16]:


col_headers = creditcard_df.columns.values
i = 1
fig, ax = plt.subplots(8, 4, figsize = (18, 30))
for col_headers in col_headers:
    plt.subplot(8, 4, i)
    sns.kdeplot(fair_tran[col_headers], bw = 0.4, label = "Fair", shade = True, color = "blue", linestyle = "solid")
    sns.kdeplot(fraud_tran[col_headers], bw = 0.4, label = "Fraud", shade = True, color = "green", linestyle = "dashed")
    plt.title(col_headers, fontsize = 15)
    i += 1
plt.show


# ## *CREATING DATASET FOR TRAINING AND TESTING THE MODEL*

# In[17]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
creditcard_df["Normalized_Amount"] = sc.fit_transform(creditcard_df["Amount"].values.reshape(-1, 1))
creditcard_df = creditcard_df.drop(["Amount"], axis = 1)


# In[18]:


creditcard_df


# In[19]:


x = creditcard_df.drop(["Class"], axis = 1)
y = creditcard_df["Class"]


# In[20]:


x


# In[21]:


y


# In[22]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# ## *TRAINING THE MODEL WITH THE DATASET*

# In[23]:


from sklearn.naive_bayes import GaussianNB
NB_classifier = GaussianNB()
NB_classifier.fit(x_train, y_train)


# ## *EVALUATING THE ACCURACY OF THE MODEL*

# In[24]:


from sklearn.metrics import confusion_matrix, classification_report
y_model_train_op = NB_classifier.predict(x_train)
cMatrix = confusion_matrix(y_train, y_model_train_op)
sns.heatmap(cMatrix, annot = True)


# In[26]:


y_model_test_op = NB_classifier.predict(x_test)
cMatrix = confusion_matrix(y_test, y_model_test_op)
sns.heatmap(cMatrix, annot = True)


# In[27]:


print(classification_report(y_test, y_model_test_op))


# ## *IMPROVING THE ACCURACY OF THE MODEL*

# In[28]:


x = creditcard_df.drop(["Time", "V5", "V6", "V7", "V8", "V13", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"], axis = 1)


# In[29]:


x


# In[30]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
NB_classifier = GaussianNB()
NB_classifier.fit(x_train, y_train)
y_model_test_op = NB_classifier.predict(x_test)
cMatrix = confusion_matrix(y_test, y_model_test_op)
sns.heatmap(cMatrix, annot = True)


# In[31]:


print(classification_report(y_test, y_model_test_op))


# In[32]:


print("Actual total number of Fraud Transactions in the dataset is", sum(y_test))


# In[33]:


print("Predicted total number of Fraud Transactions by the system is", sum(y_model_test_op))

