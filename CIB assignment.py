#!/usr/bin/env python
# coding: utf-8

# ## Importing modules

# In[87]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading the dataset

# In[43]:


#extracting the dataset

iris=load_iris()
iris_df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(iris_df.info())


# In[44]:


target_df = pd.DataFrame(data= iris.target, columns= ['species'])
def converter(specie):
    if specie == 0:
        return 'setosa'
    elif specie == 1:
        return 'versicolor'
    else:
        return 'virginica'
target_df['species'] = target_df['species'].apply(converter)

# Concatenate the DataFrames
iris_df = pd.concat([iris_df, target_df], axis= 1)


# In[45]:


iris_df


# In[46]:


## An statistical overview of the dataset:

iris_df.describe()


# In[47]:


## .info() prints a concise summary of a DataFrame.

iris_df.info()


# ## Pre processing the dataset

# In[52]:


# check for null values

iris_df.isnull().sum()


# In[48]:


sns.pairplot(iris_df, hue= 'species')


# ## Exploratory Data Analysis 

# In[50]:





# ### Histogram

# In[55]:


iris_df['sepal length (cm)'].hist()


# In[56]:


iris_df['sepal width (cm)'].hist()


# In[57]:


iris_df['petal length (cm)'].hist()


# In[58]:


iris_df['petal width (cm)'].hist()


# ### scatter plot

# In[65]:


colors=['red','orange','blue']
species=['Iris-virginica','Iris-versicolor','Iris-setosa']


# In[67]:


for i in range(3):
    x=iris_df[iris_df['species']==species[i]]
    plt.scatter(x['sepal length (cm)'],x['sepal width (cm)'],c=colors[i],label=species[i])
    
plt.xlabel('Sepal Length')
plt.xlabel('Sepal Width')
plt.legend()


# ### Coorelation matrix

# In[69]:


iris_df.corr()


# In[72]:


corr=iris_df.corr()
fig, ax=plt.subplots(figsize=(5,4))
sns.heatmap(corr,annot=True,ax=ax,cmap='coolwarm')


# In[ ]:





# ## Label Encoder

# In[77]:


from sklearn.preprocessing import LabelEncoder     # Converting Objects to Numerical dtype

le=LabelEncoder()

iris_df['species']=le.fit_transform(iris_df['species'])
iris_df.head()


# ## Model Training

# In[81]:


# Variables

X= iris_df.drop(columns=['species'])
y= iris_df['species']


# In[103]:


# Splitting the Dataset 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 7)


# ## Linear Regression

# In[104]:


# Instantiating LinearRegression() Model
lr = LinearRegression()


# In[105]:


# Training/Fitting the Model
lr.fit(X_train, y_train)


# In[106]:


# Making Predictions
lr.predict(X_test)
pred = lr.predict(X_test)


# In[114]:


# Evaluating Model's Performance
print('Mean Absolute Error:', mean_absolute_error(y_test, pred))
print('Mean Squared Error:', mean_squared_error(y_test, pred))
print('Mean Root Squared Error:', np.sqrt(mean_squared_error(y_test, pred)))
print('Accuracy:',model.score(X_test,y_test)*100)


# In[115]:


iris_df.loc[6]


# In[116]:


d = {'sepal length (cm)' : [4.6],
    'sepal width (cm)' : [3.4],
    'petal length (cm)' : [1.4],
    'petal width (cm)' : [0.3],
    'species' : 0}
test_df = pd.DataFrame(data= d)
test_df


# In[117]:


pred = lr.predict(X_test)
print('Predicted Sepal Length (cm):', pred[0])
print('Actual Sepal Length (cm):', 4.6)


# In[ ]:





# ## Logistic Regression

# In[124]:


LogReg=LogisticRegression()


# In[125]:


LogReg.fit(X_train,y_train)


# ### Print metric to get the performance

# In[126]:


print('Accuracy:',LogReg.score(X_test,y_test)*100)


# In[ ]:




