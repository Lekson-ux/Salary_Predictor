
# coding: utf-8

# In[68]:


pd.__version__


# In[73]


# In[74]:


import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))


# In[75]:


import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# In[76]:


import streamlit as st
st.__version__


# In[ ]:


# matplotlib==2.1.2
# numpy==1.19.2
# pandas==0.22.0
# streamlit==1.9.0
# sklearn==0.24.2


# In[72]:


import pandas as pd
import matplotlib.pyplot as plt             
import numpy as np


# In[1]:


# reading data frame
import pandas as pd
import matplotlib.pyplot as plt             

df = pd.read_csv("survey_results_public.csv")


# In[2]:


df.shape


# In[3]:


df.head(5)


# In[5]:


df.columns


# In[4]:


df.info()


# In[5]:


df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedComp"]]
df = df.rename({"ConvertedComp": "Salary"}, axis=1)
df.head()


# In[6]:


df.info()


# In[7]:


#dropping all rows in the salary column with nan value
df = df[df["Salary"].notnull()]
df.head()


# In[8]:


df.info()


# In[9]:


#drop all rows that not numbers
df = df.dropna()
df.isnull().sum()


# In[10]:


df= df[df['Employment']== 'Employed full-time']
df= df.drop('Employment', axis=1)
df.info()


# In[11]:


df.columns


# In[12]:


df['Country'].value_counts()


# # Exploratory Data Analysis

# In[13]:


def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


# In[14]:


country_map = shorten_categories(df.Country.value_counts(), 400)
df['Country'] = df['Country'].map(country_map)
df.Country.value_counts()


# ## Outlier Dectection with Boxplot

# In[16]:


fig, ax = plt.subplots(1,1, figsize=(12, 7))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation= 90)
plt.show();


# # Outlier  dectection

# In[17]:


upper_limit = df.Salary.mean() + 3*df.Salary.std()
upper_limit


# In[19]:


lower_limit = df.Salary.mean() -3*df.Salary.std()
lower_limit


# In[20]:


# viewing outliers
df_with_outliers= df[(df.Salary>upper_limit) | (df.Salary<lower_limit)]
df_with_outliers


# In[21]:


df_with_outliers.shape


# ### Removing Outliers and generating new data frame

# In[22]:


df_no_outlier = df[(df.Salary<upper_limit) & (df.Salary>lower_limit)]
df_no_outlier.head()


# In[23]:


df_no_outlier.shape


# ## Boxplot after Outlier removal

# In[24]:


fig, ax = plt.subplots(1,1, figsize=(12, 7))
df_no_outlier.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show();


# In[25]:


df_no_outlier.isnull().sum()


# In[26]:


df_no_outlier.shape


# In[27]:


df.shape


# In[28]:


#dropping all rows in the salary column with nan value
df_no_outlier =df_no_outlier[df_no_outlier["Salary"].notnull()]
df_no_outlier.head()


# In[30]:


df_no_outlier.shape


# In[31]:


df_no_outlier['Country'].value_counts()


# # Label Encoding

# In[32]:


df_no_outlier["YearsCodePro"].unique()


# In[33]:


#cleaning and transforming the YearsCodePro column 
def clean_experience(x):
    if x ==  'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

df_no_outlier['YearsCodePro'] = df_no_outlier['YearsCodePro'].apply(clean_experience)


# In[34]:


df_no_outlier["YearsCodePro"].unique()


# In[35]:


df_no_outlier["EdLevel"].unique()


# In[36]:


def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

df_no_outlier['EdLevel'] = df_no_outlier['EdLevel'].apply(clean_education)


# In[37]:


df_no_outlier["EdLevel"].unique()


# In[38]:


#Transform EdLevel column to numeric data
from sklearn.preprocessing import LabelEncoder
le_education = LabelEncoder()
df_no_outlier['EdLevel'] = le_education.fit_transform(df_no_outlier['EdLevel'])
df_no_outlier["EdLevel"].unique()
#le.classes_


# In[39]:


#Transform Country column to numeric data
le_country = LabelEncoder()
df_no_outlier['Country'] = le_country.fit_transform(df_no_outlier['Country'])
df_no_outlier["Country"].unique()


# # Data splitting

# In[40]:


df_no_outlier.info()


# In[41]:


X = df_no_outlier.drop("Salary", axis=1)
y = df_no_outlier["Salary"]


# In[42]:


#dividing data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state= 42)


# In[43]:


X_train.shape


# In[44]:


y_train.shape


# # Model Bulding

# ### Linear Regression

# In[46]:


#Creating a Linear Regression model
from sklearn.linear_model import LinearRegression
#instantiate
LR = LinearRegression()
#fit
train_Lr= LR.fit(X_train, y_train)
#predict
y_pred_lr= LR.predict(X_test)
# # get cross val scores
# get_cv_scores(train_lr)


# ### Decision Tree Regression

# In[47]:


#Decision tree regressor model
from sklearn.tree import DecisionTreeRegressor
#instantiate
Dt= DecisionTreeRegressor(random_state=42)
#fit
train_dt= Dt.fit(X_train, y_train)
#predict
y_pred_dt= Dt.predict(X_test)


# ### Random Forest Regression

# In[48]:


#Random forest tree model
from sklearn.ensemble import RandomForestRegressor
#instantiate
Rf= RandomForestRegressor(random_state=42)
#fit
train_rf= Rf.fit(X_train, y_train)
#predict
y_pred_rf= Rf.predict(X_test)


# # Model Evaluation

# In[49]:


# Evaluating Linear Regressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
error1 = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print('${:,.2f}'.format(error1))


# In[50]:


from sklearn.metrics import r2_score
r2 = round(r2_score(y_test, y_pred_lr), 2)
print('r2 score for perfect model is', r2)


# In[51]:


#Evaluating DecisonTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
error2= np.sqrt(mean_squared_error(y_test, y_pred_dt))
print('${:,.2f}'.format(error2))


# In[52]:


from sklearn.metrics import r2_score
r2 = round(r2_score(y_test, y_pred_dt), 2)
print('r2 score for perfect model is', r2)


# In[53]:


# Evaluating RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
error3= np.sqrt(mean_squared_error(y_test, y_pred_rf))
print('${:,.2f}'.format(error3))


# In[54]:


from sklearn.metrics import r2_score
r2 = round(r2_score(y_test, y_pred_rf), 2)
print('r2 score for perfect model is', r2)


# ### Hence the best ML Algorithm is RANDOM FOREST REGRESSOR. Since the lower the error, the better the model

# # Parameters and Hyperparameters Tuning

# In[55]:


# Available hyperparameters in 
from pprint import pprint
print('Parameters currently in use:\n')
pprint(Rf.get_params())


# In[56]:


from sklearn.model_selection import GridSearchCV

max_depth = [None, 2,4,6,8,10,12]
parameters = {"max_depth": max_depth}

model = RandomForestRegressor(random_state=0)
gs = GridSearchCV(model, parameters, scoring='neg_mean_squared_error')
gs.fit(X_train, y_train)


# In[57]:


model = gs.best_estimator_

model.fit(X_train, y_train)
f_pred = model.predict(X_test)
f_error = np.sqrt(mean_squared_error(y_test, f_pred))
print("${:,.2f}".format(f_error))


# # Predictive System

# In[59]:


X.head()


# In[60]:


# country, edlevel, yearscode
X = np.array([["United States", 'Master’s degree', 2 ]])
X


# In[61]:


X[:, 0] = le_country.transform(X[:,0])
X[:, 1] = le_education.transform(X[:,1])
X = X.astype(float)
X


# In[62]:


y_pred= model.predict(X)
y_pred


# # Saving model

# In[63]:


import pickle


# In[64]:


data = {"model": model, "le_country": le_country, "le_education": le_education}
with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)


# ### using saved model

# In[65]:


with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)

regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


# In[66]:


y_pred = regressor_loaded.predict(X)
y_pred


# In[ ]:




