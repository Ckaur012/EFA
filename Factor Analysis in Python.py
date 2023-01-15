#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Factor analysis is a linear statistical model. It is used to explain the variance among the observed variable and condense a set of the observed variable into the unobserved variable called factors. Observed variables are modeled as a linear combination of factors and error terms (Source). Factor or latent variable is associated with multiple observed variables, who have common patterns of responses. Each factor explains a particular amount of variance in the observed variables. It helps in data interpretations by reducing the number of variables.
# 
# Factor analysis is a method for investigating whether a number of variables of interest X1, X2,……., Xl, are linearly related to a smaller number of unobservable factors F1, F2,..……, Fk.
# 
# 

# # Types of Factor Analysis

# Exploratory Factor Analysis: It is the most popular factor analysis approach among social and management researchers. Its basic assumption is that any observed variable is directly associated with any factor.
# 
# Confirmatory Factor Analysis (CFA): Its basic assumption is that each factor is associated with a particular set of observed variables. CFA confirms what is expected on the basic.

# # How does Factor Analysis Work?
# The primary objective of factor analysis is to reduce the number of observed variables and find unobservable variables. These unobserved variables help the market researcher to conclude the survey. This conversion of the observed variables to unobserved variables can be achieved in two steps:
# 
# Factor Extraction: In this step, the number of factors and approach for extraction selected using variance partitioning methods such as principal components analysis and common factor analysis.
# Factor Rotation: In this step, rotation tries to convert factors into uncorrelated factors — the main goal of this step to improve the overall interpretability. There are lots of rotation methods that are available such as: Varimax rotation method, Quartimax rotation method, and Promax rotation method.
# 

# # Choosing the Number of Factors
# Kaiser criterion is an analytical approach, which is based on the more significant proportion of variance explained by factor will be selected. The eigenvalue is a good criterion for determining the number of factors. Generally, an eigenvalue greater than 1 will be considered as selection criteria for the feature.
# 
# The graphical approach is based on the visual representation of factors' eigenvalues also called scree plot. This scree plot helps us to determine the number of factors where the curve makes an elbow.
# 
# 

# In[2]:


#!pip install factor_analyzer


# In[3]:


# Import required libraries
import pandas as pd
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt


# In[4]:


df= pd.read_csv("/Users/charanjeetkaur/Downloads/bfi.csv")


# In[5]:


df.columns


# In[6]:


# Dropping unnecessary columns
df.drop(['gender', 'education', 'age'],axis=1,inplace=True)


# In[7]:


# Dropping missing values rows
df.dropna(inplace=True)


# In[8]:


df.info()


# In[9]:


df.head()


# # Adequacy Test
# Before you perform factor analysis, you need to evaluate the “factorability” of our dataset. Factorability means "can we found the factors in the dataset?". There are two methods to check the factorability or sampling adequacy:
# 
# Bartlett’s Test
# Kaiser-Meyer-Olkin Test
# Bartlett’s test of sphericity checks whether or not the observed variables intercorrelate at all using the observed correlation matrix against the identity matrix. If the test found statistically insignificant, you should not employ a factor analysis.

# In[10]:


from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(df)
chi_square_value, p_value


# In this Bartlett ’s test, the p-value is 0. The test was statistically significant, indicating that the observed correlation matrix is not an identity matrix.
# 
# Kaiser-Meyer-Olkin (KMO) Test measures the suitability of data for factor analysis. It determines the adequacy for each observed variable and for the complete model. KMO estimates the proportion of variance among all the observed variable. Lower proportion id more suitable for factor analysis. KMO values range between 0 and 1. Value of KMO less than 0.6 is considered inadequate.

# In[11]:


from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(df)


# In[12]:


kmo_model


# The overall KMO for our data is 0.84, which is excellent. This value indicates that you can proceed with your planned factor analysis.
# 
# Choosing the Number of Factors
# For choosing the number of factors, you can use the Kaiser criterion and scree plot. Both are based on eigenvalues.

# In[17]:


# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer()
fa.fit(df)
eigen_values, vectors = fa.get_eigenvalues()


# In[18]:


eigen_values


# Here, you can see only for 6-factors eigenvalues are greater than one. It means we need to choose only 6 factors (or unobserved variables).

# In[33]:


# Create scree plot using matplotlib
plt.scatter(range(1,df.shape[1]+1),eigen_values)
plt.plot(range(1,df.shape[1]+1),eigen_values)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()


# The scree plot method draws a straight line for each factor and its eigenvalues. Number eigenvalues greater than one considered as the number of factors.
# 
# Here, you can see only for 6-factors eigenvalues are greater than one. It means we need to choose only 6 factors (or unobserved variables).

# # Performing Factor Analysis

# In[42]:


#Create factor analysis object and perform factor analysis
fa = FactorAnalyzer()
fa.set_params(n_factors=6, rotation='varimax')
fa.fit(df)
loadings = fa.loadings_


# In[43]:


fa.loadings_


# In[45]:


## formatting as data frame to see the factors in better way
pd.DataFrame(loadings, columns=['Factor 1', 'Factor 2','Factor3','Factor4','Factor5','Factor6'],index=df.columns)


# Factor 1 has high factor loadings for E1,E2,E3,E4, and E5 (Extraversion)
# 
# Factor 2 has high factor loadings for N1,N2,N3,N4, and N5 (Neuroticism)
# 
# Factor 3 has high factor loadings for C1,C2,C3,C4, and C5 (Conscientiousness)
# 
# Factor 4 has high factor loadings for O1,O2,O3,O4, and O5 (Opennness)
# 
# Factor 5 has high factor loadings for A1,A2,A3,A4, and A5 (Agreeableness)
# 
# Factor 6 has none of the high loagings for any variable and is not easily interpretable. Its good if we take only five factors.
# 

# In[46]:


# Get variance of each factors
Var = fa.get_factor_variance()


# In[47]:


Var


# In[49]:


pd.DataFrame(Var, columns=['Factor 1', 'Factor 2','Factor3','Factor4','Factor5','Factor6'],index=['SS Loadings',
                                                                                                  'Proportion Var',
                                                                                                 'Cumulative Var'])


# Total 43% cumulative Variance explained by the 6 factors.

# In[ ]:




