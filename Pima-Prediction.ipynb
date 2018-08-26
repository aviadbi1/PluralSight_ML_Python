
# coding: utf-8

# # Pima Indian Diabetes Prediction

# Import some basic libraries.
# * Pandas - provided data frames
# * matplotlib.pyplot - plotting support
# 
# Use Magic %matplotlib to display graphics inline instead of in a popup window.
# 

# In[1]:


import pandas as pd                 # pandas is a dataframe library
import matplotlib.pyplot as plt      # matplotlib.pyplot plots data

#get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading and Reviewing the Data

# In[2]:


df = pd.read_csv("./data/pima-data.csv")


# In[3]:


df.shape


# In[4]:


df.head(5)


# In[5]:


df.tail(5)


# ### Definition of features
# From the metadata on the data source we have the following definition of the features.
# 
# | Feature  | Description | Comments |
# |--------------|-------------|--------|
# | num_preg     | number of pregnancies         |
# | glucose_conc | Plasma glucose concentration a 2 hours in an oral glucose tolerance test         |
# | diastolic_bp | Diastolic blood pressure (mm Hg) |
# | thickness | Triceps skin fold thickness (mm) |
# |insulin | 2-Hour serum insulin (mu U/ml) |
# | bmi |  Body mass index (weight in kg/(height in m)^2) |
# | diab_pred |  Diabetes pedigree function |
# | Age (years) | Age (years)|
# | skin | ???? | What is this? |
# | diabetes | Class variable (1=True, 0=False) |  Why is our data boolean (True/False)? |
# 

# ## Check for null values

# In[19]:


df.isnull().values.any()


# ### Correlated Feature Check

# Helper function that displays correlation by color.  Red is most correlated, Blue least.

# In[6]:


def plot_corr(df, size=11):
    """
    Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot

    Displays:
        matrix of correlation between columns.  Blue-cyan-yellow-red-darkred => less to more correlated
                                                0 ------------------>  1
                                                Expect a darkred line running from top left to bottom right
    """

    corr = df.corr()    # data frame correlation function
    _, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)   # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks


# In[7]:


plot_corr(df)


# In[8]:


df.corr()


# In[9]:


df.head(5)


# The skin and thickness columns are correlated 1 to 1.  Dropping the skin column

# In[10]:


del df['skin']


# In[11]:


df.head(5)


# Check for additional correlations

# In[12]:


plot_corr(df)


# The correlations look good.  There appear to be no coorelated columns.

# ## Mold Data

# ### Data Types
# 
# Inspect data types to see if there are any issues.  Data should be numeric.

# In[13]:


df.head(5)


# Change diabetes from boolean to integer, True=1, False=0

# In[14]:


diabetes_map = {True : 1, False : 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)


# Verify that the diabetes data type has been changed.

#      
#      
#      
#      
#      
#      
#      
#      
#      
#      
#      
#      
#      
#      
#      
#      

# In[15]:


df.head(5)


# ### Check for null values

# In[16]:


df.isnull().values.any()


# No obvious null values.

# ### Check class distribution 
# 
# Rare events are hard to predict

# In[17]:


num_obs = len(df)
num_true = len(df.loc[df['diabetes'] == 1])
num_false = len(df.loc[df['diabetes'] == 0])
print("Number of True cases:  {0} ({1:2.2f}%)".format(num_true, (num_true/num_obs) * 100))
print("Number of False cases: {0} ({1:2.2f}%)".format(num_false, (num_false/num_obs) * 100))


# Good distribution of true and false cases.  No special work needed.

# ## Save pre-processed dataframe for later use

# In[18]:


df.to_pickle("./data/pima-data-processed.p")

