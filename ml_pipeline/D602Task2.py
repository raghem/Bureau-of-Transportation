#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

file_path = 'T_ONTIME_REPORTING.csv'

df = pd.read_csv(file_path)

print(df)


# In[2]:


df.head()


# In[3]:


#Renaming the columns

df.rename(columns = {
    'YEAR': 'Year',
    'MONTH': 'Month',
    'DAY_OF_MONTH': 'Day',
    'DAY_OF_WEEK': 'DayofWeek',
    'ORIGIN_AIRPORT_ID': 'OriginAirportID',
    'DEST_AIRPORT_ID': 'DestAirportID',
    'DEST_AIRPORT_SEQ_ID': 'DestAirportSeqID',
    'CRS_DEP_TIME': 'CRSDepTime',
    'DEP_TIME': 'DepTime',
    'DEP_DELAY_NEW': 'DepDelayMinutes',
    'DEP_DELAY': 'DepDelay',
    'CRS_ARR_TIME': 'CRSArrTime',
    'ARR_TIME': 'ArrTime',
    'ARR_DELAY': 'ArrDelay',
    'ARR_DELAY_NEW': 'ArrDelayMinutes'
}, inplace=True)


# In[4]:


# Specify columns to keep
columns_needed = [
    'Year', 'Month', 'Day', 'DayofWeek', 'OriginAirportID',
    'DestAirportID', 'DestAirportSeqID',
    'CRSDepTime', 'DepTime', 'DepDelay', 'DepDelayMinutes', 
    'CRSArrTime', 'ArrTime', 'ArrDelay', 'ArrDelayMinutes'
]

# Filter the dataframe to keep only the specified columns
filtered_df = df[columns_needed]

filtered_df


# In[5]:


# Selecting the relevant columns and create a column
filtered_df = filtered_df[[    'Year', 'Month', 'Day', 'DayofWeek', 'OriginAirportID',
    'DestAirportID', 'DestAirportSeqID',
    'CRSDepTime', 'DepTime', 'DepDelay', 'DepDelayMinutes', 
    'CRSArrTime', 'ArrTime', 'ArrDelay', 'ArrDelayMinutes'
]].copy()


print(filtered_df)


# In[6]:


filtered_df.to_csv('filtered_dataset.csv', index = False)


# In[7]:


# Filtering rows 

filtered_df = filtered_df[filtered_df['DestAirportID'] == 11298].copy()

print(filtered_df)


# In[8]:


# Check for missing values by column
missing_values = filtered_df.isnull().sum()

# Print results
print(missing_values)


# In[9]:


# Drop rows with missing values
filtered_df.dropna(subset = ['DepTime', 'DepDelay',
                             'DepDelayMinutes', 'ArrTime', 'ArrDelay', 'ArrDelayMinutes'], inplace = True)

# Check for missing values by column
missing_values = filtered_df.isnull().sum()

# Print results
print(missing_values)


# In[10]:


# Dropping spaces of each cell
filtered_df = filtered_df.map(lambda x: x.strip() if isinstance(x, str) else x)

print(filtered_df)


# In[11]:


# Saving filtered dataset

filtered_df.to_excel('filtered_dataframe.xlsx', index = False)


# In[ ]:




