#!/usr/bin/env python
# coding: utf-8

# <h1>Model to predict airport flight delays</h1>
# 
# Much of this model is taken from the work of Fabien Daniel:
# https://www.kaggle.com/code/fabiendaniel/predicting-flight-delays-tutorial/notebook.  
# The difference here is that we are modeling delays for all arrival airports and all airlines given a single departure airport.
# 
# We also incorporate MLflow tracking using the Python API.  
# 
# Input parameters for this script include:
# * num_alpha_increments:  The number of different Ridge regression alpha penalty values to try, spaced by 0.2 apart
#   
# Dependencies:
# * cleaned_data.csv is the input data file, structured appropriately.  The structure of this data file must be:
# 
# Outputs:
# * log file named "polynomial_regression.txt" containing information about the model training process
# * MLFlow experiment named with current date containing model training runs, one for each value of the Ridge regression penalty
# 
# | YEAR | MONTH | DAY | DAY_OF_WEEK | ORG_AIRPORT | DEST_AIRPORT | SCHEDULED_DEPARTURE | DEPARTURE_TIME | DEPARTURE_DELAY | SCHEDULED_ARRIVAL | ARRIVAL_TIME | ARRIVAL_DELAY |
# |:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
# | integer | integer | integer | integer | string | string | integer | integer | integer | integer | integer | integer |
# 

# In[1]:


# Here we import the packages we will need
import datetime
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, OneHotEncoder
from sklearn import metrics, linear_model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import logging
import os
import pickle
import json
import joblib

# In[2]:


# set up the argument parser
parser = argparse.ArgumentParser(description='Parse the parameters for the polynomial regression')
parser.add_argument('num_alphas', metavar='N', type=int, help='Number of Lasso penalty increments')
order = 1
#args = parser.parse_args()
#num_alpha_increments = args[0]
# Uncomment the two lines above and comment the line below to run this script from the command prompt or as part of an 
# MLFlow pipeline
num_alphas = 20


# In[3]:


# configure logger
logname = "polynomial_regression.txt"
logging.basicConfig(filename=logname,
                    filemode='w',
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.info("Flight Departure Delays Polynomial Regression Model Log")


# In[4]:


# read the data file
df = pd.read_excel("filtered_dataframe.xlsx")
tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})


# In[5]:


def grab_month_year(df:pd.DataFrame) -> tuple:
    """
    grab_month_year is a function to extract the month and year of the flights in the departure delay dataset.

    Parameters
    ----------
    df : pd.DataFrame
        the input data set in Pandas data frame format.

    Returns
    -------
    tuple
        (month,year) of the data set.

    Raises
    ------
    Exception
        If more than one month or year are found in the data set.
    """
    months = pd.unique(df['Month'])
    years = pd.unique(df['Year'])
    if len(months) >1:
        raise Exception("Multiple months found in data set, only one acceptable")
    else:
        month = int(months[0])
    if len(years) > 1:
        raise Exception("Multiple years found in data set, only one acceptable")
    else:
        year = int(years[0])
    return (month, year)


# In[6]:


def format_hour(string: str) -> datetime:
    """
    format_hour is a function to convert an 'HHMM' string input to a time in datetime format.

    Parameters
    ----------
    string : string
        An hour and minute in 'HHMM' format.

    Returns
    -------
    datetime
        An hour and minute (datetime.time).  Returns nan if input string is null.

    """    
    if pd.isnull(string):
        return np.nan
    else:
        if string == 2400: string = 0
        string = "{0:04d}".format(int(string))
        hour = datetime.time(int(string[0:2]), int(string[2:4]))
        return hour

def combine_date_hour(x: list) -> datetime:
    """
    combine_date_hour is a function that combines a date and time to produce a datetime.datetime

    Parameters
    ----------
    x : list
        A list containing a date and a time in datetime format.

    Returns
    -------
    datetime
        A combined date and time in datetime format. Returns nan if time is null.

    """
    if pd.isnull(x.iloc[0]) or pd.isnull(x.iloc[1]):
        return np.nan
    else:
        return datetime.datetime.combine(x.iloc[0],x.iloc[1])


def create_flight_time(df: pd.DataFrame, col: str) -> pd.Series:
    """
    create_flight_time is a function that combines two columns of a data frame to produce a datetime.datetime series.

    Parameters
    ----------
    df : pd.DataFrame
        A data frame containing flight departure delay data
    col: string
        The name of one of the columns in the data frame containing flight departure delay data

    Returns
    -------
    pd.Series
        A Pandas series of datetimes with combined date and time

    """
    list = []
    for index, cols in df[['Date', col]].iterrows():
        if pd.isnull(cols.iloc[1]):
            list.append(np.nan)
        elif float(cols.iloc[1]) == 2400:
            cols.iloc[0] += datetime.timedelta(days=1)
            cols.iloc[1] = datetime.time(0,0)
            list.append(combine_date_hour(cols))
        else:
            cols.iloc[1] = format_hour(cols.iloc[1])
            list.append(combine_date_hour(cols))
    return pd.Series(list)


# In[7]:


def create_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    create_df is a function that wrangles data within a flight departure delay data frame into the format needed for ML training.

    Parameters
    ----------
    df : pd.DataFrame
        A data frame containing flight departure delay data

    Returns
    -------
    pd.DataFrame
        A Pandas data frame with modified columns and data formats suitable for regression model training

    """
    df2 = df[['CRSDepTime','CRSArrTime',
                                    'DestAirportID','DepDelay']]
    df2 = df2.dropna(how = 'any')
    df2.loc[:,'weekday'] = df2['CRSDepTime'].apply(lambda x:x.weekday())
    #____________________
    # delete delays > 1h
    df2.loc[:,'DepDelay'] = df2['DepDelay'].apply(lambda x:x if x < 60 else np.nan)
    df2 = df2.dropna(how = 'any')
    #_________________
    # formating times
    fct = lambda x:x.hour*3600+x.minute*60+x.second
    df2.loc[:,'hour_depart'] = df2['CRSDepTime'].apply(lambda x:x.time())
    df2.loc[:,'hour_depart'] = df2['hour_depart'].apply(fct)
    df2.loc[:,'hour_arrive'] = df2['CRSArrTime'].apply(fct)
    df2 = df2[['hour_depart','hour_arrive',
            'DestAirportID','DepDelay','weekday']]
    df3 = df2.groupby(['hour_depart', 'hour_arrive', 'DestAirportID'],
                      as_index = False).mean()
    return df3


# In[8]:


nowdate = datetime.date.today()
# creates an experiment name that changes every day
experiment_name = "Airport Departure Delays, experiment run on " + str(nowdate)
# creates new experiment if there is not one yet today, otherwise sets the experiment to the existing one for today
#experiment = mlflow.set_experiment(experiment_name)
run_name = "Run started at " + datetime.datetime.now().strftime("%H:%M")
    


# In[9]:


df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
(month,year) = grab_month_year(df)
logging.info("Month and year of data: %s %s", month, year)
df['CRSDepTime'] = create_flight_time(df, 'CRSDepTime')
df['DepTime'] = df['DepTime'].apply(format_hour)
df['CRSArrTime'] = df['CRSArrTime'].apply(format_hour)
df['ArrTime'] = df['ArrTime'].apply(format_hour)


# In[10]:


# define training data as the first 3 weeks of the month, and test data as that from the fourth week of the month
df_train = df[df['CRSDepTime'].apply(lambda x:x.date()) < datetime.date(year, month, 23)]
df_test  = df[df['CRSDepTime'].apply(lambda x:x.date()) > datetime.date(year, month, 23)]


# In[11]:


df3 = create_df(df_train)


# In[12]:


# perform one-hot encoding of all destination airports in training data
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df3['DestAirportID'])
#_________________________________________________________
zipped = zip(integer_encoded, df3['DestAirportID'])
label_airports = list(set(list(zipped)))
label_airports.sort(key = lambda x:x[0])
#_________________________________________________
onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#_________________________________________________
b = np.array(df3[['hour_depart', 'hour_arrive']])
X = np.hstack((onehot_encoded, b))
Y = np.array(df3['DepDelay'])
Y = Y.reshape(len(Y), 1)
logging.info("Airport one-hot encoding successful")


# In[13]:


# train/validation split at 30%
X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=0.3)


# In[14]:



score_min = 10000
alpha_max = num_alphas * 2
count = 1
# loop through all alpha values
for alpha in range(0, alpha_max, 2):
    run_num = "Training Run Number " + str(count)
    # create a Ridge regressor with the stated alpha
    ridgereg = Ridge(alpha = alpha/10)
    # create polynomial features based on the polyniomial order
    poly = PolynomialFeatures(degree = order)
    # fit the model using the training data
    X_ = poly.fit_transform(X_train)
    ridgereg.fit(X_, Y_train)
    X_ = poly.fit_transform(X_validate)
    # predict against the validation data
    result = ridgereg.predict(X_)
    # how well did the model do when compared to the validation actuals?
    score = metrics.mean_squared_error(result, Y_validate)
    #mlflow.log_param("alpha",alpha/10)
    #mlflow.log_metric("Training Data Mean Squared Error",score)
    #mlflow.log_metric("Training Data Average Delay",np.sqrt(score))
    if score < score_min:
        score_min = score
        parameters = [alpha, order]
        logging.info("n={} alpha={} , MSE = {:<0.5}".format(order, alpha/10, score))
        count +=1
 # train and predict on validation data with optimal alpha found
X_ = poly.fit_transform(X_validate)
tresult = ridgereg.predict(X_)
tscore = metrics.mean_squared_error(tresult, Y_validate)
logging.info('Training Data Final MSE = {}'.format(round(tscore, 2)))
#mlflow.log_metric("Training Data Mean Squared Error",tscore)
#mlflow.log_metric("Training Data Average Delay",np.sqrt(tscore))
logging.info("Model training loop completed with %s iterations", count-1)


# In[16]:


# create a data frame of the test data
df3 = create_df(df_test)

label_conversion = dict()
for s in label_airports:
    label_conversion[s[1]] = int(s[0])

# export airport label conversion for test data to json file for later use
jsonout = json.dumps(label_conversion)
f = open("airport_encodings.json","w")

# write json object to file
f.write(jsonout)

# close file
f.close()
logging.info("Export of airport one-hot encoding successful")
df3.loc[:,'DestAirportID'] = df3.loc[:,'DestAirportID'].map(pd.Series(label_conversion))

# manually one-hot encode destination airports for test data
for index, label in label_airports:
    temp = df3['DestAirportID'] == index
    temp = temp.apply(lambda x:1.0 if x else 0.0)
    if index == 0:
        matrix = np.array(temp)
    else:
        matrix = np.vstack((matrix, temp))
matrix = matrix.T

b = np.array(df3[['hour_depart', 'hour_arrive']])
# Reshape matrix to have 2 dimensions if it's 1D
if len(matrix.shape) == 1:
    matrix = matrix.reshape(-1, 1)
# Make sure both arrays have the same number of rows
if matrix.shape[0] != b.shape[0]:
    # You might need to transpose matrix if the dimensions don't match
    matrix = matrix.reshape(b.shape[0], -1)

X_test = np.hstack((matrix, b))
Y_test = np.array(df3['DepDelay'])
Y_test = Y_test.reshape(len(Y_test), 1)
logging.info("Wrangling of test data successful")


# In[18]:


# create a data frame of the test data
df3 = create_df(df_test)

label_conversion = dict()
for s in label_airports:
    label_conversion[s[1]] = int(s[0])

# export airport label conversion for test data to json file for later use
jsonout = json.dumps(label_conversion)
f = open("airport_encodings.json","w")

# write json object to file
f.write(jsonout)

# close file
f.close()
logging.info("Export of airport one-hot encoding successful")
df3.loc[:,'DestAirportID'] = df3.loc[:,'DestAirportID'].map(pd.Series(label_conversion))

# manually one-hot encode destination airports for test data
for index, label in label_airports:
    temp = df3['DestAirportID'] == index
    temp = temp.apply(lambda x:1.0 if x else 0.0)
    if index == 0:
        matrix = np.array(temp)
    else:
        matrix = np.vstack((matrix, temp))
matrix = matrix.T

b = np.array(df3[['hour_depart', 'hour_arrive']])
# Reshape matrix to have 2 dimensions if it's 1D
if matrix.ndim == 1:
    matrix = matrix.reshape(-1, 1)  # Convert 1D array to 2D column vector
X_test = np.hstack((matrix, b))
Y_test = np.array(df3['DepDelay'])
Y_test = Y_test.reshape(len(Y_test), 1)
logging.info("Wrangling of test data successful")


# In[19]:


# create polynomial features based on order
X_ = poly.fit_transform(X_test)
# predict on last week of month data
result = ridgereg.predict(X_)
score = metrics.mean_squared_error(result, Y_test)
logging.info('Test Data MSE = {}'.format(round(score, 2)))
logging.info("Predictions using test data successful")


# In[20]:


logging.info('Test Data average delay = {:.2f} min'.format(np.sqrt(score)))


# In[21]:


# export final model
finalname = 'finalized_model.pkl'
pickle.dump(ridgereg, open(finalname, 'wb'))
logging.info("Final model export successful")


# In[22]:


# create and export model performance plot
tips = pd.DataFrame()

# Fix the indexing for 1D arrays
# Check if result and Y are 1D or 2D and handle accordingly
if len(result.shape) == 1:
    tips["prediction"] = pd.Series([float(s) for s in result])  # For 1D array
else:
    tips["prediction"] = pd.Series([float(s) for s in result[:,0]])  # For 2D array

if len(Y.shape) == 1:
    tips["original_data"] = pd.Series([float(s) for s in Y])  # For 1D array
else:
    tips["original_data"] = pd.Series([float(s) for s in Y[:,0]])  # For 2D array

# Alternative simpler approach if you know both are 1D:
# tips["prediction"] = pd.Series(result.astype(float))
# tips["original_data"] = pd.Series(Y.astype(float))

sns.jointplot(x="original_data", y="prediction", data=tips, height=6, ratio=7,
              joint_kws={'line_kws':{'color':'limegreen'}}, kind='reg')
plt.xlabel('Mean delays (min)', fontsize=15)
plt.ylabel('Predictions (min)', fontsize=15)
plt.plot(list(range(-10,25)), list(range(-10,25)), linestyle=':', color='r')
plt.savefig("model_performance_test.jpg", dpi=300)
logging.info("Model performance plot export successful")


# In[23]:


# create and export model performance plot
tips = pd.DataFrame()

# Fix the indexing for 1D arrays
# Check if arrays are 1D and adjust indexing accordingly
if len(result.shape) == 1:
    tips["prediction"] = pd.Series([float(s) for s in result])  # For 1D array
else:
    tips["prediction"] = pd.Series([float(s) for s in result[:,0]])  # For 2D array

if len(Y.shape) == 1:
    tips["original_data"] = pd.Series([float(s) for s in Y])  # For 1D array
else:
    tips["original_data"] = pd.Series([float(s) for s in Y[:,0]])  # For 2D array

sns.jointplot(x="original_data", y="prediction", data=tips, height = 6, ratio = 7,
              joint_kws={'line_kws':{'color':'limegreen'}}, kind='reg')
plt.xlabel('Mean delays (min)', fontsize = 15)
plt.ylabel('Predictions (min)', fontsize = 15)
plt.plot(list(range(-10,25)), list(range(-10,25)), linestyle = ':', color = 'r')
plt.savefig("model_performance_test.jpg",dpi=300)
logging.info("Model performance plot export successful")


# In[25]:


# TO DO: create an MLFlow run within the current experiment that logs the following as artifacts, parameters, 
# or metrics, as appropriate, within the experiment: 
# 1.  The informational log files generated from the import_data and clean_data scripts
# 2.  the input parameters (alpha and order) to the final regression against the test data
# 3.  the performance plot
# 4.  the model performance metrics (mean squared error and the average delay in minutes)

# YOUR CODE GOES HERE
# 1. The informational log files generated from the import_data and clean_data scripts
# Check if file exists before logging it
file_path = 'polynominal_regression.txt'
try:
    # You can either create the file if it doesn't exist
    if not os.path.exists(file_path):
        with open(model, 'w') as f:
            f.write("This is a placeholder for regression information")
        #mlflow.log_artifact(file_path)
except Exception as e:
        print(f"Error logging artifact: {e}")
        # Alternatively, you can log a different existing file or skip this step

# 2. The input parameters (alpha and order) to the final regression against the test data
alpha = 20
order = 1
#mlflow.log_param('alpha', alpha)
#mlflow.log_param('order', order)
model = make_pipeline(PolynomialFeatures(order), Ridge(alpha=alpha))
model.fit(X, Y)

x_test_poly = poly.transform(X_test)
predictions = ridgereg.predict(x_test_poly)
mse = mean_squared_error(Y_test, predictions)
average_delay = predictions.mean()

# 3. The performance plot
plt.figure()
plt.plot(Y_test, label = 'Actual', alpha = 0.7)
plt.plot(predictions, label = 'Predicted', alpha = 0.7)
plt.title('Model Perfomance')
plt.xlabel('Sample')
plt.ylabel('Delay')
plt.legend()
plt.savefig('performance_plot.png')
#mlflow.log_artifact('performance_plot.png')

# 4. The model performance metrics (mean squared error and the average delay in minutes)
# These should be inside the mlflow.start_run context
#mlflow.log_metric('mean_squared_error', mse)
#mlflow.log_metric('average_delay_minutes', average_delay)

print(mse)
print(average_delay)
artifact_dir = "./artifacts"
os.makedirs(artifact_dir, exist_ok=True)

# Save model
joblib.dump(model, os.path.join(artifact_dir, "model.pkl"))

# Save results
with open(os.path.join(artifact_dir, "results.json"), "w") as f:
    json.dump({"alpha": alpha, "order": order, "mse": mse}, f)
# End the run inside the context manager
#mlflow.end_run()

logging.shutdown()


# In[ ]:




