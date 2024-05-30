#!/usr/bin/env python
# coding: utf-8

# # **Regression Analysis**
# **The goal** is to build a multiple linear regression model and evaluate the model for ride fares based on a variety of variables.
# 

# ### Task 1. Imports and loading
# Import the packages that you've learned are needed for building linear regression models.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error


# In[2]:


# Load dataset into dataframe 
df0=pd.read_csv("2017_Yellow_Taxi_Trip_Data.csv") 


# ### Task 2a. Explore data with EDA
# 
# Analyze and discover data, looking for correlations, missing data, outliers, and duplicates.

# In[3]:


# Check for data type
df0.info()


# In[4]:


# Check for missing data
df0.isna().sum()


# In[5]:


# Check for duplicates
df0.duplicated().sum()


# In[6]:


# Check for unusual data
df0.describe(include='all')


# In[7]:


# Check unusual data size
mask = ((df0["fare_amount"] < 0) | (df0["extra"] < 0) | (df0["mta_tax"] < 0) | (df0["improvement_surcharge"] < 0) | (df0["total_amount"] < 0))
print("total data size: ", len(df0))
print("Unusual data size: ", len(df0[mask]))
print("Unusual data size porpotion: ", len(df0[mask])/len(df0))


# Note: Identified some unexpected data points (e.g., negative fare amounts). Given their limited quantity, we opted to remove them to minimize their impact on following analysis.

# In[8]:


# Remove unusual data
df = df0[~mask]
df.describe()


# Note: Some things stand out from this table of summary statistics. For instance, there are clearly some outliers in several variables, like `tip_amount` (200) and `total_amount` (1,200). Also, a number of the variables, such as mta_tax, seem to be almost constant throughout the data, which would imply that they would not be expected to be very predictive.

# ### Task 2b. Convert pickup & dropoff columns to datetime
# 

# In[9]:


# Check the format of the data
df[["tpep_pickup_datetime","tpep_dropoff_datetime"]].info()


# In[10]:


# Convert datetime columns to datetime
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
df[["tpep_pickup_datetime","tpep_dropoff_datetime"]].info()


# ### Task 2c. Create duration column

# Create a new column called `duration` that represents the total number of minutes that each taxi ride took.

# In[11]:


# Create `duration` column
df["duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
df.head()


# ### Task 2d. Use Box plots to check outliers

# Check outliers for each feature: `trip_distance`, `fare_amount`, `duration`.

# In[12]:


#Create box plots
df_long = pd.melt(df, value_vars=['trip_distance', 'fare_amount', 'duration'], var_name='Columns', value_name='Values')
fig, ax = plt.subplots(3, 1, figsize=(12, 6))
sns.boxplot(x='trip_distance', data=df, ax=ax[0])
ax[0].set_title('Boxplot for Outlier detection')
sns.boxplot(x='fare_amount', data=df, ax=ax[1])
sns.boxplot(x='duration', data=df, ax=ax[2])
plt.tight_layout()
plt.show()


# Note: 
# 1. All three variables contain outliers. Some are extreme, but others not so much.
# 2. It's 30 miles from the southern tip of Staten Island to the northern end of Manhattan and that's in a straight line. With this knowledge and the distribution of the values in this column, it's reasonable to leave these values alone and not alter them. However, the values for `fare_amount` and `duration` definitely seem to have problematic outliers on the higher end.
# 3. There are trip distances of 0, let's investigate further to determine if these represent errors in the data or if they're extremely short trips that were rounded down to zero."

# ### Task 2e. Imputations

# #### `trip_distance` outliers
# 
# Sort the column values, eliminate duplicates, and inspect the least 10 values. Check if they are rounded values or precise values?

# In[13]:


a = set(df["trip_distance"])
a_sort = sorted(a)
a_sort[:10]


# Note: The distances are captured with a high degree of precision. However, it might be possible for trips to have distances of zero if a passenger summoned a taxi and then changed their mind. 
# Next, check if there are enough zero values in the data to pose a problem?

# In[14]:


# Calculate the count of rides where the trip_distance is zero.
mask = df["trip_distance"] == 0
(df[mask].size)/(df.size)


# Note: 0.0065 rides is relatively insignificant, but it's unlikely to have much of an effect on the model. Therefore, the trip_distance column will remain untouched with regard to outliers.
# 
# 

# #### `fare_amount` outliers

# The range of values in the fare_amount column is large. The maximum fare amount in this dataset is nearly 1,000, which seems very unlikely. High values for this feature can be capped based on intuition and statistics. The interquartile range (IQR) is 8. The standard formula of Q3 + (1.5 * IQR) yields 26.50. That doesn't seem appropriate for the maximum fare cap. In this case, we'll use a factor of `6`, which results in a cap of 62.50

# In[15]:


df['fare_amount'].describe()


# In[16]:


def outlier_imputer(column_list, iqr_factor):
    '''
    Impute upper-limit values in specified columns based on their interquartile range.

    Arguments:
        column_list: A list of columns to iterate over
        iqr_factor: A number representing x in the formula:
                    Q3 + (x * IQR). Used to determine maximum threshold,
                    beyond which a point is considered an outlier.

    The IQR is computed for each column in column_list and values exceeding
    the upper threshold for each column are imputed with the upper threshold value.
    '''
    for col in column_list:
        # Reassign minimum to zero
        df.loc[df[col] < 0, col] = 0

        # Calculate upper threshold
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_threshold = q3 + (iqr_factor * iqr)
        print(col)
        print('q3:', q3)
        print('upper_threshold:', upper_threshold)

        # Reassign values > threshold to threshold
        df.loc[df[col] > upper_threshold, col] = upper_threshold
        print(df[col].describe())
        print()


# In[17]:


# Impute the maximum value as Q3 + (6 * IQR).
outlier_imputer(['fare_amount'], 6) 


# #### `duration` outliers
# 

# In[18]:


# Impute the maximum value as Q3 + (6 * IQR).
outlier_imputer(['duration'], 6) 


# ### Task 3a. Feature engineering

# #### Create `mean_distance` and `mean_duration` column
# 
# When deployed, the model will not know the `distance` and `duration` of a trip until after the trip occurs, so we cannot train a model that uses this feature. Therefore, we can identify patterns and use those to estimate the duration of future trips. In essence, we're leveraging historical data to generalize about unseen scenarios.
# 
# Create a column called `mean_distance` that captures the mean distance for each group of trips that share pickup and dropoff points.
# 
# For example, if your data were:
# 
# |Trip|Start|End|Distance|
# |--: |:---:|:-:|    |
# | 1  | A   | B | 1  |
# | 2  | C   | D | 2  |
# | 3  | A   | B |1.5 |
# | 4  | D   | C | 3  |
# 
# The results should be:
# ```
# A -> B: 1.25 miles
# C -> D: 2 miles
# D -> C: 3 miles
# ```

# In[19]:


# Create `pickup_dropoff` column
df["pickup_dropoff"]=df["PULocationID"].apply(str) + " - " + df["DOLocationID"].apply(str)


# In[20]:


df.head()


# In[21]:


# Caculate the mean duration and mean distance for each group of trips 
grouped_distiance_duration = df[["pickup_dropoff", "trip_distance","duration"]].groupby("pickup_dropoff").mean().reset_index()
grouped_distiance_duration.rename(columns={'trip_distance': 'mean_trip_distance', 'duration': 'mean_duration'}, inplace=True)
grouped_distiance_duration


# In[22]:


# Merge data
df = pd.merge(df, grouped_distiance_duration, on='pickup_dropoff', how='left')
df.head()


# #### Create `day` and `month` columns
# 
# Create two new columns, `day` (name of day) and `month` (name of month) by extracting the relevant information from the `tpep_pickup_datetime` column.

# In[23]:


# Create 'day' col
df["day"] = df["tpep_pickup_datetime"].dt.day_name()
# Create 'month' col
df["month"] = df["tpep_pickup_datetime"].dt.month_name()


# #### Create `rush_hour` column
# 
# Define rush hour as:
# * Any weekday (not Saturday or Sunday) AND
# * Either from 06:00&ndash;10:00 or from 16:00&ndash;20:00
# 
# Create a binary `rush_hour` column that contains a 1 if the ride was during rush hour and a 0 if it was not.

# In[24]:


def is_weekday_and_time_range(dt):
  """
  Checks if the datetime object falls on a weekday (not Saturday or Sunday)
  and within the time ranges 06:00-10:00 or 16:00-20:00.

  Args:
      dt (pd.Timestamp): The datetime object to check.

  Returns:
      int: 1 if conditions are met, 0 otherwise.
  """
  # Check weekday (Monday = 0, Sunday = 6)
  if dt.weekday() in [0, 1, 2, 3, 4]:
    # Check time range
    if (dt.hour >= 6 and dt.hour < 10) or (dt.hour >= 16 and dt.hour < 20):
      return 1
  return 0

# Create a new column with 0 or 1 based on conditions
df['rush_hour'] = df['tpep_pickup_datetime'].apply(is_weekday_and_time_range)

# Print the modified DataFrame
df.head()


# ### Task 4. Scatter plot
# 
# Create a scatterplot to visualize the relationship between `mean_duration` and `fare_amount`.

# In[25]:


# Create a scatterplot to visualize the relationship between variables of interest
f = plt.figure()
f.set_figwidth(5)
f.set_figheight(5)
sns.scatterplot(x='mean_duration', y='fare_amount', data=df)
plt.ylim(0, 70)
plt.xlim(0, 70)
plt.xlabel("mean_duration")
plt.ylabel("fare_amount")
plt.title("Scatter Plot")
plt.grid(True)
plt.show()


# Note: There are two horizontal lines aorune fare amounts of 50 ~ 65, we know that the one of lines represents 62.5 which we imputed for outliers. Next, check the value of the rides in the second horizontal line in the scatter plot.

# In[26]:


# Count for each group of fare amount 
mask = df["fare_amount"] >= 50
df["fare_amount"][mask].value_counts().head()


# In[27]:


# Examine the first 30 of the trips which fare amount is 52. 
pd.set_option('display.max_columns', None)
mask = df["fare_amount"] == 52
df[mask].head(30)


# Note: It seems that almost all of the trips in the first 30 rows where the fare amount was $52 either begin or end at location 132, and all of them have a RatecodeID of 2.
# 
# There is no readily apparent reason why PULocation 132 should have so many fares of 52 dollars. They seem to occur on all different days, at different times, with both vendors, in all months. However, there are many toll amounts of  5.76and
#  5.54. This would seem to indicate that location 132 is in an area that frequently requires tolls to get to and from. It's likely this is an airport.
# 
# The data dictionary says that RatecodeID of 2 indicates trips for JFK, which is John F. Kennedy International Airport. A quick Google search for "new york city taxi flat rate $52" indicates that in 2017 (the year that this data was collected) there was indeed a flat fare for taxi trips between JFK airport (in Queens) and Manhattan.
# 
# Because RatecodeID is known from the data dictionary, the values for this rate code can be imputed back into the data after the model makes its predictions. This way you know that those data points will always be correct.

# ### Task 5. Isolate modeling variables
# 
# Drop features that are redundant, irrelevant, or that will not be available in a deployed environment.

# In[28]:


df.info()


# In[29]:


df2 = df[["VendorID", "passenger_count", "fare_amount", "mean_trip_distance", "mean_duration","rush_hour"]]
df2.info()


# ### Task 6. Pair plot
# 
# Create a pairplot to visualize pairwise relationships between `fare_amount`, `mean_duration`, and `mean_distance`.

# In[30]:


# Create a pairplot to visualize pairwise relationships between variables in the data
sns.pairplot(df2[["fare_amount", "mean_trip_distance", "mean_duration"]])
plt.show()


# Note: These variables all show linear correlation with each other. Investigate this further.

# ### Task 7. Identify correlations

# Next, code a correlation matrix to help determine most correlated variables.

# In[31]:


# Calculate the Pearson correlation matrix
correlation_matrix = df2.corr()
correlation_matrix


# Visualize a correlation heatmap of the data.

# In[32]:


# Create correlation heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# Note: `mean_duration` and `mean_distance` are both highly correlated with the target variable of fare_amount They're also both correlated with each other, with a Pearson correlation of 0.87.
# 
# Though the highly correlated predictor variables can be bad for linear regression models. However, the prediction itself is more important in this case. Therefore, keep modeling with both variables even though they are correlated. 

# ### Task 8a. Split data into outcome variable and features

# Set your X and y variables. X represents the features and y represents the outcome (target) variable.

# In[33]:


# Remove the target column from the features
X = df2.drop(columns='fare_amount')

# Set y variable
y = df2[["fare_amount"]]

# Display first few rows
X.head(5)


# ### Task 8b. Pre-process data
# 

# Dummy encode categorical variables

# In[34]:


# Convert VendorID to string
X["VendorID"] = X["VendorID"].astype(str)

# Get dummies
X = pd.get_dummies(X, drop_first=True)
X.head()


# ### Split data into training and test sets

# Create training and testing sets. The test set should contain 20% of the total samples. Set `random_state=0`.

# In[35]:


# Create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ### Standardize the data
# 
# Standardize the `X_train` variables. Assign the results to a variable called `X_train_scaled`.

# In[36]:


# Standardize the X training data
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
X_train_scaled.columns = X_train.columns
X_train_scaled  


# In[37]:


# Standardize the X test data
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test))
X_test_scaled.columns = X_test.columns
X_test_scaled  


# ### Task 8c. Fit and evaluate model

# #### 1st model

# In[38]:


# Combine standardized features
data_std_scaled = pd.concat([X_train_scaled, y_train.reset_index()], axis=1) 

# Fit 1st model to the training data
formula = "fare_amount ~ + passenger_count + mean_trip_distance + mean_duration + rush_hour + VendorID_2"
model_1 = sm.formula.ols(formula, data_std_scaled).fit()
print(model_1.summary())

# Calcuate evluation metrics for training data
print("training data")
model_1_predictions = model_1.predict(data_std_scaled)

# Calculate and print r2
model_1_r2 = r2_score(y_train, model_1_predictions)
print(f"R-squared : {model_1_r2:.2f}")
# Calculate and print MAE
model_1_mae = mean_absolute_error(y_train, model_1_predictions)
print(f"Mean Absolute Error (MAE): {model_1_mae:.2f}")
# Calculate and print MSE
model_1_mse = mean_squared_error(y_train, model_1_predictions)
print(f"Mean Squared Error (MSE): {model_1_mse:.2f}")
# Calculate and print RMSE (square root of MSE)
model_1_rmse = np.sqrt(model_1_mse)
print(f"Root Mean Squared Error (RMSE): {model_1_rmse:.2f}")
print()

# Calcuate evluation metrics for testing data
print("Testing data")
test_data_std_scaled = pd.concat([X_test_scaled, y_test.reset_index()], axis=1) 
model_1_test_data_predictions = model_1.predict(test_data_std_scaled)

# Calculate and print r2
model_1_test_data_r2 = r2_score(y_test, model_1_test_data_predictions)
print(f"R-squared : {model_1_r2:.2f}")
# Calculate and print MAE
model_1_test_data_mae = mean_absolute_error(y_test, model_1_test_data_predictions)
print(f"Mean Absolute Error (MAE): {model_1_mae:.2f}")
# Calculate and print MSE
model_1_test_data_mse = mean_squared_error(y_test, model_1_test_data_predictions)
print(f"Mean Squared Error (MSE): {model_1_mse:.2f}")
# Calculate and print RMSE (square root of MSE)
model_1_test_data_rmse = np.sqrt(model_1_test_data_mse)
print(f"Root Mean Squared Error (RMSE): {model_1_test_data_rmse:.2f}")


# Note: The model performance is high on both training and test sets, suggesting that there is little bias in the model and that the model is not overfit. In fact, the test scores were even better than the training scores.
# 
# For the test data, an R2 of 0.84 means that 84% of the variance in the fare_amount variable is described by the model.
# 
# Consider the `passenger_count` and `VendorID` are not significnat for level of 5%, which revealed that `passenger_count` and `VendorID` have low importance. Therefore, remove these two features and rebuild model to assess if performance improves due to a more focused feature set.

# #### 2nd model

# In[39]:


# Fit 2nd model to the training data
formula = "fare_amount ~ + mean_trip_distance + mean_duration + rush_hour"
model_2 = sm.formula.ols(formula, data_std_scaled).fit()
print(model_2.summary())

# Calcuate evluation metrics for training data
model_2_predictions = model_2.predict(data_std_scaled)

# Calculate and print r2
model_2_r2 = r2_score(y_train, model_2_predictions)
print(f"R-squared : {model_1_r2:.2f}")
# Calculate and print MAE
model_2_mae = mean_absolute_error(y_train, model_2_predictions)
print(f"Mean Absolute Error (MAE): {model_2_mae:.2f}")
# Calculate and print MSE
model_2_mse = mean_squared_error(y_train, model_2_predictions)
print(f"Mean Squared Error (MSE): {model_2_mse:.2f}")
# Calculate and print RMSE (square root of MSE)
model_2_rmse = np.sqrt(model_2_mse)
print(f"Root Mean Squared Error (RMSE): {model_2_rmse:.2f}")

print()
print("test data")
model_2_test_data_predictions = model_2.predict(test_data_std_scaled)
# Calculate and print r2
model_2_test_data_r2 = r2_score(y_test, model_2_test_data_predictions)
print(f"R-squared : {model_2_r2:.2f}")
# Calculate and print MAE
model_2_test_data_mae = mean_absolute_error(y_test, model_2_test_data_predictions)
print(f"Mean Absolute Error (MAE): {model_2_mae:.2f}")
# Calculate and print MSE
model_2_test_data_mse = mean_squared_error(y_test, model_2_test_data_predictions)
print(f"Mean Squared Error (MSE): {model_2_mse:.2f}")
# Calculate and print RMSE (square root of MSE)
model_2_test_data_rmse = np.sqrt(model_2_test_data_mse)
print(f"Root Mean Squared Error (RMSE): {model_2_test_data_rmse:.2f}")


# Note: While removing `passenger_count` and `VendorID` did not result in significant performance improvement, we've opted to retain these features in the first model for the sake of model interpretability. These features provide valuable insights into the factors affecting fare amounts, even if their statistical significance is below the 5% threshold."

# ### Task 9a. Results
# 
# Use the code cell below to get `actual`,`predicted`, and `residual` for the testing set, and store them as columns in a `results` dataframe.

# In[40]:


# Create a `results` dataframe
results = pd.DataFrame(data={'actual': y_test["fare_amount"],
                             'predicted': model_1_test_data_predictions.ravel()})
results['residual'] = results['actual'] - results['predicted']
results.head()


# ### Task 9b. Visualize model results

# Create a scatterplot to visualize `actual` vs. `predicted`.

# In[41]:


# Create a scatter plot
fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x='actual', y='predicted', data=results)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("Actual vs. predicted")
plt.grid(True) 
plt.plot([0,60], [0,60], c='red', linewidth=2)
plt.show()


# Visualize the distribution of the `residuals` using a histogram.

# In[42]:


# Create the histogram
sns.histplot(
    data=results,
    x="residual", 
    kde=True, 
    linewidth=0.5,  
    alpha=0.7,  
    stat="density"  
)
plt.title('Distribution of the residuals')
plt.xlabel('residual value')
plt.ylabel('count')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlim(-15,15)
plt.show()


# Note: The distribution of the residuals is approximately normal and has a mean aorund 0. The residuals represent the variance in the outcome variable that is not explained by the model. A normal distribution around zero is good, as it demonstrates that the model's errors are evenly distributed and unbiased.

# Create a scatterplot of `residuals` over `predicted`.

# In[43]:


# Create a scatterplot of `residuals` over `predicted`
sns.scatterplot(x='predicted', y='residual', data=results)
plt.xlabel("predicted")
plt.ylabel("residual")
plt.title("Scatterplot of residuals over predicted values'")
plt.grid(True)
plt.axhline(0, c='red')
plt.show()


# Note: The most model's residuals are evenly distributed above and below zero, with the exception of the sloping lines from the upper-left corner to the lower-right corner, which we know are the imputed maximum of 62.50 and the flat rate of 52 for JFK airport trips.

# ### Task 9c. Coefficients

# In[44]:


# Get the model's coefficients. 
model_1.params


# In[45]:


# Translate the coefficient of `mean_trip_distance` back to miles instead of standard deviation 
# 1. Calculate SD of `mean_distance` in X_train data
print(X_train['mean_trip_distance'].std())
# 2. Divide the model coefficient by the standard deviation
print(7.163740 / X_train['mean_trip_distance'].std())


# The coefficients reveal that `mean_trip_distance` was the feature with the greatest weight in the model's final prediction. And controlling for other variables, **for every +1 change in standard deviation, the fare amount increases by a mean of 7.13**. In intuitive interpretation: **for every 3.57 miles traveled, the fare increased by a mean of 7.13.** In simpler terms: **for every 1 mile traveled, the fare increased by a mean of 2.00**.
