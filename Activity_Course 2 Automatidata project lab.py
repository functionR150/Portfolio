#!/usr/bin/env python
# coding: utf-8

# ## **Automatidata project**
# **Course 2 - Get Started with Python**

# Welcome to the Automatidata Project!
# 
# You have just started as a data professional in a fictional data consulting firm, Automatidata. Their client, the New York City Taxi and Limousine Commission (New York City TLC), has hired the Automatidata team for its reputation in helping their clients develop data-based solutions.
# 
# The team is still in the early stages of the project. Previously, you were asked to complete a project proposal by your supervisor, DeShawn Washington. You have received notice that your project proposal has been approved and that New York City TLC has given the Automatidata team access to their data. To get clear insights, New York TLC's data must be analyzed, key variables identified, and the dataset ensured it is ready for analysis.
# 
# A notebook was structured and prepared to help you in this project. Please complete the following questions.

# # Course 2 End-of-course project: Inspect and analyze data
# 
# In this activity, you will examine data provided and prepare it for analysis.  This activity will help ensure the information is,
# 
# 1.   Ready to answer questions and yield insights
# 
# 2.   Ready for visualizations
# 
# 3.   Ready for future hypothesis testing and statistical methods
# <br/>    
# 
# **The purpose** of this project is to investigate and understand the data provided.
#   
# **The goal** is to use a dataframe contructed within Python, perform a cursory inspection of the provided dataset, and inform team members of your findings. 
# <br/>  
# *This activity has three parts:*
# 
# **Part 1:** Understand the situation 
# * Prepare to understand and organize the provided taxi cab dataset and information.
# 
# **Part 2:** Understand the data
# 
# * Create a pandas dataframe for data learning, future exploratory data analysis (EDA), and statistical activities.
# 
# * Compile summary information about the data to inform next steps.
# 
# **Part 3:** Understand the variables
# 
# * Use insights from your examination of the summary data to guide deeper investigation into specific variables.
# 
# 
# <br/> 
# Follow the instructions and answer the following questions to complete the activity. Then, you will complete an Executive Summary using the questions listed on the PACE Strategy Document.
# 
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work. 
# 
# 

# # **Identify data types and relevant variables using Python**
# 

# <img src="images/Pace.png" width="100" height="100" align=left>
# 
# # **PACE stages**
# 

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## PACE: **Plan**
# 
# Consider the questions in your PACE Strategy Document and those below to craft your response:

# ### **Task 1. Understand the situation**
# 
# *   How can you best prepare to understand and organize the provided taxi cab information? 

# 1. Read data dictionary.
# 2. Excute EDA by python.

# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## PACE: **Analyze**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.

# ### **Task 2a. Build dataframe**
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

# Create a pandas dataframe for data learning, and future exploratory data analysis (EDA) and statistical activities.
# 
# **Code the following,**
# 
# *   import pandas as `pd`. pandas is used for buidling dataframes.
# 
# *   import numpy as `np`. numpy is imported with pandas
# 
# *   `df = pd.read_csv('Datasets\NYC taxi data.csv')`
# 
# **Note:** pair the data object name `df` with pandas functions to manipulate data, such as `df.groupby()`.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[1]:


#Import libraries and packages listed above
import numpy as np
import pandas as pd

# Load dataset into dataframe
df = pd.read_csv('2017_Yellow_Taxi_Trip_Data.csv')
print("done")


# ### **Task 2b. Understand the data - Inspect the data**
# 
# View and inspect summary information about the dataframe by coding the following:
# 
# 1. `df.head(10)`
# 2. `df.info()`
# 3. `df.describe()`
# 
# Consider the following two questions:
# 
# **Question 1:** When reviewing the `df.info()` output, what do you notice about the different variables? Are there any null values? Are all of the variables numeric? Does anything else stand out?
# 
# **Question 2:** When reviewing the `df.describe()` output, what do you notice about the distributions of each variable? Are there any questionable values?

# 1. float64(8), int64(7), object(3). No any null vlaue.
# 2. There are some varible has negative values(fare_amount,	extra,	mta_tax, improvement_surcharge,	total_amount)

# In[ ]:


df.head(10)


# In[2]:


df.info()


# In[3]:


df.describe()


# In[69]:


mask = (df["fare_amount"] < 0) | (df["extra"] <0) | (df["mta_tax"] <0) | (df["improvement_surcharge"] <0) | (df["total_amount"] <0)
df[mask]


# ### **Task 2c. Understand the data - Investigate the variables**
# 
# Sort and interpret the data table for two variables:`trip_distance` and `total_amount`.
# 
# **Answer the following three questions:**
# 
# **Question 1:** Sort your first variable (`trip_distance`) from maximum to minimum value, do the values seem normal?
# 
# **Question 2:** Sort by your second variable (`total_amount`), are any values unusual?
# 
# **Question 3:** Are the resulting rows similar for both sorts? Why or why not?
# 

# 1. Several values are 0, which appears to be abnormal.
# 2. There is one very large value, 1200, and some values are negative. 
# 3. No, the `total_amount` does not appear to decrease proportionally with a decrease in `trip_distance`.

# In[6]:


df[["trip_distance"]].sort_values(by='trip_distance', ascending=False)


# In[7]:


df[["total_amount"]].sort_values(by='total_amount', ascending=False)


# In[11]:


total_amount_and_distance = df[["total_amount", "trip_distance"]].sort_values(by='trip_distance', ascending=False)
total_amount_and_distance.head(30)


# In[29]:


total_amount_and_distance.reset_index(drop=True)


# In[33]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))  # Set the figure size

# sns.lineplot(data=total_amount_and_distance)
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.title('Line Plot of Columns A and B')
# plt.show()

# Create the line plot
fig, ax1 = plt.subplots()  # Create a figure and primary axis

sns.lineplot(x=total_amount_and_distance.reset_index(drop=True).index, y='total_amount', data=total_amount_and_distance.reset_index(drop=True), ax=ax1)  # Plot A on primary axis (left)
ax2 = ax1.twinx()  # Create a secondary axis that shares the X-axis
sns.lineplot(x=total_amount_and_distance.reset_index(drop=True).index, y='trip_distance', data=total_amount_and_distance.reset_index(drop=True), ax=ax2, color='red')  # Plot B on secondary axis (right, red color)

# Customize labels and title (optional)
ax1.set_xlabel('Index')
ax1.set_ylabel('Values (total_amount)', color='blue')  # Set color for left axis label
ax2.set_ylabel('Values (trip_distance)', color='red')  # Set color for right axis label
plt.title('Line Plots of Columns total_amount and trip_distance by total_amount DESC (Separate Y-Axes)')
plt.show()


# In[70]:


sns.pairplot(df)
plt.show()


# According to the data dictionary, the payment method was encoded as follows:
# 
# 1 = Credit card  
# 2 = Cash  
# 3 = No charge  
# 4 = Dispute  
# 5 = Unknown  
# 6 = Voided trip

# In[40]:


# What is the average tip for trips paid for with credit card?

# df.info()
# df.columns
df[["tip_amount","payment_type"]].groupby("payment_type").mean()

# What is the average tip for trips paid for with cash?


# In[51]:


# How many times is each vendor ID represented in the data?
df[["VendorID"]].value_counts()


# In[55]:


# df.column
# What is the mean total amount for each vendor?
df[["total_amount", "VendorID"]].groupby("VendorID").mean()


# In[ ]:


# Filter the data for credit card payments only

# Filter the credit-card-only data for passenger count only


# In[64]:


# Calculate the average tip amount for each passenger count (credit card payments only)
df.columns
tip_amount_by_payment_and_passenger = df[["payment_type", "tip_amount", "passenger_count"]].groupby(["payment_type", "passenger_count"]).mean()
tip_amount_by_payment_and_passenger


# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## PACE: **Construct**
# 
# **Note**: The Construct stage does not apply to this workflow. The PACE framework can be adapted to fit the specific requirements of any project. 
# 
# 
# 

# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## PACE: **Execute**
# 
# Consider the questions in your PACE Strategy Document and those below to craft your response.
# 

# ### **Given your efforts, what can you summarize for DeShawn and the data team?**
# 
# *Note for Learners: Your notebook should contain data that can address Luana's requests. Which two variables are most helpful for building a predictive model for the client: NYC TLC?*

# 1. fare_amount, and tip_amount looks most helpful to the total_amount prediction
# 2. payment_type, passanger_count looks most helpful to the fare_amount prediciton 

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
