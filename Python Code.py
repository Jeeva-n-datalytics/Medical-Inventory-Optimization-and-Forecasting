
############################   MEDICAL INVENTORY OPTIMIZATION Pythom code ######

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import pyplot
import mysql.connector
import csv
import seaborn as sns
import scipy.stats as stats
import pylab
from sklearn.metrics import mean_squared_error
import statsmodels.graphics.tsaplots as tsa_plots
from math import sqrt

# Connect to MySQL database
conn = mysql.connector.connect(
    host="127.0.0.1",
    database="med_inventory",
    user="root",
    password="root")


with conn.cursor() as cur:
    cur.execute("SELECT * FROM dataset")
    rows = cur.fetchall()
    
    
for row in rows:
    print(row)
pharma_data = pd.DataFrame(rows, columns=["Typeofsales", "Patient_ID", "Specialisation", "Dept", "Dateofbill", "Quantity",
                           "ReturnQuantity", "Final_Cost", "Final_Sales", "RtnMRP", "Formulation", "DrugName", "SubCat", "SubCat1"])


pharma_data.head(10)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

#### Dataframe Shape & Data Type

pharma_data.shape

pharma_data.dtypes

#### Type Casting

pharma_data["Patient_ID"] = pharma_data["Patient_ID"].astype('str')
pharma_data["Final_Sales"] = pharma_data["Final_Sales"].astype('float32')
pharma_data["Final_Cost"] = pharma_data["Final_Cost"].astype('float32')

pharma_data.dtypes

#### Handling Duplicates

duplicate = pharma_data.duplicated()  
sum(duplicate)

# Remove duplicates
pharma_data = pharma_data.drop_duplicates() 
duplicate = pharma_data.duplicated()
sum(duplicate) 

#### Handling Missing Values

pharma_data.replace('', pd.NA, inplace=True)

pharma_data.isnull().sum()

group_cols = ['Typeofsales', 'Specialisation', 'Dept']

##### Imputation (Mode)

# Impute missing values in Formulation column based on the mode of the group
for col in ['Formulation', 'DrugName', 'SubCat', 'SubCat1']:
    pharma_data[col] = pharma_data.groupby(group_cols)[col].apply(
        lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x)
    
    
pharma_data.isnull().sum()
    

# Still there are some missing values that need to be dropped
pharma_data.dropna(inplace=True)
pharma_data= pharma_data.reset_index(drop=True)
pharma_data.isnull().sum()
    
#### Data Manipulation

date_column = 'Dateofbill'
pharma_data[date_column] = pd.to_datetime(pharma_data[date_column])

# Sort dataset by date column in ascending order
pharma_data = pharma_data.sort_values(by=date_column, ascending=True)
pharma_data


# Specify Final Cost column to round
column_name = 'Final_Cost'

# Specify number of decimal places to round to 0
decimal_places = 0

# Round the values in the column to 0
pharma_data[column_name] = pharma_data[column_name].apply(
    lambda x: round(x, decimal_places))

# Specify Final Sales column to round
column_name1 = 'Final_Sales'

# Specify number of decimal places to round to 0
decimal_places1 = 0

# Round values in the column to 0
pharma_data[column_name1] = pharma_data[column_name1].apply(
    lambda x: round(x, decimal_places1))

pharma_data.drop(columns=["ReturnQuantity"], axis=1, inplace=True)


#### Describe Data

pharma_data.head(10)

pharma_data.describe()

#### First Moment Business Decision 

### Measure of Central Tendancy

# Mean
pharma_data.mean()


# Median
pharma_data.median()


# Mode
pharma_data.mode()

#### Second Moment Business Decision

##### Measure of Dispersion

# Variance
pharma_data.var()


# Standard Deviation
print(pharma_data.std())


#### Third Moment Business Decision

##### Skewness
pharma_data.skew()


##### Kurtosis
pharma_data.kurt()


#### EDA
pharma_data.Quantity.max()


plt.hist(pharma_data.Quantity, color = 'red', bins = 20, alpha = 1)
plt.xlim(0,160)

pharma_data.Final_Cost.max()



plt.hist(pharma_data.Final_Cost, color = 'red', bins = 500, alpha = 1)
plt.xlim(0,3500)

pharma_data.Final_Sales.max()


plt.hist(pharma_data.Final_Sales, color = 'red', bins = 500, alpha = 1)
plt.xlim(0,4000)

# Positively skewed shows means greater than median

pharma_data.RtnMRP.max()

plt.hist(pharma_data.RtnMRP, color = 'red', bins = 100, alpha = 1)
plt.xlim(0,1000)


# Convert date formate to month
pharma_data['Dateofbill'] = pd.to_datetime(pharma_data['Dateofbill'])
pharma_data['Dateofbill'] = pharma_data['Dateofbill'].dt.strftime('%b')
pharma_data.head()

# Pivot the DataFrame based on SubCat of drugs
data_pivoted = pharma_data.pivot_table(index="SubCat", columns="Dateofbill", values="Quantity")

# Result
data_pivoted.head()

#### Data Distribution

# Distribution of data
stats.probplot(pharma_data.Quantity, dist="norm", plot=pylab)


## Data Transformation : Log Transformation

# Transform the data to a normal distribution
stats.probplot(np.log(pharma_data.Quantity),dist="norm",plot=pylab)


## Bar Plot (Quantity of drugs sold by Month)

sns.barplot(data = pharma_data, x = 'Dateofbill', y = 'Quantity')
plt.title('Quantity of drugs sold by Month')
plt.show()


## Trend in Quantity

Month = pharma_data.groupby('Dateofbill')['Quantity'].sum()
plt.plot(Month.index, Month.values, color = 'blue')
plt.title('Quantity Trend')
plt.xlabel('Month')
plt.ylabel('Quantity')

plt.show()



# ## Automated Libraries

# AutoEDA
import sweetviz
my_report = sweetviz.analyze([pharma_data, "pharma_data"])

my_report.show_html('Report.html')

## AutoEDA (D-Tale)

import dtale as dt
dt.show(pharma_data)

