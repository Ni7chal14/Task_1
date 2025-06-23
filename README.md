# Task_1
Elevate Labs Internship Task-1 23rd June 2025

#For data importing and handling
import pandas as pd
import numpy as np

#For data visualiaztion
import matplotlib.pyplot as plt
import seaborn as sns

#For standardizing the numerical features
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# 1.) Import the dataset and explore basic info (nulls, data types)

#Importing the dataset
df=pd.read_csv("/content/Titanic-Dataset.csv")

#Info of the imported dataset
df.info()

#Number of nulls in this dataset
print(df.isnull().sum())

#First 5 data of the dataset
print(df.head())

# 2.) Handle missing values using mean/median/imputation.

# Replacing the null/empty values with median of Age
df['Age'].fillna(df['Age'].median(),inplace=True)
# df['Age'].fillna(np.median(df['Age'].dropna()), inplace=True)  <-------we can also use numpy for median.

# Replacing the null/empty values with the mode of Embarked
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)

# Dropped the cabin column as it has too many empty or missing values
df.drop(columns=['Cabin'],inplace=True)

# Checked if there are any null or missing values left
print(df.isnull().sum())

# 3.) Convert categorical features into numerical using encoding.

le=LabelEncoder()
#Encoded Sex column and changed the categories of male and female to numerical that is 0 and 1.
#Used Laber Encoding
df['Sex']=le.fit_transform(df['Sex']).astype(int)

#Encoded Embarked column and changed the categories of C,S,Q to numerical that is if either of S or Q is 1 then the 1 marked column is the value and if both are 0 then C is the value
#Used One-hot encoded
df=pd.get_dummies(df,columns=['Embarked'], drop_first=True)
df[['Embarked_Q', 'Embarked_S']] = df[['Embarked_Q', 'Embarked_S']].astype(int)

print(df.head())

#4.) Standardize the numerical features.

#Combined the numerical columns into one place that is array
cols=['Age', 'Fare', 'SibSp', 'Parch']

#Used StandardScaler() to standardize the columns
scaler=StandardScaler()
df[cols]=scaler.fit_transform(df[cols])

print(df.head())

#5.Visualize outliers using boxplots and remove them.

# Plot boxplots for numerical columns
for col in cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

#making a copy so that data is not overwritten
df_outlier_removed = df.copy()
for col in cols:
    Q1 = df_outlier_removed[col].quantile(0.25)
    Q3 = df_outlier_removed[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Keep only rows within the bounds
    df_outlier_removed = df_outlier_removed[
        (df_outlier_removed[col] >= lower_bound) & 
        (df_outlier_removed[col] <= upper_bound)
    ]


print("Original shape:", df.shape)
print("After removing outliers:", df_outlier_removed.shape)








1. What are the different types of missing data?

Ans : There are three types:

MCAR – Missing completely at random (no pattern)

MAR – Missing at random (depends on other columns)

MNAR – Missing not at random (depends on itself)

2. How do you handle categorical variables?

Ans : For 2 unique values, I use Label Encoding. For more than 2 categories, I prefer One-Hot Encoding to avoid confusing the model.

3. What’s the difference between normalization and standardization?

Ans : Normalization scales data between 0 and 1. Standardization centers it around mean = 0, std = 1. I usually standardize when data has outliers or different scales.

4. How do you detect outliers?

Ans : I use boxplots to visualize, and the IQR method to remove values that are too far from the rest.

5. Why is preprocessing important in ML?

Ans : Because raw data is messy. Preprocessing improves accuracy, performance, and helps the model learn the real patterns.

6. What is one-hot encoding vs label encoding?

Ans : Label Encoding: Converts categories to numbers (like male = 1, female = 0) One-Hot Encoding: Creates separate columns for each category (like Embarked_Q, Embarked_S)

7. How do you handle data imbalance?

Ans : I use techniques like SMOTE, undersampling, or class weights to make sure the model doesn’t favor the majority class too much.
