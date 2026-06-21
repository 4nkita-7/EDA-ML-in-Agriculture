#Exploratory Data Analysis (EDA) and Machine Learning on Agricultural Yield Dataset

# Part A: Understanding the Dataset

#1 Dataset Overview
import pandas as pd

df = pd.read_csv("/Users/ankitabehera/Downloads/agriculture_yield_dataset.csv")


print("Rows and Columns:", df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nFirst 10 Records:")
print(df.head(10))

#2 Data Types and Missing Values
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

#3 Descriptive Statistics
print(df.describe())
''' Feature with highest mean value
 rainfall_mm = 754.05
Feature with highest standard deviation
 rainfall_mm = 255.10 '''


# Part B: Exploratory Data Analysis (EDA)

#4 Distribution Analysis
import matplotlib.pyplot as plt

cols = ['rainfall_mm',
        'temperature_c',
        'fertilizer_kg',
        'yield_ton_per_hectare']

for col in cols:
    plt.figure(figsize=(5,4))
    plt.hist(df[col], bins=20)
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

'''Observations
* rainfall_mm
Values range approximately from 300–1200 mm.
Distribution appears fairly uniform.
No extreme outliers are visible.

* temperature_c
Most values lie between 18°C and 38°C.
Distribution is almost symmetric.
No noticeable skewness.

* fertilizer_kg
Values spread between about 50–250 kg.
Distribution is fairly even.
No obvious outliers.

*yield_ton_per_hectare
Most values lie between 4 and 6 tons/hectare.
Distribution is approximately bell-shaped.
Very few extreme values.'''


#5 Crop Type Analysis
import seaborn as sns

print(df['crop_type'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x='crop_type', data=df)
plt.show()

''' Most frequent crop - Cotton'''

#6 Soil Type Analysis
print(df['soil_type'].value_counts())

plt.figure(figsize=(5,4))
sns.countplot(x='soil_type', data=df)
plt.show()
''' Most common soil type - Clay'''

#7 Yield Distribution
plt.figure(figsize=(5,4))
plt.hist(df['yield_ton_per_hectare'], bins=20)
plt.title("Yield Distribution")
plt.show()
'''  1)Distribution is approximately normal.
     2)No significant outliers are observed.'''

#8 Scatter Plot Analysis
plt.figure(figsize=(5,4))
plt.scatter(df['rainfall_mm'],
            df['yield_ton_per_hectare'])
plt.xlabel("Rainfall")
plt.ylabel("Yield")
plt.show()


plt.figure(figsize=(5,4))
plt.scatter(df['fertilizer_kg'],
            df['yield_ton_per_hectare'])
plt.xlabel("Fertilizer")
plt.ylabel("Yield")
plt.show()

''' Rainfall shows a stronger relationship with crop yield compared to fertilizer usage.'''

#9 Correlation Analysis
corr = df.select_dtypes(include='number').corr()

plt.figure(figsize=(7,5))
sns.heatmap(corr,
            annot=True,
            cmap='coolwarm')

plt.show()

print(corr['yield_ton_per_hectare'].sort_values(ascending=False))

''' Top 3 geatures correlated with yield are - 
rainfall_mm - 0.554 ; irrigation_hours - 0.543 ; fertilizer_kg - 0.278'''

#10 Group-Based Analysis
crop_avg = df.groupby('crop_type')['yield_ton_per_hectare'].mean()

soil_avg = df.groupby('soil_type')['yield_ton_per_hectare'].mean()

print(crop_avg)

print()

print(soil_avg)

''' 1)Highest yielding crop - Rice
    2)Highest yielding soil - Loamy'''



# Part C: Data Preparation

#11 Feature Encoding
categorical_columns = ['crop_type',
                       'soil_type']

print(categorical_columns)

encoded_df = pd.get_dummies(df)

print(encoded_df.head())

''' Categorical columns are:
crop_type
soil_type '''

#12 Feature Selection
X = encoded_df.drop('yield_ton_per_hectare',
                    axis=1)

y = encoded_df['yield_ton_per_hectare']

print(X.head())

print(y.head())

''' Target Variable
yield_ton_per_hectare
'''



# Part D: Machine Learning

#13 Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

X,
y,
test_size=0.2,
random_state=42
)

print("X_train:",X_train.shape)
print("X_test:",X_test.shape)

print("y_train:",y_train.shape)
print("y_test:",y_test.shape)

#14 Linear Regression Model
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("Intercept")
print(model.intercept_)

coefficients = pd.DataFrame({

'Feature':X.columns,
'Coefficient':model.coef_

})

print(coefficients)

''' 1) Intercept: 1.9111
    2) Feature with highest positive coefficient: crop_type_Rice
    3)Coefficient = 0.477 '''