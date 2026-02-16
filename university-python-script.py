# %% [markdown]
# # University Admission Analysis by Joshua Osarumwense
# 
# 

# %%
#Import Pandas Library, import my dataset and view first 5 rows

import pandas as pd
df = pd.read_csv(r"C:\Users\USER\Desktop\DATA ANALYSIS\PORTFOLIO\University Admission Analysis using Python\university_admission_dataset.csv")
df.head()


# %%
# Remove leading and trailing spaces from all column names in the DataFrame
df.columns = df.columns.str.strip()

# %%
#Check for column names, data types and null values

df.info()


# %%
# Check for missing values
df.isnull().sum()


# %%
# Check for duplicates
df.duplicated().sum()

# %%
#Generating a statistical summary of the numerical columns

df.describe()

# %%
import matplotlib.pyplot as plt

# Plot histogram for the TOEFL Score
plt.figure(figsize=(8, 6))
plt.hist(df['TOEFL Score'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of TOEFL Scores')
plt.xlabel('TOEFL Score')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()


# %%
import matplotlib.pyplot as plt

# Plot histogram for the GRE Score
plt.figure(figsize=(8, 6))
plt.hist(df['GRE Score'], bins=10, color='lightcoral', edgecolor='black')
plt.title('Distribution of GRE Scores')
plt.xlabel('GRE Score')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()


# %%
import matplotlib.pyplot as plt

# Count the values for the 'Research' column (0 and 1)
research_counts = df['Research'].value_counts()

# Plot the pie chart
plt.figure(figsize=(6, 6))
plt.pie(research_counts, labels=['No Research', 'Research'], autopct='%1.1f%%', colors=['lightblue', 'salmon'])
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
plt.show()


# %%
#Generating a Correlation Matrix and displaying as a structured table

import pandas as pd 
from IPython.display import display 

correlation_matrix=df.corr()

display(correlation_matrix)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Drop the 'Serial No.' column before calculating the correlation matrix
df_without_serial_no = df.drop(columns=['Serial No.'])

# Calculate the correlation matrix
correlation_matrix = df_without_serial_no.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.3)
plt.title("Correlation Heatmap of Factors involved in University Admission")
plt.show()

# %%
#Taking a Closer look at the correlation between Admission, GRE and TOEFL scores

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

correlation1 = np.corrcoef(df['GRE Score'], df['Chance of Admission'])[0, 1]
correlation2 = np.corrcoef(df['TOEFL Score'], df['Chance of Admission'])[0, 1]

# First scatter plot: GRE Score vs Chance of Admission
sns.scatterplot(x="GRE Score", y="Chance of Admission", data=df, ax=axes[0])
axes[0].set_title(f"GRE Score vs Chance of Admission\nCorrelation Coefficient :{correlation1:.2f}")

# Second scatter plot: TOEFL Score vs Chance of Admission
sns.scatterplot(x="TOEFL Score", y="Chance of Admission", data=df, ax=axes[1])
axes[1].set_title(f"TOEFL Score vs Chance of Admission\nCorrelation Coefficient :{correlation2:.2f}")


plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

correlation=np.corrcoef(df["CGPA"],df["Chance of Admission"])[0,1]

sns.scatterplot(x="CGPA", y="Chance of Admission", data=df)
plt.title(f"CGPA vs Chance of Admission\nCorrelation Coefficient :{correlation:.2f}")
plt.show()

# %%
from sklearn.model_selection import train_test_split

# Features (X) and target (y)
X = df.drop(['Serial No.', 'Chance of Admission'], axis=1)  # Drop the target and non-relevant columns
y = df['Chance of Admission']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions using the test data
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')


# %%
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Set up the parameter grid for tuning (without 'normalize')
param_grid = {
    'fit_intercept': [True, False]  # Only tune the 'fit_intercept' parameter
}

# Initialize GridSearchCV with the Linear Regression model
grid_search = GridSearchCV(estimator=LinearRegression(), param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Apply scaling using StandardScaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform the features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the grid search to the scaled training data
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters found by GridSearchCV
print(f'Best Parameters: {grid_search.best_params_}')

# Use the best model found from GridSearchCV
best_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred_best = best_model.predict(X_test_scaled)

# Evaluate the best model's performance
from sklearn.metrics import mean_squared_error, r2_score

mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

# Print the evaluation metrics for the best model
print(f'Best Model - Mean Squared Error: {mse_best}')
print(f'Best Model - R^2 Score: {r2_best}')


# %%



