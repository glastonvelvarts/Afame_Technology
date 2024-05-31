# Project Overview

This project involves analyzing a sizable sales dataset to glean insightful information and using the Titanic dataset to build a survival prediction model. The goal is to identify patterns, best-selling items, and revenue indicators to help in business decision-making and to predict whether a passenger on the Titanic survived or not.

## Part 1: Sales Data Analysis

### Goal

Use sales data analysis to find patterns, best-selling items, and revenue indicators to help in business decision making.

### Steps and Methodology

#### 1. Data Cleaning and Preparation

- **Load Dataset**: Import the sales dataset.
- **Handle Missing Values**: Impute or remove missing values to ensure data quality.

#### 2. Exploratory Data Analysis (EDA)

- **Visualize Data**: Create visualizations to understand the distribution of features and their relationship with revenue.
- **Correlation Analysis**: Analyze the correlation between features and revenue to identify important predictors.

#### 3. Revenue Analysis

- **Total Sales**: Calculate total sales and other revenue measures.
- **Sales Trends Over Time**: Analyze sales trends over time using time-series analysis.
- **Best-Selling Products**: Identify the best-selling products based on sales data.

#### 4. Visualization

- **Visualize Sales Trends**: Create visualizations to depict sales trends over time.
- **Visualize Best-Selling Products**: Create visualizations to show the best-selling products.

### Example Code Snippets

#### Total Sales Calculation

```python
import pandas as pd

# Load dataset
df_sales = pd.read_excel('ECOMM_DATA.xlsx')

# Ensure 'Order Date' is in proper date-time format
df_sales['Order Date'] = pd.to_datetime(df_sales['Order Date'])

# Calculate total sales
total_sales = df_sales['Sales'].sum()
print("Total Sales:", total_sales)
```

#### Sales Trends Over Time

```python
import matplotlib.pyplot as plt

# Set 'Order Date' as the index
df_sales.set_index('Order Date', inplace=True)

# Resample the data to get monthly sales
monthly_sales = df_sales['Sales'].resample('M').sum()

# Plot the data
plt.figure(figsize=(14, 7))
plt.plot(monthly_sales, marker='o', linestyle='-', color='red')  # Set line color to red
plt.title('Monthly Sales Trends')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show()
```

#### Best-Selling Products

```python
# Identify best-selling products
best_selling_products = df_sales.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)
print("Best-Selling Products:")
print(best_selling_products)
```

## Part 2: Titanic Survival Prediction

### Goal

Use the Titanic dataset to build a model that predicts whether a passenger on the Titanic survived or not.

### Steps and Methodology

#### 1. Data Cleaning and Preparation

- **Load Dataset**: Import the Titanic dataset.
- **Handle Missing Values**: Impute or remove missing values to ensure data quality.
- **Data Transformation**: Convert categorical variables to numerical values using techniques like label encoding.

#### 2. Exploratory Data Analysis (EDA)

- **Visualize Data**: Create visualizations to understand the distribution of features and their relationship with the target variable (Survived).
- **Correlation Analysis**: Analyze the correlation between features and the target variable to identify important predictors.

#### 3. Feature Engineering

- **Feature Selection**: Select relevant features based on exploratory analysis and domain knowledge.
- **Create New Features**: Generate new features that might help improve model performance.

#### 4. Model Building

- **Train-Test Split**: Split the data into training and testing sets.
- **Model Training**: Train various models to predict survival, including:
  - Random Forest Classifier
  - Stacking Ensemble Model (to improve accuracy by combining multiple models)

#### 5. Model Evaluation

- **Accuracy Score**: Calculate the accuracy of each model.
- **Classification Report**: Generate a classification report to evaluate model performance in terms of precision, recall, and F1-score.
- **Model Comparison**: Compare the performance of different models and select the best one based on evaluation metrics.

### Example Code Snippets

#### Data Cleaning

```python
import pandas as pd

# Load dataset
df_titanic = pd.read_csv('titanic_data.csv')

# Handle missing values
df_titanic['Age'].fillna(df_titanic['Age'].mean(), inplace=True)
df_titanic['Embarked'].fillna(df_titanic['Embarked'].mode()[0], inplace=True)
df_titanic.drop(columns=['Cabin'], inplace=True)

# Convert categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_titanic['Sex'] = le.fit_transform(df_titanic['Sex'])
df_titanic['Embarked'] = le.fit_transform(df_titanic['Embarked'])
```

#### Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Define features and target
X = df_titanic.drop(columns=['Survived'])
y = df_titanic['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Initialize and train model
rf_model = RandomForestClassifier(n_estimators=160, random_state=42)
rf_model.fit(X_train_imputed, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_imputed)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
```

#### Stacking Ensemble Model

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Define base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=160, random_state=42)),
    ('svc', SVC(probability=True)),
    ('xgb', XGBClassifier())
]

# Define stacking classifier
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression()
)

# Train stacking model
stacking_model.fit(X_train_imputed, y_train)

# Make predictions
y_pred_stack = stacking_model.predict(X_test_imputed)

# Evaluate model
accuracy_stack = accuracy_score(y_test, y_pred_stack)
print("Stacking Model Accuracy:", accuracy_stack)
print("Stacking Model Classification Report:")
print(classification_report(y_test, y_pred_stack))
```

## Conclusion

This project demonstrates the ability to work with and extract knowledge from large datasets, providing data-driven suggestions for improving sales tactics and building a robust model to predict Titanic passenger survival. Using advanced machine learning techniques, including Random Forest and Stacking Ensemble models, significantly improves prediction accuracy.
