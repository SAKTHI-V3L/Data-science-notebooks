import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load data
data = pd.read_csv(r"C:\Users\svelo\Downloads\Attrition data.csv")

# Display basic information about the data
print(data.info())

# Drop columns that are not useful or have constant values
data.drop(['Over18', 'StandardHours'], axis=1, inplace=True)

# Separate numeric and categorical columns
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Fill missing values
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# For categorical columns, fill missing values with the mode
for col in categorical_cols:
    mode_value = data[col].mode()
    if not mode_value.empty:
        data[col].fillna(mode_value[0], inplace=True)

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Display the first few rows to verify preprocessing
print(data.head())

# Split data into features and target
X = data.drop('Attrition', axis=1)
y = data['Attrition']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Identify feature importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns)

# Initialize the Dash app
app = dash.Dash(__name__)

# Dynamic visualizations based on data characteristics
def create_dynamic_figures(data):
    figures = []
    for column in data.columns:
        if column == 'Attrition':
            continue
        
        unique_values = data[column].nunique()
        if data[column].dtype == 'object' or unique_values < 10:
            # Bar plot for categorical or low cardinality features
            fig = px.bar(data, x=column, y='Attrition', title=f'Attrition by {column}', barmode='group')
        else:
            # Box plot for numerical features
            fig = px.box(data, x='Attrition', y=column, title=f'{column} Distribution by Attrition')
        
        figures.append(fig)
    
    # Add feature importance plot
    fig_importance = px.bar(feature_importances.sort_values(ascending=False),
                            title='Feature Importances')
    figures.append(fig_importance)
    
    return figures

# Create initial figures
figures = create_dynamic_figures(data)

# Define the app layout
app.layout = html.Div([
    html.H1('Employee Attrition Dashboard'),
    
    html.Div([
        dcc.Graph(figure=fig) for fig in figures
    ])
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
