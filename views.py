import pandas as pd
from django.shortcuts import render
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import os
from sklearn.linear_model import LinearRegression
from django.core.files.storage import FileSystemStorage
# Load dataset from CSV
def load_dataset(file_path):
    return pd.read_csv(file_path)
# Preprocess the dataset
def preprocess_data(df):
    # Label Encoding for Risk Level (If exists in the dataset)
    if 'Risk_Level' in df.columns:
        le = LabelEncoder()
        df['Risk_Level'] = le.fit_transform(df['Risk_Level'])
    
    # Scaling the numerical features
    scaler = StandardScaler()
    df[['Monthly_Savings', 'Investment_Horizon', 'Current_Investment']] = scaler.fit_transform(
        df[['Monthly_Savings', 'Investment_Horizon', 'Current_Investment']]
    )
    return df, scaler

# Train the Investment Model using KMeans Clustering
def train_investment_model(df):
    features = ['Monthly_Savings', 'Risk_Level', 'Investment_Horizon', 'Current_Investment']
    X = df[features]
    
    # Clustering the data into 3 clusters (for simplicity)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
    return kmeans, df

# Train Expense Prediction Model using Linear Regression
def train_expense_model(df):
    X = df[['Mthly_HH_Income', 'No_of_Fly_Members', 'Emi_or_Rent_Amt', 'Annual_HH_Income', 'No_of_Earning_Members']]
    y = df['Mthly_HH_Expense']
    
    model = LinearRegression()
    model.fit(X, y)
    return model

# Train Lifestyle Affordability Model using Linear Regression
def train_lifestyle_model(df):
    X = df[['Mthly_HH_Income', 'Emi_or_Rent_Amt', 'Annual_HH_Income']]
    y = df['Mthly_HH_Expense']
    
    model = LinearRegression()
    model.fit(X, y)
    return model

# Investment recommendations logic
def recommend_investments(savings, risk_level, horizon, model, scaler, data):
    input_data = pd.DataFrame({
        'Monthly_Savings': [savings],
        'Risk_Level': [risk_level],
        'Investment_Horizon': [horizon],
        'Current_Investment': [0]
    })
    
    # Scaling the input data
    input_scaled = scaler.transform(input_data)
    
    # Predicting the cluster for the input
    cluster = model.predict(input_scaled)[0]
    
    # Get recommendations based on the cluster
    cluster_data = data[data['Cluster'] == cluster]
    recommendations = {
        'Bonds (%)': cluster_data['Bonds_Percentage'].mean(),
        'Stocks (%)': cluster_data['Stocks_Percentage'].mean(),
        'Mutual Funds (%)': cluster_data['Mutual_Funds_Percentage'].mean()
    }
    return recommendations

# Expense Prediction logic
def predict_expenses(income, family_size, rent, annual_income, earning_members, model):
    input_data = pd.DataFrame({
        'Mthly_HH_Income': [income],
        'No_of_Fly_Members': [family_size],
        'Emi_or_Rent_Amt': [rent],
        'Annual_HH_Income': [annual_income],
        'No_of_Earning_Members': [earning_members]
    })
    
    expenses = model.predict(input_data)
    return expenses[0]

# Lifestyle Affordability logic
def check_lifestyle_affordability(income, rent, annual_income, model):
    input_data = pd.DataFrame({
        'Mthly_HH_Income': [income],
        'Emi_or_Rent_Amt': [rent],
        'Annual_HH_Income': [annual_income]
    })
    
    predicted_expense = model.predict(input_data)
    
    if income >= predicted_expense:
        return "Affordable"
    else:
        return "Not Affordable"

# Django view to handle the input form and generate recommendations
def index(request):
    recommendations = None
    expenses = None
    lifestyle_status = None

    if request.method == 'POST' and request.FILES['csv_file']:
        # Get uploaded file
        csv_file = request.FILES['csv_file']
        
        # Save file temporarily
        fs = FileSystemStorage()
        file_path = fs.save(csv_file.name, csv_file)
        
        # Load dataset, preprocess, and train the models
        dataset = load_dataset(file_path)
        processed_data, scaler = preprocess_data(dataset)
        
        # Train models
        kmeans_model, clustered_data = train_investment_model(processed_data)
        expense_model = train_expense_model(dataset)
        lifestyle_model = train_lifestyle_model(dataset)
        
        # Get user input from the form
        savings = float(request.POST.get('monthly_savings'))
        risk_level = int(request.POST.get('risk_level'))
        horizon = int(request.POST.get('investment_horizon'))
        income = float(request.POST.get('income'))
        family_size = int(request.POST.get('family_size'))
        rent = float(request.POST.get('rent'))
        annual_income = float(request.POST.get('annual_income'))
        earning_members = int(request.POST.get('earning_members'))
        
        # Get investment recommendations
        recommendations = recommend_investments(savings, risk_level, horizon, kmeans_model, scaler, clustered_data)
        
        # Predict monthly expenses
        expenses = predict_expenses(income, family_size, rent, annual_income, earning_members, expense_model)
        
        # Check lifestyle affordability
        lifestyle_status = check_lifestyle_affordability(income, rent, annual_income, lifestyle_model)

    return render(request, 'index.html', {
        'recommendations': recommendations,
        'expenses': expenses,
        'lifestyle_status': lifestyle_status
    })
