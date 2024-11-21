import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class SalesPredictionBot:
    def __init__(self, historical_data=None):
        """
        Initialize the Sales Prediction Bot
        
        Args:
            historical_data (pd.DataFrame): Optional historical sales data
        """
        self.model = None
        self.scaler = StandardScaler()
        self.historical_data = historical_data
        
    def preprocess_data(self, data):
        """
        Preprocess input data for model training
        
        Args:
            data (pd.DataFrame): Input sales data
        
        Returns:
            pd.DataFrame: Processed features
        """
        # Feature engineering
        data['month'] = pd.to_datetime(data['date']).dt.month
        data['quarter'] = pd.to_datetime(data['date']).dt.quarter
        data['year'] = pd.to_datetime(data['date']).dt.year
        
        # Add lag features
        for lag in [1, 2, 3]:
            data[f'sales_lag_{lag}'] = data['sales'].shift(lag)
        
        # Rolling statistics
        data['sales_rolling_mean'] = data['sales'].rolling(window=3).mean()
        data['sales_rolling_std'] = data['sales'].rolling(window=3).std()
        
        return data.dropna()
    
    def train_model(self, features, target, test_size=0.2):
        """
        Train a Random Forest Regressor for sales prediction
        
        Args:
            features (pd.DataFrame): Input features
            target (pd.Series): Target variable (sales)
            test_size (float): Proportion of data for testing
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        print("Model Performance Metrics:")
        print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
        print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        print(f"R-squared Score: {r2_score(y_test, y_pred):.2f}")
    
    def predict_sales(self, future_data):
        """
        Predict future sales
        
        Args:
            future_data (pd.DataFrame): DataFrame with features for prediction
        
        Returns:
            np.array: Predicted sales values
        """
        if self.model is None:
            raise ValueError("Model must be trained first!")
        
        # Preprocess future data
        processed_data = self.preprocess_data(future_data)
        
        # Scale features
        future_scaled = self.scaler.transform(processed_data)
        
        # Predict
        predictions = self.model.predict(future_scaled)
        return predictions
    
    def visualize_predictions(self, actual, predicted):
        """
        Visualize actual vs predicted sales
        
        Args:
            actual (pd.Series): Actual sales values
            predicted (np.array): Predicted sales values
        """
        plt.figure(figsize=(12, 6))
        plt.plot(actual.index, actual.values, label='Actual Sales', color='blue')
        plt.plot(actual.index, predicted, label='Predicted Sales', color='red', linestyle='--')
        plt.title('Sales: Actual vs Predicted')
        plt.xlabel('Time')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

# Example usage
def main():
    # Sample data creation (replace with your actual sales data)
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='M')
    sales = np.cumsum(np.random.normal(1000, 200, len(dates))) + \
            np.linspace(0, 5000, len(dates))  # Add trend
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales
    })
    
    # Initialize and use the Sales Prediction Bot
    bot = SalesPredictionBot()
    
    # Preprocess data
    processed_data = bot.preprocess_data(df)
    
    # Prepare features and target
    features = processed_data.drop(['date', 'sales'], axis=1)
    target = processed_data['sales']
    
    # Train model
    bot.train_model(features, target)
    
    # Predict future sales
    future_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
    future_df = pd.DataFrame({'date': future_dates})
    
    predictions = bot.predict_sales(future_df)
    
    # Visualize results
    bot.visualize_predictions(target, bot.model.predict(
        bot.scaler.transform(features)
    ))

if __name__ == "__main__":
    main()
