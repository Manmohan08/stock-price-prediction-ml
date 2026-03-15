from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate(df):
    """
    Trains a Random Forest Classifier to predict stock price direction.
    """
    print("Training the machine learning model...")
    
    # Features for the model
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50']
    
    X = df[features]
    y = df['Target']
    
    # Split into train and test sets (chronological split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Initialize and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    predictions = model.predict(X_test)
    
    # Evaluation
    acc = accuracy_score(y_test, predictions)
    print(f"\\nModel Accuracy: {acc:.2f}\\n")
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    
    return model
