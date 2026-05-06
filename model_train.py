import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# 1. Dataset creation (Sample Fruit Recognition Dataset)
# Features: [mass, width, height, color_score]
# Labels: 1: Apple, 2: Mandarin, 3: Orange, 4: Lemon
def create_dataset():
    data = {
        'mass': [192, 180, 176, 86, 84, 80, 200, 190, 185, 118, 120, 115],
        'width': [8.4, 8.0, 7.4, 6.2, 6.0, 5.8, 7.6, 7.5, 7.5, 6.1, 6.0, 5.9],
        'height': [7.3, 6.8, 7.2, 4.7, 4.6, 4.3, 10.5, 10.2, 10.0, 8.1, 8.4, 8.0],
        'color_score': [0.55, 0.59, 0.60, 0.80, 0.79, 0.77, 0.65, 0.68, 0.66, 0.70, 0.72, 0.71],
        'fruit_label': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    }
    return pd.DataFrame(data)

def train_model():
    print("Loading dataset...")
    df = create_dataset()
    
    # Features and Labels
    X = df[['mass', 'width', 'height', 'color_score']]
    y = df['fruit_label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    # 2. Train a scikit-learn model
    # We use KNN for simplicity and effectiveness on small datasets
    print("Training KNN model...")
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    
    # 3. Print accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # 4. Save the trained model
    print("Saving model to model.pkl...")
    joblib.dump(model, 'model.pkl')
    
    # Save fruit names mapping for reference
    mapping = {1: 'Apple', 2: 'Mandarin', 3: 'Orange', 4: 'Lemon'}
    joblib.dump(mapping, 'mapping.pkl')
    
    print("Done!")

if __name__ == "__main__":
    train_model()
