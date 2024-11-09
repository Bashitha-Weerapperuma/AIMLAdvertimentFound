import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from app.utils.model_manager import save_model  # Import save_model function

def train():
    try:
        # Load and preprocess data
        df = pd.read_csv('data/dataset.csv')  # Load dataset from the 'data' folder

        # Label encode the 'system_type' column
        label_encoder = LabelEncoder()
        df['system_type'] = label_encoder.fit_transform(df['system_type'])

        # Features (excluding the target column 'label')
        X = df.drop(columns=['label'])         
        # Target variable (label column)
        y = df['label']                        

        # Split data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model (RandomForestClassifier as an example)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Make predictions on the test data
        y_pred = model.predict(X_test)
        
        # Check accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save the trained model using the save_model function
        save_model(model)  # This function is defined in app/utils/model_manager.py
        
        # Print model accuracy as a percentage
        print(f"Model trained successfully! Accuracy: {accuracy * 100:.2f}%")
    
    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    train()
