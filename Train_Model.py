import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle  # Import pickle for model saving

wine = pd.read_csv('wine_data.csv')

le = LabelEncoder()
wine['type'] = le.fit_transform(wine['type'])

X = wine.drop('quality', axis=1)
y = wine['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model (same as before)
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)

# Save the model to a pickle file
filename = 'winequalitymodel.pkl'  # Replace with your desired filename
with open(filename, 'wb') as f:
    pickle.dump(model_rf, f)

print(f"Model saved to pickle file: {filename}")