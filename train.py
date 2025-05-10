import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load data
data = pd.read_csv("data/emails.csv")

# 2. Convert text to numbers
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(data['text'])
y = data['label']

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. Save for production
joblib.dump(vectorizer, "models/vectorizer.joblib")
joblib.dump(model, "models/phishing_detector.joblib")
print("✅ Training complete! Model saved to /models")