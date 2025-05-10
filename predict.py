import joblib

# Load your trained model
model = joblib.load("models/phishing_detector.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")

def predict(email_text):
    X = vectorizer.transform([email_text])
    return "⚠ PHISHING" if model.predict(X)[0] == 1 else "✅ LEGIT"

# Demo
if _name_ == "_main_":
    print(predict("Claim your free iPhone now!"))  # Should output: ⚠ PHISHING