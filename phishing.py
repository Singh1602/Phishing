import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import spacy
from flask import Flask, request, jsonify
import re
import email
from email import policy
from email.parser import BytesParser

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

class PhishingDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = RandomForestClassifier(n_estimators=100)
        
    def preprocess_text(self, text):
        # Clean and normalize text
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        doc = nlp(text)
        tokens = [token.text for token in doc if not token.is_stop]
        return ' '.join(tokens)
    
    def extract_email_features(self, email_content):
        if isinstance(email_content, bytes):
            msg = BytesParser(policy=policy.default).parsebytes(email_content)
        else:
            msg = BytesParser(policy=policy.default).parsestr(email_content)
            
        subject = msg.get('subject', '')
        body = ''
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload()
        else:
            body = msg.get_payload()
            
        # Combine subject and body for feature extraction
        full_text = f"{subject} {body}"
        return self.preprocess_text(full_text)
    
    def train(self, X_train, y_train):
        # Transform text data to TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_tfidf, y_train)
    
    def predict(self, email_content):
        processed_text = self.extract_email_features(email_content)
        features = self.vectorizer.transform([processed_text])
        prediction = self.classifier.predict(features)
        probability = self.classifier.predict_proba(features)
        return {
            'is_phishing': bool(prediction[0]),
            'confidence': float(max(probability[0]))
        }

# Initialize Flask app
app = Flask(__name__)
detector = PhishingDetector()

# Load and prepare training data
def load_training_data():
    # This is a placeholder - replace with actual dataset loading
    data = pd.read_csv('phishing_dataset.csv')  # You'll need to create/obtain this dataset
    X = data['email_content']
    y = data['is_phishing']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
X_train, X_test, y_train, y_test = load_training_data()
detector.train(X_train, y_train)

@app.route('/scan', methods=['POST'])
def scan_email():
    if not request.is_json:
        return jsonify({'error': 'Content type must be application/json'}), 400
    
    email_content = request.json.get('email_content')
    if not email_content:
        return jsonify({'error': 'Email content is required'}), 400
    
    result = detector.predict(email_content)
    return jsonify(result)

def test_model():
    # Transform test data
    X_test_processed = [detector.preprocess_text(text) for text in X_test]
    X_test_tfidf = detector.vectorizer.transform(X_test_processed)
    
    # Make predictions
    y_pred = detector.classifier.predict(X_test_tfidf)
    
    # Print classification report
    print("Model Performance:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    # Run tests
    test_model()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)
