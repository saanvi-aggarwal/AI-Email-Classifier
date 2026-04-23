import joblib
import re

model = joblib.load('models/email_classifier.pkl')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_priority(subject, body):
    text = clean_text(subject + ' ' + body)
    pred  = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    classes = model.classes_
    scores = {cls: round(float(prob)*100, 1) for cls, prob in zip(classes, proba)}
    confidence = round(max(proba)*100, 1)
    return {
        'priority':   pred,
        'confidence': confidence,
        'scores':     scores
    }