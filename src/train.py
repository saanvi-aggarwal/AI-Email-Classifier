import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

print("Loading labeled data...")
df = pd.read_csv('data/labeled_emails.csv')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['priority'],
    test_size=0.2,
    random_state=42,
    stratify=df['priority']
)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

print("\nTraining model...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')),
    ('clf',   LogisticRegression(max_iter=1000, class_weight='balanced'))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\n--- Model Results ---")
print(classification_report(y_test, y_pred, target_names=['high', 'low', 'medium']))

# Save confusion matrix as image
cm = confusion_matrix(y_test, y_pred, labels=['high', 'low', 'medium'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['high', 'low', 'medium'],
            yticklabels=['high', 'low', 'medium'])
plt.title('Confusion Matrix - Email Priority Classifier')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png')
print("\nConfusion matrix saved to models/confusion_matrix.png")

# Save the model
joblib.dump(pipeline, 'models/email_classifier.pkl')
print("Model saved to models/email_classifier.pkl")
print("\nDone! Model is ready to use.")