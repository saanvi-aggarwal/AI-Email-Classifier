import pandas as pd
import re
import email
from sklearn.model_selection import train_test_split

def parse_email(raw_message):
    msg = email.message_from_string(raw_message)
    subject = msg.get('Subject', '')
    body = ''
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                break
    else:
        body = msg.get_payload(decode=True)
        if isinstance(body, bytes):
            body = body.decode('utf-8', errors='ignore')
        elif body is None:
            body = str(msg.get_payload())
    return str(subject), str(body)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def label_email(subject, body):
    text = (subject + ' ' + body).lower()
    high_keywords = ['urgent', 'asap', 'immediately', 'deadline', 'critical',
                     'action required', 'important', 'respond', 'emergency']
    low_keywords  = ['newsletter', 'unsubscribe', 'promotion', 'offer',
                     'no-reply', 'noreply', 'advertisement', 'deal', 'sale']
    if any(k in text for k in high_keywords):
        return 'high'
    elif any(k in text for k in low_keywords):
        return 'low'
    else:
        return 'medium'

print("Loading emails... (this may take 1-2 minutes)")
df = pd.read_csv('data/emails.csv')

# Use only 5000 emails to keep it fast
df = df.sample(5000, random_state=42).reset_index(drop=True)

print("Parsing subjects and bodies...")
subjects, bodies = [], []
for raw in df['message']:
    s, b = parse_email(raw)
    subjects.append(s)
    bodies.append(b)

df['subject'] = subjects
df['body']    = bodies
df['text']    = df['subject'] + ' ' + df['body']
df['text']    = df['text'].apply(clean_text)
df['priority'] = df.apply(lambda r: label_email(r['subject'], r['body']), axis=1)

# Drop rows with very short text
df = df[df['text'].str.len() > 20].reset_index(drop=True)

print("\nPriority distribution:")
print(df['priority'].value_counts())

df[['text', 'priority']].to_csv('data/labeled_emails.csv', index=False)
print("\nSaved to data/labeled_emails.csv")
print("Total labeled emails:", len(df))
