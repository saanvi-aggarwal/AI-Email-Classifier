# 📧 AI Email Priority Classifier

An AI-powered tool that automatically classifies emails as High, Medium, or Low priority using Machine Learning.

## 🔗 Links
- **Live App**: [Try it here](https://ai-email-classifier-f95aplsphch9npsbbetctl.streamlit.app/)
- **GitHub**: [Repository](https://github.com/saanvi-aggarwal/AI-Email-Classifier)

---

## 🚩 Problem It Solves
People waste hours daily triaging emails manually. 
This project automates that using NLP and ML.

---

## 🧠 How It Works
1. Raw emails are cleaned and preprocessed
2. TF-IDF converts text into numbers
3. Logistic Regression model classifies priority
4. Streamlit displays the result instantly

---

## 📊 Model Performance
| Class | F1 Score |
|-------|----------|
| High | 0.64 |
| Low | 0.72 |
| Medium | 0.89 |
| **Overall Accuracy** | **81%** |

---

## 🛠️ Tech Stack
Python, scikit-learn, Streamlit, Pandas, Joblib

---

## 🚀 Run Locally
```
git clone https://github.com/saanvi-aggarwal/AI-Email-Classifier
cd ai-email-classifier
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run src/app.py
```
