# ============================================================
#   SPAM DETECTION USING NLP
#   Author  : Faiz Ahmed Khan
#   GitHub  : github.com/shadowsming
#   Tools   : Python, Scikit-learn, Pandas, TF-IDF, NLP
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── STEP 1: LOAD DATASET ─────────────────────────────────
print("=" * 50)
print("SPAM DETECTION - NLP PROJECT")
print("=" * 50)

df = pd.read_csv('spam_dataset.csv')

print(f"\n✅ Dataset Loaded")
print(f"   Total Messages : {len(df)}")
print(f"   Spam Messages  : {len(df[df['label'] == 'spam'])}")
print(f"   Ham Messages   : {len(df[df['label'] == 'ham'])}")
print(f"\nSample Data:")
print(df.head())


# ── STEP 2: CLEAN THE TEXT ───────────────────────────────
def clean_text(text):
    """
    Clean raw text:
    - Convert to lowercase
    - Remove special characters and numbers
    - Remove extra whitespace
    """
    text = text.lower()                        # lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)   # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

df['clean_message'] = df['message'].apply(clean_text)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

print("\n✅ Text Cleaned")
print(df[['message', 'clean_message', 'label']].head(3))


# ── STEP 3: SPLIT DATA ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_message'],
    df['label_num'],
    test_size=0.2,       # 80% train, 20% test
    random_state=42
)

print(f"\n✅ Data Split")
print(f"   Training samples : {len(X_train)}")
print(f"   Testing samples  : {len(X_test)}")


# ── STEP 4: TF-IDF VECTORIZATION ─────────────────────────
# Convert text into numbers that the ML model can understand
tfidf = TfidfVectorizer(max_features=500, stop_words='english')
X_train_tf = tfidf.fit_transform(X_train)
X_test_tf  = tfidf.transform(X_test)

print(f"\n✅ TF-IDF Vectorization Done")
print(f"   Feature Matrix Shape: {X_train_tf.shape}")


# ── STEP 5: TRAIN MODELS ─────────────────────────────────
# Model 1: Naive Bayes (best for text classification)
nb_model = MultinomialNB()
nb_model.fit(X_train_tf, y_train)
nb_preds  = nb_model.predict(X_test_tf)
nb_acc    = accuracy_score(y_test, nb_preds)

# Model 2: Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tf, y_train)
lr_preds  = lr_model.predict(X_test_tf)
lr_acc    = accuracy_score(y_test, lr_preds)

print(f"\n✅ Models Trained")
print(f"   Naive Bayes Accuracy        : {nb_acc:.2%}")
print(f"   Logistic Regression Accuracy: {lr_acc:.2%}")


# ── STEP 6: DETAILED REPORT ──────────────────────────────
print("\n── Naive Bayes Classification Report ──")
print(classification_report(y_test, nb_preds, target_names=['Ham', 'Spam']))

print("── Logistic Regression Classification Report ──")
print(classification_report(y_test, lr_preds, target_names=['Ham', 'Spam']))


# ── STEP 7: VISUALIZATIONS ───────────────────────────────
# Plot 1: Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Spam Detection — Model Performance', fontsize=14, fontweight='bold')

cm1 = confusion_matrix(y_test, nb_preds)
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
axes[0].set_title(f'Naive Bayes (Accuracy: {nb_acc:.2%})')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

cm2 = confusion_matrix(y_test, lr_preds)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
axes[1].set_title(f'Logistic Regression (Accuracy: {lr_acc:.2%})')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n✅ confusion_matrix.png saved")

# Plot 2: Dataset Distribution
fig2, ax = plt.subplots(figsize=(6, 4))
counts = df['label'].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=['#2ecc71', '#e74c3c'], width=0.5, edgecolor='white')
ax.set_title('Dataset Distribution', fontsize=13, fontweight='bold')
ax.set_xlabel('Label')
ax.set_ylabel('Count')
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1, str(val), ha='center', fontweight='bold')
ax.set_facecolor('#f8f9fa')
fig2.patch.set_facecolor('#f8f9fa')
plt.tight_layout()
plt.savefig('distribution.png', dpi=120, bbox_inches='tight')
plt.close()
print("✅ distribution.png saved")


# ── STEP 8: TEST WITH YOUR OWN MESSAGES ──────────────────
def predict_spam(message):
    """Predict if a message is spam or ham."""
    cleaned = clean_text(message)
    vectorized = tfidf.transform([cleaned])
    prediction = nb_model.predict(vectorized)[0]
    return "🚨 SPAM" if prediction == 1 else "✅ HAM (Not Spam)"

print("\n── Test With Custom Messages ──")
test_messages = [
    "Congratulations! You have won a 1000 cash prize. Call now!",
    "Hey, are you coming to dinner tonight?",
    "FREE entry! Win FA Cup final tickets. Text WIN to 87575 now",
    "Can you send me the notes from today's class?",
    "URGENT: Your account will be suspended. Verify now!",
]

for msg in test_messages:
    result = predict_spam(msg)
    print(f"   {result} → \"{msg[:55]}...\"" if len(msg) > 55 else f"   {result} → \"{msg}\"")

print("\n" + "=" * 50)
print("Bhai free recharge mil raha hai click kar"
"Kal college mein practical hai aana")
print("=" * 50)
