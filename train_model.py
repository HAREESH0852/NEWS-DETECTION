import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

# Load dataset
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# Correct labeling
df_fake["class"] = 1  # FAKE = 1
df_true["class"] = 0  # REAL = 0

# Combine both
df = pd.concat([df_fake, df_true], axis=0).sample(frac=1).reset_index(drop=True)
df = df.drop(["title", "subject", "date"], axis=1)

# Clean function (light cleaning)
def wordopt(text):
    text = text.lower()
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)
    text = re.sub(r'[^\\w\\s]', '', text)
    text = re.sub(r'\\n', '', text)
    text = re.sub(r'\\w*\\d\\w*', '', text)
    return text

df["text"] = df["text"].apply(wordopt)

x = df["text"]
y = df["class"]

# Vectorize
vectorizer = TfidfVectorizer()
x_vectorized = vectorizer.fit_transform(x)

# ✅ Perform oversampling to balance dataset:
ros = RandomOverSampler(random_state=42)
x_resampled, y_resampled = ros.fit_resample(x_vectorized, y)

# Train model
model = LogisticRegression()
model.fit(x_resampled, y_resampled)

# Save model & vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model trained & saved successfully")
