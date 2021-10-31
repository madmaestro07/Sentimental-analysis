import pandas as pd  # Import pandas library
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score

df = pd.read_csv('Reviews.csv') #Load movie reviews dataset

# print(df.head())

# stopwords = set(stopwords)
# print(stopwords)


df = df[df['Score'] != 3]

df['sentiment'] = df['Score'].apply(lambda rating : +1 if rating > 3 else -1)

positive = df[df['sentiment'] == 1]
negative = df[df['sentiment'] == -1]


def remove_punctuation(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"'))
    return final


df = df.dropna(subset=['Summary','Text'])


df['Text'] = df['Text'].apply(remove_punctuation)
df['Summary'] = df['Summary'].apply(remove_punctuation)


new_ds = df[['Summary','sentiment']]
print(new_ds.head())


index = df.index
df['random_number'] = np.random.randn(len(index))
train = df[df['random_number'] <= 0.8]
test = df[df['random_number'] > 0.8]


vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['Summary'])
test_matrix = vectorizer.transform(test['Summary'])


lr = LogisticRegression()

X_train = train_matrix
X_test = test_matrix
y_train = train['sentiment']
y_test = test['sentiment']

lr.fit(X_train,y_train)

predictions = lr.predict(X_test)

new = np.asarray(y_test)
print(confusion_matrix(predictions,y_test))

print(accuracy_score(y_test,predictions))