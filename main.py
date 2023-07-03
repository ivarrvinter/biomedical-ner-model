import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv('hetionet-v1.0-nodes.tsv', delimiter='\t')

df.iloc[:, 2] = df.iloc[:, 2].str.lower()

X = df.iloc[:, 1]
y = df.iloc[:, 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vec = TfidfVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

model = LogisticRegression(C=14, max_iter=470)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)
