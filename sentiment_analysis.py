import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
data = pd.read_csv('./Tweets.csv')
data = data.dropna()
data = data.reset_index(drop=True)
texts = np.array(data["text"])
keywords = np.array(data["selected_text"])
labels = np.array(data["sentiment"])
# print(texts," ",keywords," ",labels)

vectorizer = TfidfVectorizer(stop_words='english')
texts = vectorizer.fit_transform(texts).toarray()

print(texts)

x_train, x_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42)
print(type(x_test))
print(data["text"].dtype)


le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

RF_Model = RandomForestClassifier(n_estimators=200, random_state=50, n_jobs=-1)
RF_Model.fit(x_train, y_train)
print("Model training finished")


test_prediction = RF_Model.predict(x_test)
# test_features = np.expand_dims(test_features, axis=0)
# test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))
# Inverse le transform to get original label back.
test_prediction = le.inverse_transform(test_prediction)
y_test = le.inverse_transform(y_test)

# Print overall accuracy
print("Accuracy = ", metrics.accuracy_score(y_test, test_prediction))
print("Classification report: \n")
print(metrics.classification_report(test_prediction, y_test))
