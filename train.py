from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200)
model.fit(X, y)
joblib.dump(model, "model.pkl")
print("Model trained successfully.")

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200)
model.fit(X, y)
y_pred = model.predict(X)
y_true = y
acc = accuracy_score(y_pred, y_true)
