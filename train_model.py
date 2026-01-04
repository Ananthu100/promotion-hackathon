import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

df = pd.read_csv("data/train.csv")

X = df.drop("is_promoted", axis=1)
y = df["is_promoted"]

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5)

pipe = Pipeline([
    ("prep", preprocess),
    ("model", model)
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

pipe.fit(X_train, y_train)
preds = pipe.predict_proba(X_val)[:,1]

print("AUC:", roc_auc_score(y_val, preds))

pickle.dump(pipe, open("model/model.pkl", "wb"))
