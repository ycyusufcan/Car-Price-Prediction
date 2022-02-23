from sklearn.externals import joblib
from tahmin import preprocesing, FILE_PATH, MODEL_YOLU
from ayarlar import TEST_DATA_PATH, TRAIN_DATA_PATH
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import cross_validate, KFold


if __name__ == "__main__":
  df = pd.read_json(TRAIN_DATA_PATH)
  model = LinearRegression()
  fold = KFold(n_splits=20, shuffle=True)

  X = preprocesing(df)
  y = df["price"]

  df1 = pd.DataFrame(cross_validate(model, X, y, cv=fold, n_jobs=-1, return_train_score=True,
  scoring=['r2', 'neg_mean_squared_error']))
  df1.describe().to_json("tahmin_sonuclari")

  model.fit(X, y)
  joblib.dump(model, MODEL_YOLU)