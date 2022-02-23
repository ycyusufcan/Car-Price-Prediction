import json
import pandas as pd
from sklearn.externals import joblib
from ayarlar import TEST_DATA_PATH

MODEL_YOLU = "tahminci.joblib"

with open("numeric_Columns_in_list.json", "r") as f:
  NUMERIC = json.load(f)

with open("str_get_dummies_Columns_in_list.json", "r") as f:
  STR_GET_DUMMIES = json.load(f)

with open("get_dummies_Columns_in_list.json", "r") as f:
  GET_DUMMIES = json.load(f)

with open("All_Column_Names.json", "r") as f:
  ALL_COLUMNS = json.load(f)

FILE_PATH = TEST_DATA_PATH

def converter(data):
  data[NUMERIC] = data[NUMERIC].astype(float)
  return data[NUMERIC]

def str_get_dummies(data):
  liste = []
  for i in STR_GET_DUMMIES:
    liste.append(data[i].str.get_dummies(","))
  return pd.concat(liste, axis=1)

def get_dummies_(data):
  return pd.get_dummies(data[GET_DUMMIES])

def merge_all(*dataframes):
  return pd.concat(dataframes, axis=1)

def column_injection(data):
  return data.reindex(ALL_COLUMNS, axis=1).fillna(0)

def preprocesing(data):
  part1 = converter(data)
  part2 = str_get_dummies(data)
  part3 = get_dummies_(data)
  merged = merge_all(part1, part2, part3)
  X = column_injection(merged)
  return X


if __name__ == "__main__":
  df = pd.read_json(FILE_PATH)
  model = joblib.load(MODEL_YOLU)
  X = preprocesing(df)
  y_pred = model.predict(X)
  pd.Series(y_pred).to_csv("tahminler")