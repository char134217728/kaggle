import pandas as pd
import numpy as np
import pprint as pp
from sklearn import tree
from datetime import datetime

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

def fill_median(df, col):
    df[col] = df[col].fillna(df[col].dropna(how='any').median())

def fill_mode(df, col):
    df[col] = df[col].fillna(df[col].dropna(how='any').mode()[0])

def change_value(df, col, vl):
    for i in range(len(df)):
        df.at[i, col] = vl[df.at[i, col]]

fill_median(train, "Age")
fill_mode(train, "Embarked")
fill_median(test, "Fare")
fill_median(test, "Age")

embarked_hash = {
    "S":0,
    "C":1,
    "Q":2
}
sex_hash = {
    "male":0,
    "female":1
}
change_value(train, "Embarked", embarked_hash)
change_value(test, "Embarked", embarked_hash)
change_value(train, "Sex", sex_hash)
change_value(test, "Sex", sex_hash)


# 「train」の目的変数と説明変数の値を取得
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# 決定木の作成
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# 「test」の説明変数の値を取得
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# 「test」の説明変数を使って「my_tree_one」のモデルで予測
my_prediction = my_tree_one.predict(test_features)


# PassengerIdを取得
PassengerId = np.array(test["PassengerId"]).astype(int)

# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

# my_tree_one.csvとして書き出し
my_solution.to_csv("./csv/"+datetime.now().strftime("%Y%m%d%H%M%S")+".csv", index_label = ["PassengerId"])
