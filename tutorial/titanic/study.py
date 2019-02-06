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

# 追加となった項目も含めて予測モデルその2で使う値を取り出す
features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values

# 決定木の作成とアーギュメントの設定
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)

# tsetから「その2」で使う項目の値を取り出す
test_features_2 = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# 「その2」の決定木を使って予測をしてCSVへ書き出す
my_prediction_tree_two = my_tree_two.predict(test_features_2)
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution_tree_two = pd.DataFrame(my_prediction_tree_two, PassengerId, columns = ["Survived"])

# my_tree_one.csvとして書き出し
my_solution_tree_two.to_csv("./ans/"+datetime.now().strftime("%Y%m%d%H%M%S")+".csv", index_label = ["PassengerId"])
