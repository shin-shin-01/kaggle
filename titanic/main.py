import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


def preprocess(df):
    """前処理"""
    # 名前は分類に必要ないので削除
    df.drop(columns='Name', inplace=True)

    # Cabinは Nanが多いので一旦削除
    df.drop(columns='Cabin', inplace=True)

    # Ticketもバラバラなので一旦削除
    # len(df["Ticket"].unique()) => 681
    df.drop(columns='Ticket', inplace=True)

    # 男女をダミー変数化
    # df2['Sex'].unique() => male , female
    df['Sex'] = df['Sex'].apply(lambda sex: 1 if sex=="male" else 0)

    # タイタニック号に乗っている家族の数
    df["Families"] = df["SibSp"] + df["Parch"]
    # - 独身, １世帯, それ以上 に分けてみる
    df["Families"] = df["Families"].apply(lambda x: "s" if x == 0 else "m" if x < 3 else "l")

    # One-Hot-Encoding: Embarked
    df = pd.get_dummies(df)

    # Nanは中央値に変換
    df.fillna(df.median(), inplace=True)

    return df


def run_ml(X, y):
    # 標準化
    std_scl = StandardScaler()
    std_scl.fit(X)
    X = std_scl.transform(X)

    # 学習（全データを用いて学習）
    # - GridSearchで良いパタメータを探す & 交差検証
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],  'gamma' : [0.001, 0.01, 0.1, 1, 10, 100]}
    model = GridSearchCV(SVC(kernel='linear'), param_grid, cv=5)
    model.fit(X, y)

    return model


if __name__ == "__main__":
    # train.csv で学習
    df = pd.read_csv("./data/input/train.csv", index_col=0)
    df = preprocess(df)
    # - トレーニングデータを説明変数(X)と目的変数(y)に分割
    cols = ["Pclass", "Sex", "Fare", "Embarked_C", "Embarked_S", "Families_m", "Families_s"]
    model = run_ml(df[cols], df["Survived"])

    # test.csv で最終結果出力
    df_test = pd.read_csv("./data/input/test.csv", index_col=0)
    df_test = preprocess(df_test)
    
    # - データの整形
    df_submit = pd.DataFrame({
        'PassengerId': list(df_test.index),
        'Survived': model.predict(df_test[cols])
    })
    # - CSV出力
    df_submit.to_csv("./data/output/gender_submission.csv", index=False)
