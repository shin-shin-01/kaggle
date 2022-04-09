import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

    # One-Hot-Encoding: Embarked
    df = pd.get_dummies(df)

    # Nanは中央値に変換
    df.fillna(df.median(), inplace=True)

    return df


def run_ml(df):
    """学習・テストデータでのスコア出力"""
    # トレーニングデータを説明変数(X)と目的変数(y)に分割
    X = df.drop(columns="Survived")
    y = df["Survived"]

    # 学習用データと検証用データに分割
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=None)

    # 標準化
    std_scl = StandardScaler()
    std_scl.fit(X_train)
    X_train = std_scl.transform(X_train)
    X_test = std_scl.transform(X_test)

    # 学習・テスト
    svm = SVC(kernel='linear', random_state=None, C=0.1)
    svm.fit(X_train, y_train)
    score = svm.score(X_test,y_test)
    print(score)

    return svm


if __name__ == "__main__":
    # train.csv で学習
    df = pd.read_csv("./data/input/train.csv", index_col=0)
    df = preprocess(df)
    svm = run_ml(df)

    # test.csv で最終結果出力
    df_test = pd.read_csv("./data/input/test.csv", index_col=0)
    df_test = preprocess(df_test)
    
    # - データの整形
    df_submit = pd.DataFrame({
        'PassengerId': list(df_test.index),
        'Survived': svm.predict(df_test)
    })
    # - CSV出力
    df_submit.to_csv("./data/output/gender_submission.csv", index=False)
