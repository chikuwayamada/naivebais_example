from bais import NaiveBayes1
import numpy as np
# メイン処理テスト

# ファイルを読み込み(自動的にtsvを解釈してnparrayにしてくれる
data = np.genfromtxt('vote_filled.tsv', dtype=int)

# データは最終列がクラスラベル(y)、それ以外が特徴量(x)となっている。
x = data[:, :-1]
y = data[:, -1]

clr = NaiveBayes1()

# 学習を実行
clr.fit(x,y)

# クラスの予測を行う
predict_y = clr.predict(x[:10,:])
for i in range(10):
    print(i, y[i], predict_y)