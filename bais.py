import numpy as np
import scipy as sp
import sklearn


class NaiveBayes1(object):
    """
    単純ベイズクラス1
    親クラスはobjectとする
    """

    def __init__(self):
        """
        コンストラクタ
        """
        self.pY_ = None
        self.pXgY_ = None
        pass

    def fit(self, X, y):
        """
        学習を行う関数
        :param X: 訓練データ特徴ベクトル集合
        :param y: クラスラベル集合
        :return:
        """
        # 訓練事例数(Samples)と特徴数(features)を引数から取得
        n_samples = X.shape[0]
        n_features = X.shape[1]

        print('fit start: ns:' +str(n_samples) + ' nf:'+str(n_features))

        # この単純ベイズではクラス、特徴ともに2値(0か1)とするため、それを定義する
        n_classes = 2
        n_fvalues = 2

        # 特徴の事例数とクラスラベルの事例数は一致している必要があるので、異なる場合はエラー
        if n_samples != len(y):
            raise ValueError('Missmached number of samples')

        # クラス分布の学習を行う
        # おかわりもいいぞ！

        # 各クラスごとに事例数をカウントする
        # クラス数次元の整数型ベクトルを生成し、クラスラベルごとの要素に加算していく
        nY = np.zeros(n_classes, dtype=int)
        for i in range(n_samples):
            nY[y[i]] += 1
        print('count sumple in class :'+str(nY))

        # モデルパラメータ pY_ を計算する。後で書き換えるので、empty()で初期化。
        # 割り算結果を実数でとるためにfloat型へ変換
        self.pY_ = np.empty(n_classes, dtype=float)
        for i in range(n_classes):
            self.pY_[i] = nY[i] / float(n_samples)

        ### 特徴の分布を学習する ###
        # 3次元ベクトル[特徴数][特徴値数][クラスラベルの数]を作成する
        # 各特徴値、各クラスごとに事例数を数える
        nXY = np.zeros((n_features, n_fvalues, n_classes), dtype=int)
        print('start fitting')
        for i in range(n_samples):
            for j in range(n_features):
                # X[i,j]は配列スライス
                nXY[j, X[i, j], y[i]] += 1
        print(nXY)

        # モデルパラメータ pXgY_を計算
        # 特徴値、クラス毎の事例数をクラス毎の事例数で割る
        self.pXgY_ = np.empty((n_features, n_fvalues, n_classes), dtype=float)
        for j in range(n_features):
            for xi in range(n_fvalues):
                for yi in range(n_classes):
                    self.pXgY_[j, xi, yi] = nXY[j, xi, yi] / float(nY[yi])

        print('fit end param')
        print(self.pXgY_)
        pass


    def predict(self, X):
        """
        クラスの予測を行う関数
        :param x: 未知の特徴ベクトル集合
        :return:
        """
        # 訓練事例数(Samples)と特徴数(features)を引数から取得
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # この単純ベイズではクラス、特徴ともに2値とするため、それを定義する
        n_classes = 2
        n_fvalues = 2

        # 計算結果を格納する領域
        y = np.empty(n_samples, dtype=int)

        # 最初の次元を走査するイテレータを利用し、行の内容をxiにコピーする
        for i, xi in enumerate(X):
            # Numpyのユニバーサル関数を利用してクラスレベルを予測する(配列を渡すと要素全体に関数を適用してくれる)
            # log Pr[y] + Σj logPr[x j|y]の数式を処理に置き換える

            # まずは log Pr[y] を求める。 パラメータのpY_を対数関数(log)をかけたもの
            logpXY = np.log(self.pY_)

            # 次にlogPr[x j|y] を計算する
            # xiに対し、yが0と1それぞれの場合の対数の同時確立　log(Pr)(x|y)を計算し、大きいほうをクラスレベルとする
            # pXgY[j]の要素を確率関数、xi[j]を未知特徴ベクトル値、(0|1)を:を用いて両方、を対数関数にかけた結果を加算する
            for j in range(n_features):
                # ここで使用している + も、ユニバーサル関数と同じく、配列の各要素に働く
                logpXY = logpXY + np.log(self.pXgY_[j, xi[j], :])

            # 上記の2つの計算をもっと頭よくした式
            # axisは和をとる次元の軸を指定する。2次元配列の場合、0なら列和、1なら行和になる
            # logpXY = np.log(self.pY_) + \
            #         np.sum(np.log(self.pXgY_[np.arange(n_features), xi, :]),
            #                axis=0)


            # 上で計算できたlogpXYのうち最も大きな値をとる要素が予測クラスになる
            # np.argmax()で行列中の最大をとれる
            y[i] = np.argmax(logpXY)

            pass
        return y


