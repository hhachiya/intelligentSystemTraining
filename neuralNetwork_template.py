# -*- coding: utf-8 -*-

import numpy as np
import data
import matplotlib.pylab as plt
import logisticRegression_template as lr
import pdb

#-------------------
# クラスの定義始まり
class neuralNetwork(lr.logisticRegression):
	#------------------------------------
	# 1) 学習データおよびモデルパラメータの初期化
	# x: 学習入力データ（入力ベクトルの次元数×データ数のnumpy.array）
	# t: one-hot学習カテゴリデータ（カテゴリ数×データ数のnumpy.array）
	# hDim: 中間層のノード数hDim（スカラー）
	# batchSize: 学習データのバッチサイズ（スカラー、0の場合は学習データサイズにセット）
	def __init__(self, x, t, hDim=20, batchSize=0):
		# デフォルトの初期化
		self.init(x,t,batchSize)

		# モデルパラメータをランダムに初期化
		xDim = x.shape[0]	# 入力データの次元
		hDim = hDim			# 隠れ層のノード数
		tDim = t.shape[0]	# カテゴリ数
		
		self.W1 = np.random.normal(0.0, pow(hDim, -0.5), (xDim + 1, hDim))
		self.W2 = np.random.normal(0.0, pow(tDim, -0.5), (hDim + 1, tDim))

	#------------------------------------

	#------------------------------------
	# 3) 最急降下法によるパラメータの更新
	# alpha: 学習率（スカラー）
	# printLoss: 評価値の表示指示（真偽値）
	def update(self, alpha=0.1,printEval=True):

		# 次のバッチ
		x, t = self.nextBatch(self.batchSize)
		
		# データ数
		dNum = x.shape[1]
		
		# 中間層の計算
		h = self.hidden(x)
		
		# 事後確率の予測と真値の差分
		predict = self.predict(x,h)
		predict_error =  predict - t
		
		# self.W1とW2の更新
		#【self.W1の更新】
		#【self.W2の更新】

		# 交差エントロピーとAccuracyを標準出力
		if printEval:
			# 交差エントロピーの記録
			self.losses = np.append(self.losses, self.loss(self.x[:,self.validInd],self.t[:,self.validInd]))

			# 正解率エントロピーの記録
			self.accuracies = np.append(self.accuracies, self.accuracy(self.x[:,self.validInd],self.t[:,self.validInd]))
		
			print("loss:{0:02.3f}, accuracy:{1:02.3f}".format(self.losses[-1],self.accuracies[-1]))
	#------------------------------------
	
	#------------------------------------
	# 5) 事後確率の計算
	# x: 入力データ（入力ベクトルの次元数×データ数のnumpy.array）
	# h: 中間層のデータh（中間層のノード数×データ数のnumpy.array）
	def predict(self, x, h = []):
		if not len(h):
			h = self.hidden(x)
		return self.softmax(np.matmul(self.W2[:-1].T, h) + self.W2[-1][np.newaxis].T)
	#------------------------------------
	
	#------------------------------------
	# 6) シグモイドの計算
	# x: 入力データ（入力ベクトルの次元数×データ数のnumpy.array）
	def sigmoid(self,x):
		sigmoid = x		#【シグモイド関数の計算】
		return sigmoid
	#------------------------------------

	#------------------------------------
	# 7) 中間層
	# x: 入力データ（入力ベクトルの次元数×データ数のnumpy.array）
	def hidden(self, x):
		h = self.sigmoid(np.matmul(self.W1[:-1].T, x) + self.W1[-1][np.newaxis].T)
		return h
	#------------------------------------
	
# クラスの定義終わり
#-------------------

#-------------------
# メインの始まり
if __name__ == "__main__":
	# 1) 人工データの生成（簡単な場合）
	#myData = data.artificial(300,150,mean1=[1,2],mean2=[-2,-1],mean3=[2,-2],cov=[[1,-0.8],[-0.8,1]])
	
	# 1) 人工データの生成（難しい場合）
	myData = data.artificial(300,150,mean1=[1,2],mean2=[-2,-1],mean3=[4,-2],mean3multi=[-2,4],cov=[[1,0],[0,1]])

	# 2) 3階層のニューラルネットワークモデルの作成
	classifier = neuralNetwork(myData.xTrain, myData.tTrain,hDim=20)

	# 3) 学習前の事後確率と学習データの描画
	myData.plotClassifier(classifier,"train",prefix="posterior_NN_before")

	# 4) モデルの学習
	Nite = 1000  # 更新回数
	learningRate = 0.01  # 学習率
	decayRate = 0.99999  # 減衰率
	for ite in np.arange(Nite):
		print("Training ite:{} ".format(ite+1),end='')
		classifier.update(alpha=learningRate)
		
		# 5）更新幅の減衰
		learningRate *= decayRate

	# 6) 評価
	loss = classifier.loss(myData.xTest,myData.tTest)
	accuracy = classifier.accuracy(myData.xTest,myData.tTest)
	print("Test loss:{}, accuracy:{}".format(loss,accuracy))
	
	# 7) 学習した事後確率と学習データの描画
	myData.plotClassifier(classifier,"train",prefix="posterior_NN_after")
	myData.plotClassifier(classifier,"test",prefix="posterior_NN_after_test")
	
#メインの終わり
#-------------------
