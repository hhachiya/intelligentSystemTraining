# -*- coding: utf-8 -*-
import numpy as np
import data
import matplotlib.pylab as plt
import classifier_template as classifier
import pdb

#-------------------
# クラスの定義始まり
class logisticRegression(classifier.basic):
	#------------------------------------
	# 1) 学習データおよびモデルパラメータの初期化
	# x: 学習入力データ（入力ベクトルの次元数×データ数のnumpy.array）
	# t: one-hot学習カテゴリデータ（カテゴリ数×データ数のnumpy.array）
	# batchSize: 学習データのバッチサイズ（スカラー、0の場合は学習データサイズにセット）
	def __init__(self, x, t, batchSize=0):
		# デフォルト初期化
		self.init(x,t,batchSize)
	
		# モデルパラメータをランダムに初期化
		xDim = x.shape[0]
		tDim = t.shape[0]
		self.W = np.random.normal(0.0, pow(xDim, -0.5), (xDim, tDim))
		self.b = np.random.normal(0.0, pow(tDim, -0.5), (tDim, 1))
	#------------------------------------

	#------------------------------------
	# 2) ソフトマックスの計算
	# x: カテゴリ数×データ数のnumpy.array
	def softmax(self,x):
		# x-max(x):expのinfを回避するため
		e = np.exp(x-np.max(x))
		return e/np.sum(e,axis=0)
	#------------------------------------
	
	#------------------------------------
	# 3) 最急降下法によるパラメータの更新
	# alpha: 学習率（スカラー）
	# printEval: 評価値の表示指示（真偽値）
	def update(self, alpha=0.1,printEval=True):
	
		# 次のバッチ
		x, t = self.nextBatch(self.batchSize)

		# データ数
		dNum = x.shape[1]
		
		# Wの更新
		predict_minus_t = self.predict(x) - t
		self.W -= alpha * 0 # 【wの勾配の計算】
		
		# bの更新
		self.b -= alpha * 0 # 【bの勾配の計算】	

		# 交差エントロピーとAccuracyを標準出力
		if printEval:
			# 交差エントロピーの記録
			self.losses = np.append(self.losses, self.loss(self.x[:,self.validInd],self.t[:,self.validInd]))

			# 正解率エントロピーの記録
			self.accuracies = np.append(self.accuracies, self.accuracy(self.x[:,self.validInd],self.t[:,self.validInd]))
		
			print("loss:{0:02.3f}, accuracy:{1:02.3f}".format(self.losses[-1],self.accuracies[-1]))
	#------------------------------------

	#------------------------------------
	# 4) 交差エントロピーの計算
	# x: 入力データ（入力ベクトルの次元数×データ数のnumpy.array）
	# t: one-hot学習カテゴリデータ（カテゴリ数×データ数のnumpy.array）
	def loss(self, x,t):
		crossEntropy =  0		#【交差エントロピーの計算】
		return crossEntropy
	#------------------------------------

	#------------------------------------
	# 5) 事後確率の計算
	# x: 入力データ（入力ベクトルの次元数×データ数のnumpy.array）
	def predict(self, x):
		return self.softmax(np.matmul(self.W.T,x) + self.b)
	#------------------------------------
	
# クラスの定義終わり
#-------------------

#-------------------
# メインの始まり
if __name__ == "__main__":
	# 1）人工データの生成（簡単な場合）
	myData = data.artificial(300,150,mean1=[1,2],mean2=[-2,-1],mean3=[2,-2],cov=[[1,-0.8],[-0.8,1]])

	# 1) 人工データの生成（難しい場合）
	#myData = data.artificial(300,150,mean1=[1,2],mean2=[-2,-1],mean3=[4,-2],mean3multi=[-2,4],cov=[[1,0],[0,1]])

	# 2）ロジスティック回帰（2階層のニューラルネットワーク）モデルの作成
	classifier = logisticRegression(myData.xTrain, myData.tTrain)

	# 3）学習前の事後確率と学習データの描画
	myData.plotClassifier(classifier,"train",prefix="posterior_before")

	# 4）モデルの学習
	Nite = 1000  # 更新回数
	learningRate = 0.1  # 学習率
	decayRate = 0.99  # 減衰率
	for ite in np.arange(Nite):
		print("Training ite:{} ".format(ite+1),end='')
		classifier.update(alpha=learningRate)

		# 5）更新幅の減衰
		learningRate *= decayRate

	# 6）評価
	loss = classifier.loss(myData.xTest,myData.tTest)
	accuracy = classifier.accuracy(myData.xTest,myData.tTest)
	print("Test loss:{}, accuracy:{}".format(loss,accuracy))
	
	# 7）学習した事後確率と学習データの描画
	myData.plotClassifier(classifier,"train",prefix="posterior_after")
	
#メインの終わり
#-------------------
