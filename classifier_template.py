# -*- coding: utf-8 -*-
import numpy as np
import data
import matplotlib.pylab as plt
import pdb

#-------------------
# クラスの定義始まり
class basic:
	visualPath = 'visualization'

	#------------------------------------
	# 学習データおよびモデルパラメータの初期化
	# x: 学習入力データ（入力ベクトルの次元数×データ数のnumpy.array）
	# t: one-hot学習カテゴリデータ（カテゴリ数×データ数のnumpy.array）
	# batchSize: 学習データのバッチサイズ（スカラー、0の場合は学習データサイズにセット）
	def __init__(self, x, t, batchSize=0):
		
		# デフォルト初期化
		self.init(x,t,batchSize)
		
	#------------------------------------

	#------------------------------------
	# デフォルトの初期化
	# x: 学習入力データ（入力ベクトルの次元数×データ数のnumpy.array）
	# t: one-hot学習カテゴリデータ（カテゴリ数×データ数のnumpy.array）
	# batchSize: 学習データのバッチサイズ（スカラー、0の場合は学習データサイズにセット）
	def init(self, x, t, batchSize=0):
		# 学習データ
		self.x = x
		self.t = t
		self.dNum = x.shape[1]	# 入力データ数
		
		# 損失の記録
		self.losses = np.array([])
		
		# 正解率の記録
		self.accuracies = np.array([])

		# ミニバッチの初期化
		if not batchSize:
			batchSize = self.dNum
			
		self.batchSize = batchSize
		self.batchCnt = 0
		
		# データインデックスの初期化
		self.randInd = np.random.permutation(self.dNum)
		self.validInd = np.random.permutation(self.dNum)[:(int)(self.dNum*0.1)]
	
	#------------------------------------

	#------------------------------------
	# 正解率の計算
	# x: 入力データ（入力ベクトルの次元数×データ数のnumpy.array）
	# t: one-hot学習カテゴリデータ（カテゴリ数×データ数のnumpy.array）
	def accuracy(self, x, t):
		dNum = x.shape[1]
		
		# 最大の事後確率をとるカテゴリ
		maxInd = np.argmax(self.predict(x),axis=0)

		# TR(True Positive)の数
		tpNum = np.sum([t[maxInd[i],i] for i in np.arange(dNum)])
		
		# 正解率=TP/データ数
		return tpNum/dNum
	#------------------------------------

	#------------------------------------
	# ミニバッチの取り出し
	def nextBatch(self,batchSize):

		sInd = batchSize * self.batchCnt
		eInd = sInd + batchSize

		batchX = self.x[:,self.randInd[sInd:eInd]]
		batchT = self.t[:,self.randInd[sInd:eInd]]
		
		if eInd+batchSize > self.dNum:
			self.batchCnt = 0
		else:
			self.batchCnt += 1

		return batchX, batchT
	#------------------------------------

# クラスの定義終わり
#-------------------
