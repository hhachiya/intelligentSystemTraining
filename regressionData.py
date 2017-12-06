import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import re
import gzip
import pdb

#------------------------------------
# 人工データのクラス
class artificial:
	visualPath = 'visualization'
	
	#------------------------------------
	# コンストラクタ（データを生成）
	# trainNum: 学習データ数（スカラー）
	# testNum: 評価データ数（スカラー）
	# dataType: 入力次元（文字列:'1D', '2D')
	def __init__(self,trainNum, testNum, dataType='1D'):
		
		# dataTypeによって1Dと2Dを切り替え
		self.dataType = dataType
		self.xRange = [-2, 8]
		self.prefix = dataType
		
		xRangeWidth = self.xRange[1] - self.xRange[0]
					
		if self.dataType == '1D':
			# 入力データ行列Xを作成
			self.xTrain = np.random.rand(trainNum) * xRangeWidth + self.xRange[0]
			self.xTest =  np.random.rand(testNum) * xRangeWidth + self.xRange[0]
			
		elif self.dataType == '2D':
			# 入力データ行列Xを作成
			self.xTrain = np.random.rand(2,trainNum) * xRangeWidth + self.xRange[0]
			self.xTest =  np.random.rand(2,testNum) * xRangeWidth + self.xRange[0]
			
		# ラベルベクトルyを作成
		self.yTrain =  self.sampleLinearTarget(self.xTrain, noiseLvl=0.1)
		self.yTest = self.sampleLinearTarget(self.xTest, noiseLvl=0.1)
		
	#------------------------------------

	#------------------------------------
	# 目標の線形関数
	# x: 入力データ（入力次元 x データ数）
	# noiseLvl: ノイズレベル（スカラー）
	def sampleLinearTarget(self,x, noiseLvl=0):
		
		# sin関数
		if self.dataType == '1D':
			y = np.sin(0.1*x)+2
			xNum = len(x)
			
		elif self.dataType == '2D':
			y = np.sin(0.1*x[0,:] + 0.1*x[1,:])+2
			xNum = x.shape[1]
		
		# ノイズの付加
		if noiseLvl:
			y += np.random.normal(0,noiseLvl,xNum)
		
		return y

	#------------------------------------
	# データのプロット
	def plot(self):

		# 3次元データの準備
		xTrain = self.xTrain
		xTest = self.xTest
		
		if self.dataType == '1D':
			xTrainNum = len(xTrain)
			xTestNum = len(xTest)
			xTrain = np.append(xTrain[np.newaxis], np.zeros([1,xTrainNum]),axis=0)
			xTest =  np.append(xTest[np.newaxis], np.zeros([1,xTestNum]),axis=0)
			

		fig = plt.figure()
		ax = Axes3D(fig)

		# 学習データを描画
		ax.plot(xTrain[0,:], xTrain[1,:], self.yTrain, 'o', color="#FFA500", markeredgecolor='k', markersize=8)

		# 評価データを描画
		ax.plot(xTest[0,:], xTest[1,:], self.yTest, 's', color="#FFFF00", markeredgecolor='k', markersize=8)

		# plotラベルの設定
		ax.set_xlabel("x1", fontsize=14)
		ax.set_ylabel("x2", fontsize=14)
		ax.set_zlabel("y", fontsize=14)
		ax.tick_params(labelsize=14)
		ax.legend(("Training Data","Test Data"))

		# 表示範囲の設定
		ax.set_xlim(self.xRange[0],self.xRange[1])
		ax.set_ylim(self.xRange[0],self.xRange[1])
		ax.set_zlim(np.min(self.yTest) - np.min(self.yTest)*0.1, np.max(self.yTest) + np.max(self.yTest)*0.1)
		
		# 保存
		fullpath = os.path.join(self.visualPath,"{}_regressionData.png".format(self.prefix))
		plt.savefig(fullpath)
		
		# 表示
		plt.show()
	#------------------------------------
#------------------------------------