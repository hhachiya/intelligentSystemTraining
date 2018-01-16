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
	def __init__(self,trainNum, testNum, dataType='1D', isNonlinear=False):
		
		# dataTypeによって1Dと2Dを切り替え
		self.dataType = dataType
		self.xRange = [-2, 8]
		self.prefix = dataType
		
		xRangeWidth = self.xRange[1] - self.xRange[0]
					
		if self.dataType == '1D':
			# 入力データ行列Xを作成
			self.xTrain = (np.random.rand(trainNum) * xRangeWidth + self.xRange[0])[np.newaxis]
			self.xTest =  (np.random.rand(testNum) * xRangeWidth + self.xRange[0])[np.newaxis]
			
		elif self.dataType == '2D':
			# 入力データ行列Xを作成
			self.xTrain = np.random.rand(2,trainNum) * xRangeWidth + self.xRange[0]
			self.xTest =  np.random.rand(2,testNum) * xRangeWidth + self.xRange[0]
			
		# ラベルベクトルyを作成
		if isNonlinear:
			self.yTrain = self.sampleNonLinearTarget(self.xTrain, noiseLvl=0.1)
			self.yTest = self.sampleNonLinearTarget(self.xTest, noiseLvl=0.1)		
		else:
			self.yTrain = self.sampleLinearTarget(self.xTrain, noiseLvl=0.1)
			self.yTest = self.sampleLinearTarget(self.xTest, noiseLvl=0.1)
		
	#------------------------------------

	#------------------------------------
	# 目標の線形関数
	# x: 入力データ（入力次元 x データ数）
	# noiseLvl: ノイズレベル（スカラー）
	def sampleLinearTarget(self,x, noiseLvl=0):
				
		# sin関数
		if self.dataType == '1D':
			y = np.sin(0.1*x)[0]+2
			xNum = x.shape[1]
			
		elif self.dataType == '2D':
			y = np.sin(0.1*x[0,:] + 0.1*x[1,:])+2
			xNum = x.shape[1]
		
		# ノイズの付加
		if noiseLvl:
			y += np.random.normal(0,noiseLvl,xNum)
		
		return y
	#------------------------------------

	#------------------------------------
	# 目標の線形関数
	# x: 入力データ（入力次元 x データ数）
	# noiseLvl: ノイズレベル（スカラー）
	def sampleNonLinearTarget(self,x, noiseLvl=0):
				
		# sin関数
		if self.dataType == '1D':
			y = np.sin(x)[0]+2
			xNum = x.shape[1]
			
		elif self.dataType == '2D':
			y = np.sin(x[0,:] + 0.1*x[1,:])+2
			xNum = x.shape[1]
		
		# ノイズの付加
		if noiseLvl:
			y += np.random.normal(0,noiseLvl,xNum)
		
		return y
	#------------------------------------
	
	#------------------------------------
	# データのプロット
	def plot(self,predict=[],isTrainPlot=True):
		if self.dataType == "1D":
			self.plot2D(predict,isTrainPlot)
		elif self.dataType == "2D":
			self.plot3D(predict)
	#------------------------------------

	#------------------------------------
	# 3次元データのプロット
	# predict: 予測結果（データ数）
	def plot3D(self,predict=[],isTrainPlot=True):

		# 3次元データの準備
		xTrain = self.xTrain
		xTest = self.xTest

		# 3次元プロット用のAxes3Dを利用
		fig = plt.figure()
		ax = Axes3D(fig)

		if isTrainPlot:
			# 学習データを描画
			ax.plot(xTrain[0,:], xTrain[1,:], self.yTrain, 'o', color="#FFA500", markeredgecolor='k', markersize=8)

		# 評価データを描画
		ax.plot(xTest[0,:], xTest[1,:], self.yTest, 's', color="#FFFF00", markeredgecolor='k', markersize=8)
		
		if len(predict):
			# 予測結果を描画
			ax.plot(xTest[0,:], xTest[1,:], predict, 'd', color="#FF0000", markeredgecolor='k', markersize=8)

		# plotラベルの設定
		ax.set_xlabel("x1", fontsize=14)
		ax.set_ylabel("x2", fontsize=14)
		ax.set_zlabel("y", fontsize=14)
		ax.tick_params(labelsize=14)
		
		if len(predict):
			if isTrainPlot:
				plt.legend(("Training Data","Test Data","Predict"))
			else:
				plt.legend(("Test Data","Predict"))
		else:
			if isTrainPlot:
				plt.legend(("Training Data","Test Data"))
			else:
				plt.legend(("Test Data"))

		# 表示範囲の設定
		ax.set_xlim(self.addMargin(self.xRange[0],"min"),self.addMargin(self.xRange[1],"max"))
		ax.set_ylim(self.addMargin(self.xRange[0],"min"),self.addMargin(self.xRange[1],"max"))
		ax.set_zlim(self.addMargin(np.min(self.yTest),"min"), self.addMargin(np.max(self.yTest),"max"))
		
		# 保存
		fullpath = os.path.join(self.visualPath,"{}_regressionData.png".format(self.prefix))
		plt.savefig(fullpath)
		          
		# 表示
		plt.show()
	#------------------------------------

	#------------------------------------
	# 2次元データのプロット
	# predict: 予測結果（データ数）
	def plot2D(self,predict=[],isTrainPlot=True):

		# 2次元データの準備
		xTrain = self.xTrain[0]
		xTest = self.xTest[0]

		if isTrainPlot:
			# 学習データを描画
			plt.plot(xTrain, self.yTrain, 'o', color="#FFA500", markeredgecolor='k', markersize=8)

		# 評価データを描画
		plt.plot(xTest, self.yTest, 's', color="#FFFF00", markeredgecolor='k', markersize=8)
		
		if len(predict):
			# 予測結果を描画
			plt.plot(xTest, predict, 'd', color="#FF0000", markeredgecolor='k', markersize=8)

		# plotラベルの設定
		plt.xlabel("x", fontsize=14)
		plt.ylabel("y", fontsize=14)
		plt.tick_params(labelsize=14)
		
		if len(predict):
			if isTrainPlot:
				plt.legend(("Training Data","Test Data","Predict"))
			else:
				plt.legend(("Test Data","Predict"))
		else:
			if isTrainPlot:
				plt.legend(("Training Data","Test Data"))
			else:
				plt.legend(["Test Data"])

		# 表示範囲の設定
		plt.xlim(self.addMargin(self.xRange[0],"min"),self.addMargin(self.xRange[1],"max"))
		plt.ylim(self.addMargin(np.min(self.yTest),"min"), self.addMargin(np.max(self.yTest),"max"))
		
		# 保存
		fullpath = os.path.join(self.visualPath,"{}_regressionData.png".format(self.prefix))
		plt.savefig(fullpath)
		          
		# 表示
		plt.show()
	#------------------------------------
	
	#------------------------------------
	# グラフの余白の計算
	# x: 余白を計算したい数値（スカラー）
	def addMargin(self,x,mode="min",margin=0.5):
		limit = [x - margin if mode=="min" else x + margin][0]
			
		return limit
	#------------------------------------
#------------------------------------