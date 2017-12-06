import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import re
import gzip
import pdb

#------------------------------------
# 人工データのクラス
class artificial:
	visualPath = 'visualization'
	
	#------------------------------------
	# コンストラクタ（データを生成）
	def __init__(self,trainNum, testNum, dataType='linear'):
		
		
		if dataType == 'linear':
			self.xRange = [-2, 8]
			self.prefix = dataType
			

			# 入力データ行列Xを作成
			xRangeWidth = self.xRange[1] - self.xRange[0]

			self.xTrain = np.random.rand(trainNum) * xRangeWidth + self.xRange[0]
			self.xTest =  np.random.rand(testNum) * xRangeWidth + self.xRange[0]
			self.xTruth = (np.arange(xRangeWidth*10) + self.xRange[0]*10)/10
			
			# ラベルベクトルyを作成
			self.yTrain =  self.sampleLinearTarget(self.xTrain, noiseLvl=0.1)
			self.yTest = self.sampleLinearTarget(self.xTest, noiseLvl=0.1)
			self.yTruth = self.sampleLinearTarget(self.xTruth, noiseLvl=0)
		
	#------------------------------------

	#------------------------------------
	# 目標の線形関数
	def sampleLinearTarget(self,x, noiseLvl=0):
		
		# sin関数
		y = np.sin(0.1*x)+2

		# ノイズの付加
		if noiseLvl:
			y += np.random.normal(0,noiseLvl,len(x))
		
		return y

	#------------------------------------
	# データのプロット
	def plot(self):
	
		# 学習データを描画
		plt.plot(self.xTrain, self.yTrain, 'o', color="#FFA500", markeredgecolor='k', markersize=8)

		# 評価データを描画
		plt.plot(self.xTest, self.yTest, 's', color="#FFFF00", markeredgecolor='k', markersize=8)
	
		# 真の関数の描画
		plt.plot(self.xTruth, self.yTruth, '-', color="#FF00FF")

		# plotラベルの設定
		plt.title("Data", fontsize=14)
		plt.xlabel("x", fontsize=14)
		plt.ylabel("y", fontsize=14)
		plt.tick_params(labelsize=14)
		plt.legend(("Training Data","Test Data","True function"))

		# 表示範囲の設定
		plt.xlim(self.xRange[0],self.xRange[1])
		plt.ylim(np.min(self.yTruth) - np.min(self.yTruth)*0.1, np.max(self.yTruth) + np.max(self.yTruth)*0.1)
		
		# 保存
		fullpath = os.path.join(self.visualPath,"{}_regressionData.png".format(self.prefix))
		plt.savefig(fullpath)
		
		# 表示
		plt.show()
