# -*- coding: utf-8 -*-
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
	def __init__(self,trainNum, testNum, mean1=[1,2],mean2=[-2,-1],mean3=[2,-2],mean3multi=[],cov=[[1,-0.8],[-0.8,1]],noiseMean=[]):
		# 各カテゴリのデータのリセット
		self.cat1_train = []
		self.cat2_train = []
		self.cat3_train = []
		self.cat1_test = []
		self.cat2_test = []
		self.cat3_test = []
	
		# 各カテゴリのデータを生成
		self.cat1_train.extend(np.random.multivariate_normal(mean1, cov, (int)(trainNum/3)))
		self.cat2_train.extend(np.random.multivariate_normal(mean2, cov, (int)(trainNum/3)))
		self.cat1_test.extend(np.random.multivariate_normal(mean1,  cov, (int)(testNum/3)))
		self.cat2_test.extend(np.random.multivariate_normal(mean2,  cov, (int)(testNum/3)))
		
		# mean3multiが設定されている場合は、カテゴリ3は多峰性を持つ
		if len(mean3multi):
			self.cat3_train.extend(np.random.multivariate_normal(mean3, cov, (int)(trainNum/6)))
			self.cat3_test.extend(np.random.multivariate_normal(mean3,  cov, (int)(testNum/6)))
			self.cat3_train.extend(np.random.multivariate_normal(mean3multi, cov, (int)(trainNum/6)))
			self.cat3_test.extend(np.random.multivariate_normal(mean3multi,  cov, (int)(testNum/6)))
		else:
			self.cat3_train.extend(np.random.multivariate_normal(mean3, cov, (int)(trainNum/3)))
			self.cat3_test.extend(np.random.multivariate_normal(mean3,  cov, (int)(testNum/3)))
		
		# ノイズデータの付与
		if len(noiseMean):
			self.cat1_train.extend(np.random.multivariate_normal(noiseMean,np.array(cov)/10,(int)(trainNum/100)))

		# 入力データ行列Xを作成
		self.xTrain = np.vstack((self.cat1_train, self.cat2_train, self.cat3_train)).T
		self.xTest = np.vstack((self.cat1_test, self.cat2_test, self.cat3_test)).T
	
		# ラベルベクトルyを作成
		self.tTrain = []
		self.tTest = []
		
		# 学習データのラベル（one-hotベクトル）を作成
		self.yTrain = np.hstack([np.zeros(len(self.cat1_train)), np.ones(len(self.cat2_train)), 2*np.ones(len(self.cat3_train))])
		[self.tTrain.append(np.array([1,0,0])) for i in np.arange(len(self.cat1_train))]   # カテゴリ1
		[self.tTrain.append(np.array([0,1,0])) for i in np.arange(len(self.cat2_train))]   # カテゴリ2
		[self.tTrain.append(np.array([0,0,1])) for i in np.arange(len(self.cat3_train))]   # カテゴリ3
		self.tTrain = np.array(self.tTrain).T
		
		# テストデータのラベル（one-hotベクトル）を作成
		self.yTest = np.hstack([np.zeros(len(self.cat1_test)), np.ones(len(self.cat2_test)), 2*np.ones(len(self.cat3_test))])
		[self.tTest.append(np.array([1,0,0])) for i in np.arange(len(self.cat1_test))]   # カテゴリ1
		[self.tTest.append(np.array([0,1,0])) for i in np.arange(len(self.cat2_test))]   # カテゴリ2
		[self.tTest.append(np.array([0,0,1])) for i in np.arange(len(self.cat3_test))]   # カテゴリ3
		self.tTest = np.array(self.tTest).T
	#------------------------------------

	#------------------------------------
	# 学習した識別境界の描画
	def plot(self, dataType="train", prefix="artificial"):

		# データの選択
		if dataType=="train":
			cat1 = self.cat1_train
			cat2 = self.cat2_train
			cat3 = self.cat3_train
		else:
			cat1 = self.cat1_test
			cat2 = self.cat2_test
			cat3 = self.cat3_test
		
		fig = plt.figure()
	
		# 学習データを描画
		# カテゴリ1
		x1, x2 = np.vstack(cat1).transpose()
		plt.plot(x1, x2, 'o', color="#FFA500", markeredgecolor='k', markersize=14)

		# カテゴリ2	
		x1, x2 = np.vstack(cat2).transpose()
		plt.plot(x1, x2, 's', color="#FFFF00", markeredgecolor='k', markersize=14)
	
		# カテゴリ3
		x1, x2 = np.vstack(cat3).transpose()
		plt.plot(x1, x2, '^', color="#FF00FF", markeredgecolor='k', markersize=14)

		# plotラベルの設定
		plt.title("Data", fontsize=14)
		plt.xlabel("x1", fontsize=14)
		plt.ylabel("x2", fontsize=14)
		plt.tick_params(labelsize=14)
		plt.legend(("category 1","category 2","category 3"))

		# 表示範囲の設定
		plt.xlim(-6, 6)
		plt.ylim(-6, 6)
		
		# 保存
		fullpath = os.path.join(self.visualPath,"{}_data.png".format(prefix))
		plt.savefig(fullpath)
		
		# 表示
		plt.show()
	#------------------------------------

	#------------------------------------
	# 学習した識別境界の描画
	def plotClassifier(self, classifier, dataType="train", catInds=[0,1,2], prefix="posterior"):

		# データの選択
		if dataType=="train":
			cat1 = self.cat1_train
			cat2 = self.cat2_train
			cat3 = self.cat3_train
		else:
			cat1 = self.cat1_test
			cat2 = self.cat2_test
			cat3 = self.cat3_test

		for catInd in catInds:
			#fig = plt.figure()
		
			# 学習データを描画
			# カテゴリ1
			x1, x2 = np.vstack(cat1).transpose()
			plt.plot(x1, x2, 'o', color="#FFA500", markeredgecolor='k', markersize=14)

			# カテゴリ2	
			x1, x2 = np.vstack(cat2).transpose()
			plt.plot(x1, x2, 's', color="#FFFF00", markeredgecolor='k', markersize=14)
	
			# カテゴリ3
			x1, x2 = np.vstack(cat3).transpose()
			plt.plot(x1, x2, '^', color="#FF00FF", markeredgecolor='k', markersize=14)
	
			# 識別境界を描画
			N = len(cat1)+len(cat2)+len(cat3)
	
			#メッシュの作成
			X1, X2 = plt.meshgrid(plt.linspace(-6,6,50), plt.linspace(-6,6,50))
			width, height = X1.shape
			X1.resize(X1.size)
			X2.resize(X2.size)
			Z = np.array([classifier.predict(np.array([[x1], [x2]]))[catInd] for (x1, x2) in zip(X1, X2)])
			X1.resize((width, height))
			X2.resize((width, height))
			Z.resize((width, height))
	
			# contourプロット
			levels=[x / 10.0 for x in np.arange(0, 11, 1)]
			CS = plt.contourf(X1,X2,Z,levels) 
	
			# contourの数値ラベル
			plt.clabel(CS, colors='black', inline=True, inline_spacing=0, fontsize=14)
	
			# contourのカラーバー
			CB = plt.colorbar(CS)
			CB.set_ticks(levels)
			CB.ax.tick_params(labelsize=14)
			for line in CB.lines: 
				line.set_linewidth(20)

			# 色空間の設定
			plt.jet()
	
			# plotラベルの設定
			plt.title("p(y={}|x)".format(catInd+1), fontsize=14)
			plt.xlabel("x1", fontsize=14)
			plt.ylabel("x2", fontsize=14)
			plt.tick_params(labelsize=14)

			# 表示範囲の設定
			plt.xlim(-6, 6)
			plt.ylim(-6, 6)

			# レジェンド
			plt.legend(["category 1","category 2","category 3"])

			# 画像として保存
			fullpath = os.path.join(self.visualPath,"{}_{}.png".format(prefix,catInd+1))
			plt.savefig(fullpath)
			
			plt.close()
	#------------------------------------
#------------------------------------

#------------------------------------
# sentimental labelled sentence用のクラス
class sentimentalLabelledSentences:
	dataPath = 'sentiment_labelled_sentences'  # データのフォルダ名
	
	#------------------------------------
	# CSVファイルの読み込み
	# fname: ファイルパス（文字列）
	def __init__(self,fname):
	
		# ファイルのパス設定
		fullpath = os.path.join(self.dataPath,fname)

		# csv形式のデータ読み込み
		self.data = pd.read_csv(fullpath,'\t')
		
		# データ数
		self.nData = len(self.data)
	#------------------------------------

	#------------------------------------
	# 文字列検索
	# keyword: 検索キーワード（文字列）
	def search(self, keyword):
		# sentence列で、keywordを含む要素のインデックスを取得
		results = self.data['sentence'].str.contains(keyword)
		
		# np.arrayとして返す
		return self.data['sentence'][results].values
	#------------------------------------

	#------------------------------------
	# 文章を単語リストに変換
	# sentence: 文章（文字列）
	def sentence2words(self,sentence):
		# 句読点
		punc = re.compile(r'[\[,\],-.?!,:;()"|0-9]')

		# 文章を小文字（lower）に変換し、スペースで分割（split）し、
		# 句読点を取り除き（punc.sub）、語wordsを取り出す
		words = [punc.sub("",word) for word in sentence.lower().split()] 

		# 空の要素を削除
		if words.count(""):
			words.pop(words.index(""))

		return words
	#------------------------------------

	#------------------------------------
	# 単語n-gram辞書の作成
	# N: wordGramのオーダー（スカラー）
	# words: 単語リスト（リスト）
	def wordNgram(self,N,words):
		
		# 各wordからN個先の語をまとめて"-"で繋ぐ
		wordNgram = ["-".join(words[ind:ind+N]) for ind in np.arange(len(words))]
		
		return wordNgram
	#------------------------------------

	#------------------------------------
	# 単語辞書の作成
	# N: wordGramのオーダー
	def makeWordNgramDict(self,N=2,trainInd=[]):
		self.gramNum = N

		# 単語n-gramを格納するリストと、単語n-gramを含む文章数を格納するnp.array
		self.wordNgramDict = []
		self.wordNgramDictCnt = np.array([])
		
		if not len(trainInd): trainInd = np.arange(len(self.data))
		
		# 各文章self.data['sentence']に対する処理
		for sentence in self.data['sentence'][trainInd]:
		
			# 文章sentenceをn-gramに変換
			words = self.sentence2words(sentence)
			wordNgram = self.wordNgram(self.gramNum,words)
			
			# wordNgramDictへの登録とwordNgramDictCntのカウント
			wgList = []		# 重複カウント防止リスト
			for wg in wordNgram:
			
				# wgが既にwordNgramDictに登録されている場合
				if self.wordNgramDict.count(wg):
					
					# 重複カウントチェック
					if not wgList.count(wg):
						# 重複カウント防止リストに新しい単語を追加
						wgList.append(wg)
						
						# インクリメント
						ind = self.wordNgramDict.index(wg)
						self.wordNgramDictCnt[ind] += 1
				
				# wgがwordNgramDictに登録されていない場合
				else:
					# 新規に要素を追加
					self.wordNgramDict.append(wg)
					self.wordNgramDictCnt = np.append(self.wordNgramDictCnt,1)
		
		# 辞書のサイズ
		self.nDict = len(self.wordNgramDict)
		
		# IDF (inverse document frequency)の計算
		self.idf = np.log(self.nDict) - np.log(self.wordNgramDictCnt) + 1

	#------------------------------------

	#------------------------------------
	# 単語の出現頻度(TF: term frequency)の計算
	# sentence: 文章（文字列）
	def tf(self,sentence):

		# 文章を単語arrayに変換
		words = self.sentence2words(sentence)
		wordNgram = self.wordNgram(self.gramNum,words)
		
		# word n-gramの出現回数
		tf = np.zeros(len(self.wordNgramDict))
		for wg in wordNgram:
		
			# wgがwordNgramDictに登録されている場合は、tfをカウント
			cnt = self.wordNgramDict.count(wg)
			if cnt:
				ind = self.wordNgramDict.index(wg)
				tf[ind] += cnt

		# tfの正規化(0割回避のために、分母に+1)
		tf = tf/(np.sum(tf)+1)
		
		return tf
	#------------------------------------

	#------------------------------------
	# 重み付き単語の出現頻度(TF-IDF)の計算
	# sentence: 文章（文字列）
	def tfidf(self,sentence):

		# tfの計算
		tf = self.tf(sentence)
	
		# tf-idfの計算
		tfidf = np.matmul(tf,np.diag(self.idf))

		return tf, tfidf
	#------------------------------------
	
	#------------------------------------
	# 学習と評価データの作成
	# gramNum: n-gramのオーダー（スカラー）
	# trainRatio: 学習データ数の割合（スカラー）
	# isRandom: データをランダムにシャッフルするか否か
	def createData(self,gramNum,trainRatio,isRandom=False):
		
		# 学習データ数
		self.trainNum = np.floor(self.nData * trainRatio).astype(int)

		# データのインデックス
		if isRandom:
			# ランダム
			randInd = np.random.permutation(len(self.data))
		else:
			randInd = np.arange(self.nData).astype(int)
		
		# 学習データのみからn-gram辞書の作成
		self.makeWordNgramDict(gramNum,randInd[:self.trainNum])

		# 各sentenceごとにtfid特徴量の抽出
		tfidfs = []
		for sentence in self.data['sentence']:
			tf,tfidf = self.tfidf(sentence)
			tfidfs.append(tfidf)
		
		# 入力データ行列Xを作成（入力次元Xデータ数）
		self.trainInd = randInd[:self.trainNum]
		self.testInd = randInd[self.trainNum:]
		self.xTrain = np.array(tfidfs)[self.trainInd].T
		self.xTest = np.array(tfidfs)[self.testInd].T
		
		# ラベルベクトルyを作成
		self.yTrain = self.data['score'][self.trainInd].values
		self.tTrain = np.array([[1,0] if y==0 else [0,1] for y in self.yTrain]).T
		self.yTest = self.data['score'][self.testInd].values
		self.tTest = np.array([[1,0] if y==0 else [0,1] for y in self.yTest]).T
	#------------------------------------
#------------------------------------


#------------------------------------
# MNISTデータのクラス
class MNIST:
	dataPath = 'MNIST'  # データのフォルダ名
	imgSize = [28,28]
	
	#------------------------------------
	# CSVファイルの読み込み
	# fname: ファイルパス（文字列）
	def __init__(self):
		
		#---------------
		# 入力データX（入力次元Xデータ数）
		# 学習用
		fp = gzip.open(os.path.join(self.dataPath,'train-images-idx3-ubyte.gz'),'rb')
		data = np.frombuffer(fp.read(),np.uint8,offset=16)
		self.xTrain = np.reshape(data,[-1,self.imgSize[0]*self.imgSize[1]]).T
		self.xTrain = self.xTrain/255

		# 評価用
		fp = gzip.open(os.path.join(self.dataPath,'t10k-images-idx3-ubyte.gz'),'rb')
		data = np.frombuffer(fp.read(),np.uint8,offset=16)
		self.xTest = np.reshape(data,[-1,self.imgSize[0]*self.imgSize[1]]).T
		self.xTest = self.xTest/255

		'''
		# 平均画像
		self.xTrainMean = np.mean(self.xTrain,axis=1)
		self.xTrain = self.xTrain - self.xTrainMean[np.newaxis,1].T
		self.xTest = self.xTest - self.xTrainMean[np.newaxis,1].T
		'''
		#---------------
		
		#---------------
		# カテゴリデータ行列tを作成（カテゴリ数 X データ数）
		# 学習用
		fp = gzip.open(os.path.join(self.dataPath,'train-labels-idx1-ubyte.gz'),'rb')
		self.yTrain = np.frombuffer(fp.read(),np.uint8,offset=8)
		self.tTrain = np.zeros([10,len(self.yTrain)])	# one-hot
		[self.tTrain.itemset(self.yTrain[ind],ind,1) for ind in np.arange(len(self.yTrain))]
		
		# 評価用
		fp = gzip.open(os.path.join(self.dataPath,'t10k-labels-idx1-ubyte.gz'),'rb')
		self.yTest = np.frombuffer(fp.read(),np.uint8,offset=8)
		self.tTest = np.zeros([10,len(self.yTest)])	# one-hot
		[self.tTest.itemset(self.yTest[ind],ind,1) for ind in np.arange(len(self.yTest))]
		#---------------		
	#------------------------------------
	
	#------------------------------------
	# 指定したインデックスの学習または評価画像のプロット
	# inds: インデックスのリスト
	# isTrain: 学習データか否か（真偽値）
	# predict: 予測カテゴリのリスト
	def plotImg(self, inds=[], isTrain=True, predict=[]):

		if isTrain:
			x = self.xTrain
			y = self.yTrain
			prefix = "train data"
		else:
			x = self.xTest
			y = self.yTest
			prefix = "test data"
			
		if not len(inds):
			inds = np.arange(len(x))
			
		for ind in inds:
			plt.imshow(np.reshape(x[:,ind],[self.imgSize[0],self.imgSize[0]]))
			
			if len(predict):
				plt.title("{} No.{}, GT:{}, predict:{}".format(prefix,ind,y[ind],predict[ind]))
			else:
				plt.title("{} No. {}, GT:{}".format(prefix,ind,y[ind]))
				
			plt.show()
	#------------------------------------

	#------------------------------------
	# 全ての数字画像のプロット
	def plotAllImg(self):
		fig, figInds = plt.subplots(ncols=10, sharex=True)
		
		for figInd in np.arange(len(figInds)):
			imgInd = np.where(self.yTest == figInd)[0][0]
			figInds[figInd].imshow(np.reshape(self.xTest[:,imgInd],[self.imgSize[0],self.imgSize[0]]))
			
		fig.show()
		plt.show()
	#------------------------------------
#------------------------------------

#------------------------------------
# MNISTデータのクラス
class fashion:
	dataPath = 'fashionmnist'  # データのフォルダ名
	imgSize = [28,28]
	
	#------------------------------------
	# CSVファイルの読み込み
	# fname: ファイルパス（文字列）
	def __init__(self):
		
		#---------------
		# 入力データX（入力次元Xデータ数）
		# 学習用
		#fp = gzip.open(os.path.join(self.dataPath,'train-images-idx3-ubyte'),'rb')
		fp = open(os.path.join(self.dataPath,'train-images-idx3-ubyte'),'rb')
		data = np.frombuffer(fp.read(),np.uint8,offset=16)
		self.xTrain = np.reshape(data,[-1,self.imgSize[0]*self.imgSize[1]]).T
		self.xTrain = self.xTrain/255

		# 評価用
		#fp = gzip.open(os.path.join(self.dataPath,'t10k-images-idx3-ubyte'),'rb')
		fp = open(os.path.join(self.dataPath,'t10k-images-idx3-ubyte'),'rb')
		data = np.frombuffer(fp.read(),np.uint8,offset=16)
		self.xTest = np.reshape(data,[-1,self.imgSize[0]*self.imgSize[1]]).T
		self.xTest = self.xTest/255

		'''
		# 平均画像
		self.xTrainMean = np.mean(self.xTrain,axis=1)
		self.xTrain = self.xTrain - self.xTrainMean[np.newaxis,1].T
		self.xTest = self.xTest - self.xTrainMean[np.newaxis,1].T
		'''
		#---------------
		
		#---------------
		# カテゴリデータ行列tを作成（カテゴリ数 X データ数）
		# 学習用
		#fp = gzip.open(os.path.join(self.dataPath,'train-labels-idx1-ubyte'),'rb')
		fp = open(os.path.join(self.dataPath,'train-labels-idx1-ubyte'),'rb')
		self.yTrain = np.frombuffer(fp.read(),np.uint8,offset=8)
		self.tTrain = np.zeros([10,len(self.yTrain)])	# one-hot
		[self.tTrain.itemset(self.yTrain[ind],ind,1) for ind in np.arange(len(self.yTrain))]
		
		# 評価用
		#fp = gzip.open(os.path.join(self.dataPath,'t10k-labels-idx1-ubyte'),'rb')
		fp = open(os.path.join(self.dataPath,'t10k-labels-idx1-ubyte'),'rb')
		self.yTest = np.frombuffer(fp.read(),np.uint8,offset=8)
		self.tTest = np.zeros([10,len(self.yTest)])	# one-hot
		[self.tTest.itemset(self.yTest[ind],ind,1) for ind in np.arange(len(self.yTest))]
		#---------------		
	#------------------------------------
	
	#------------------------------------
	# 指定したインデックスの学習または評価画像のプロット
	# inds: インデックスのリスト
	# isTrain: 学習データか否か（真偽値）
	# predict: 予測カテゴリのリスト
	def plotImg(self, inds=[], isTrain=True, predict=[]):

		if isTrain:
			x = self.xTrain
			y = self.yTrain
			prefix = "train data"
		else:
			x = self.xTest
			y = self.yTest
			prefix = "test data"
			
		if not len(inds):
			inds = np.arange(len(x))
			
		for ind in inds:
			plt.imshow(np.reshape(x[:,ind],[self.imgSize[0],self.imgSize[0]]), "gray", vmin=0, vmax=1)
			
			if len(predict):
				plt.title("{} No.{}, GT:{}, predict:{}".format(prefix,ind,y[ind],predict[ind]))
			else:
				plt.title("{} No. {}, GT:{}".format(prefix,ind,y[ind]))
				
			plt.show()
	#------------------------------------

	#------------------------------------
	# 全ての数字画像のプロット
	def plotAllImg(self):
		fig, figInds = plt.subplots(ncols=10, sharex=True)
		
		for figInd in np.arange(len(figInds)):
			imgInd = np.where(self.yTest == figInd)[0][0]
			figInds[figInd].imshow(np.reshape(self.xTest[:,imgInd],[self.imgSize[0],self.imgSize[0]]), "gray", vmin=0, vmax=1)
			
		fig.show()
		plt.show()
	#------------------------------------
#------------------------------------
