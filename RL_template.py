# -*- coding: utf-8 -*-

import gym
import numpy as np
import pickle
import glob
import os
import pdb

class RL:
	modelPath = 'models'
	visualPath = 'visualization'
	
	#------------------------------------
	# 1) 強化学習の環境および変数の初期化
	# env: 強化学習タスク環境名
	# gamma: 割引率
	# nSplit: 状態の分割数
	# isVisualizae: 可視化するか否かのフラグ
	def __init__(self, env, gamma = 0.99, nSplit=50, isVisualize=False):
	
		# 環境の読み込み
		self.env = gym.make(env)
		
		# 割引率
		self.gamma = gamma

		# 描画Flag
		self.isVisualize = isVisualize

		# 割引報酬和を格納
		self.sumRewards = []

		# 行動数
		self.nAction = self.env.action_space.n
		
		# 各状態の最小値と最大値
		self.stateMin = self.env.observation_space.low
		self.stateMax = self.env.observation_space.high

		# 状態の分割数
		self.nSplit = nSplit
	#------------------------------------
		
	#------------------------------------
	# 2) 状態および変数の初期化
	def reset(self):
		
		# 環境の初期化
		state = self.env.reset()
		
		# 割引報酬和の初期化
		self.sumReward = 0

		# ステップの初期化
		self.step = 0
		
		return state
	#------------------------------------

	#------------------------------------
	# 3) 行動の選択
	# state: 状態ベクトル
	def selectAction(self, state):
	
		action = 1 #【行動の選択】

		return action
	#------------------------------------

	#------------------------------------
	# 4) 行動の実行
	# action: 行動インデックス
	def doAction(self, action):

		# 行動の実行、次の状態・報酬・ゲーム終了FLG・詳細情報を取得
		next_state, reward, done, _ = self.env.step(action)

		# 割引報酬和の更新
		self.sumReward += 0 #【割引報酬和の計算】

		# ステップのインクリメント
		self.step += 1

		if self.isVisualize:
			self.env.render()
		
		if done:
			# doneがTrueになったら１エピソード終了
			self.sumRewards.append(self.sumReward)
		
		return next_state, reward, done
	#------------------------------------

#-------------------
# メインの始まり
if __name__ == '__main__':

	# 1) 強化学習の環境の作成
	agent = RL(env='MountainCar-v0', gamma=0.99, isVisualize=True)

	# 2) エピソード（試行）のループ
	for episode in np.arange(1001):
			
		if not episode % 500:
			agent.isVisualize = True
		else:
			agent.isVisualize = False

		# 3) 環境の初期化
		x = agent.reset()
			
		# 4) ステップのループ
		while(1):
				
			# 5) 行動を選択
			y = agent.selectAction(x)

			# 6) 行動を実行
			x_next, r, done = agent.doAction(y)
			
			x = x_next

			if done:
				break

		print('Episode:{}, sum of rewards:{}'.format(episode,agent.sumReward))
		
	# 7) 強化学習環境の終了
	agent.env.close()
#メインの終わり
#-------------------