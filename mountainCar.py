import gym
import numpy as np
import pickle
import glob
import os
import pdb

class mountainCar:
	modelDir = 'models'
	
	def __init__(self, nSplit = 40, gamma = 0.99, alpha = 0.2):
	
		# mountainCarの読み込み
		self.env = gym.make('MountainCar-v0')
		
		# 各状態の次元の分割数
		self.nSplit = nSplit
		
		# Qテーブルの初期化
		self.Q = np.zeros((self.nSplit, self.nSplit, 3))
		
		# 各状態の最小値と最大値
		self.env_low = self.env.observation_space.low
		self.env_high = self.env.observation_space.high
		
		# 各状態の細かさ
		self.env_dx = (self.env_high - self.env_low) / self.nSplit
		
		# エピソードの初期化
		self.episode = 0
		
		# ステップの初期化
		self.step = 0
		
		# 収益の履歴の初期化
		self.returns = []
		
		# 割引率
		self.gamma = gamma
		
		# 学習率
		self.alpha = alpha
		
	# Qテーブルの設定
	def setQ(self,Q):
		self.Q = Q

	# Qテーブルの保存
	def dumpQ(self):
		fname = os.path.join(self.modelDir,'mountainCar_{}.pkl'.format(episode))
		with open(fname,'wb') as fp:
			pickle.dump(self.Q,fp)

	# Qテーブルの読み込み
	def loadQ(self, postFix):
		fname = os.path.join(self.modelDir,'mountainCar_{}.pkl'.format(postFix))
		with open(fname,'rb') as fp:
			self.Q = pickle.load(fp)
		
	# 状態の取得
	def discretizeState(self,state):

		# 離散値に変換
		position = int((state[0] - self.env_low[0])/self.env_dx[0])
		velocity = int((state[1] - self.env_low[1])/self.env_dx[1])

		return position, velocity


	# 初期化
	def reset(self):
		
		# 環境の初期化
		tmp_state = self.env.reset()
		
		# 収益の初期化
		self.sum_rewards = 0
		
		# 状態の初期化
		state = self.discretizeState(tmp_state)
		
		return state
		
	# 行動の選択
	def selectAction(self, state, epsilon=0.002):
			
		if np.random.uniform(0, 1) > epsilon:
			action = np.argmax(self.Q[state[0]][state[1]])
		else:
			action = np.random.choice([0, 1, 2])
		return action

	# 行動の実行
	def takeAction(self, action):
		# 行動の実行、次の状態・報酬・ゲーム終了FLG・詳細情報を取得
		tmp_next_state, reward, done, _ = self.env.step(action)

		# 次の状態の離散化
		next_state = self.discretizeState(tmp_next_state)
		
		# 収益の更新
		self.sum_rewards += reward
		
		if done:
			# doneがTrueになったら１エピソード終了
			self.returns.append(self.sum_rewards)
		
		# ステップのインクリメント
		self.step += 1
		
		return next_state, reward, done
		
	# Qの更新
	def updateQ(self, state, action, next_state, reward):
		# 行動後の状態で得られる最大行動価値 Q(s',a')
		next_max_Qvalue = max(self.Q[next_state[0]][next_state[1]])

		# 行動前の状態の行動価値 Q(s,a)
		Qvalue = self.Q[state[0]][state[1]][action]
		
		# 行動価値関数の更新
		self.Q[state[0]][state[1]][action] = Qvalue + self.alpha * (reward + self.gamma * next_max_Qvalue - Qvalue)

	
if __name__ == '__main__':
	isDemo = True

	# mountainCarのインスタンス
	agent = mountainCar()
	

	# Qテーブルを読み込んでデモ
	if isDemo == False:
		for episode in range(0,10000,500):
			# Qテーブルの読み込み
			agent.loadQ(episode)
			
			# 環境のリセット
			state = agent.reset()
			
			print('episode: {}'.format(episode))

			for _ in range(200):
				# 表示
				agent.env.render()
				
				# 行動を選択
				action = agent.selectAction(state, epsilon=0)

				# 行動を実行
				next_state, reward, done = agent.takeAction(action)

				if done:
					break

	else:
		# 10000エピソードで学習する
		for episode in range(10000):

			# 環境のリセット
			state = agent.reset()

			# 100回に1回Qテーブルを保存
			if episode%100 == 0:
				agent.dumpQ()

			for _ in range(200):
				# 行動を選択
				action = agent.selectAction(state)

				# 行動を実行
				next_state, reward, done = agent.takeAction(action)
			
				# Qテーブルの更新
				agent.updateQ(state, action, next_state, reward)
			
				# 状態の更新
				state = next_state

				if done:
					if episode%100 == 0:
						print('episode: {}, total_reward: {}'.format(episode, agent.sum_rewards))
					break


