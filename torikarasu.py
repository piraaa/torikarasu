#
# torikarasu.py
# Created by pira on 2017/12/18.
#

#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import glob

DATA_TYPE = 2 #クラス数（tori or karasu）
DATA_NUM = 200 #各クラスの総データ数
TRAIN_NUM = 180 #各クラスの学習用データ数
TEST_NUM = 20 #各クラスの試験用データ数
N = 5 #認識実験の回数
DIM = 196 #2 #データの次元数 (特徴量の数)

path = './data/' #学習データのPath

def load_data():
	data = list()
	for i in np.arange(DATA_TYPE):
		tmp = list()
		file_list = sorted(glob.glob(path + str(i+1) + '/*.txt')) #特徴量ファイル名のリスト
		for fn in file_list:
			with open(fn, 'r') as fin:
				tmp.append([int(n) for n in fin.readlines()]) #listの各要素をstrからintへ変換したものをtmpに格納
		data.append(tmp)

	data = np.array(data)
	print('Read feature data. The Shape:' + str(data.shape) + '\n')
	return data

#直線
def f(x, a, b):
	return a*x+b

#データが2次元のとき用
def plot(fn, cls1, cls2, line1=None, line2=None):
	
	plt.figure(figsize=(5,5))

	x,y = cls1.T
	plt.plot(x, y, 'bo', ms=3, label='class1')
	x, y = cls2.T
	plt.plot(x, y, 'ro', ms=3, label='class2')

	if not (line1 is None):
		plt.plot(line1[0], line1[1], 'k-', ms=5, label='Decision Boundary')

	if not (line2 is None):
		plt.plot(line2[0], line2[1], 'g-', ms=5, label='z')

	plt.xlim(-10,10)
	plt.ylim(-10,10)
	plt.legend(loc='lower right')

	#plt.savefig(fn)
	#plt.clf()
	plt.show()

def fisher(cls1, cls2):
	#リストからnp.arrayに変換（行列の転置や内積を使うため）
	cls1 = np.array(cls1)
	cls2 = np.array(cls2)

	#学習用データの数
	n1 = len(cls1)
	n2 = len(cls2)

	#各クラスの平均値（DIM次元配列）
	mean1 = np.mean(cls1, axis=0)
	mean2 = np.mean(cls2, axis=0)

	#総クラス内共分散行列
	sw = np.zeros((DIM,DIM))
	for xn in cls1:
		xn = xn.reshape(DIM,1)
		mean1 = mean1.reshape(DIM,1)
		sw += np.dot((xn-mean1),(xn-mean1).T)
	for xn in cls2:
		xn = xn.reshape(DIM,1)
		mean2 = mean2.reshape(DIM,1)
		sw += np.dot((xn-mean2),(xn-mean2).T)
	sw_inv = np.linalg.inv(sw)

	#傾きw(xの変化量とyの変化量)を求める
	w = np.dot(sw_inv,(mean1-mean2))

	#2次元で直線z(傾きwで各クラスの平均を通る直線)を図示する
	#mean = (mean1+mean2)/2
	#a = w[1]/w[0] #傾き
	#b = a*mean[0]+mean[1] #切片 -ax+y
	#x = np.linspace(-9.8, 9.8, 1000)
	#y = f(x,a,b)
	#plot('fisher.png', cls1, cls2, (x,y1), (x,y))

	#2次元で決定境界直線を図示する
	#a1 = -w[0]/w[1]
	#b1 = a1*mean[0]+mean[1] #各クラスの平均を通る直線の切片 -ax+y
	#x1 = np.linspace(-8, 8, 1000)
	#y1 = f(x1,a1,b1)

	#plot('fisher.png', cls1, cls2, (x1,y1), (x,y))
	
	return w

if __name__ == '__main__':

	data = load_data()

	#テスト用2次元データ
	#cov = [[3,1],[1,3]] #共分散
	#cls1_train = np.random.multivariate_normal([-5,-5], cov, 100)
	#cls2_train = np.random.multivariate_normal([5,5], cov, 100)
	#cls1_test = np.random.multivariate_normal([-5,-5], cov, 20)
	#cls2_test = np.random.multivariate_normal([5,5], cov, 20)

	for i in np.arange(N):
		
		print(i+1, '回目')

		#学習用データ
		cls1_train = np.r_[data[0][0:i*20][:],data[0][(i+1)*20:200][:]]
		cls2_train = np.r_[data[1][0:i*20][:],data[1][(i+1)*20:200][:]]

		#テストデータ
		cls1_test = data[0][i*20:(i+1)*20][:]
		cls2_test = data[1][i*20:(i+1)*20][:]

		#fisherで線形変換ベクトルwを求める
		w = fisher(cls1_train, cls2_train)
		#print('w =', w, '\n')

		#判別実験
		#閾値を求める
		m1 = sum(np.dot(cls1_train,w))/TRAIN_NUM
		m2 = sum(np.dot(cls2_train,w))/TRAIN_NUM
		threshold = (m1+m2)/2
		print('threshold =', threshold[0])

		#class1
		count1 = 0
		for i in np.arange(20):
			yn = np.dot(cls1_test[i], w)
			if yn > threshold:
				count1 += 1
		print('class1 正解率:' + str(count1) + '/20 認識率:' + str(count1/TEST_NUM*100) + '[%]')

		#class2
		count2 = 0
		for i in np.arange(TEST_NUM):
			yn = np.dot(cls2_test[i], w)
			if yn < threshold:
				count2 += 1
		print('class2 正解率:' + str(count2) + '/20 認識率:' + str(count2/TEST_NUM*100) + '[%]')
		print('total  正解率:' + str(count1+count2) + '/40 認識率:' + str(((count1/TEST_NUM*100)+(count2/TEST_NUM*100))/2) + '[%]\n')
