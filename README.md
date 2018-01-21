# torikarasu
Handwriting recognition. "鳥" and "烏".

**特別演習課題**

## Fisherの線形判別分析


## 実験結果
　あとで書く予定ぷり！
 
```
Read feature data. The Shape:(2, 200, 196)

1 回目
threshold = -0.0918018510341
class1 正解率:18/20 認識率:90.0[%]
class2 正解率:16/20 認識率:80.0[%]
total  正解率:34/40 認識率:85.0[%]

2 回目
threshold = -0.0657123518397
class1 正解率:14/20 認識率:70.0[%]
class2 正解率:16/20 認識率:80.0[%]
total  正解率:30/40 認識率:75.0[%]

3 回目
threshold = -0.0791193170331
class1 正解率:16/20 認識率:80.0[%]
class2 正解率:14/20 認識率:70.0[%]
total  正解率:30/40 認識率:75.0[%]

4 回目
threshold = -0.0875604310618
class1 正解率:17/20 認識率:85.0[%]
class2 正解率:14/20 認識率:70.0[%]
total  正解率:31/40 認識率:77.5[%]

5 回目
threshold = -0.111142257417
class1 正解率:11/20 認識率:55.00000000000001[%]
class2 正解率:13/20 認識率:65.0[%]
total  正解率:24/40 認識率:60.0[%]
```

## メモ
　どうでもいいけどフィッシャーのプログラム書いてるときに30分くらい時間を無駄にした原因をメモ．  
　pythonで行列を転置したいときnumpy配列なら``.T``や``.transpose()``でできるのに，1次元のベクトルのときだけは無理らしい．転置されない．
 
 ```
>>> import numpy as np
>>> a = np.array([1,2,3])
>>> a
array([1, 2, 3])

>>> a.T
array([1, 2, 3])
>>> a.transpose()
array([1, 2, 3])
 ```
 
　1次元のベクトルを転置するときは``.reshape()``を使うしかないらしい．``.T``が使えない，短くて楽なのに…
 
 ```
 >>> a.reshape(3,1)
array([[1],
       [2],
       [3]])
 ```
 
ちなみにmatlabは``transpose()``でちゃんとベクトルを転置してくれます．他の言語の行列演算ライブラリも大体してくれそう．