'''
进行一个简单线性拟合的例子，不使用任何框架，手动求导
'''
import numpy as np


N, D_in, H, D_out = 64, 1000, 100, 10
LEARNING_RATE=1e-6

# 随机创建一些训练数据
x = np.random.randn(N, D_in)
#print("x:\n",x.shape)
y = np.random.randn(N, D_out)
#print("y:\n",y.shape)

w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

for it in range(500):
    # Forward pass
    h = x.dot(w1) # N * H
    h_relu = np.maximum(h, 0) # N * H
    y_pred = h_relu.dot(w2) # N * D_out

    # compute loss
    loss = np.square(y_pred - y).sum()
    print(it, loss)


