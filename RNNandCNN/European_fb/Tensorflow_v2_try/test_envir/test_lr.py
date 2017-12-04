from math import *
import tensorflow as tf
import matplotlib.pyplot as plt

lr0 = 0.01
lrd = 0.9996
n = 10
X = 16000

def lr_t(lr0,lrd,n,X,x):
	return lr0*(lrd**x)*0.5*(1.0+cos(2.0*pi*x*n/X))

x = [i+1 for i in range(X)]
y = []
for i in range(X):
	y.append(lr_t(lr0,lrd,n,X,i))

plt.plot(x,y)
plt.show()

print(y[X-1])