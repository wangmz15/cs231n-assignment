# -*- encoding: utf-8 -*-
import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops. 使用显式循环计算softmax损耗及其梯度。    #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train): #对于每一张图，都是X中的一行
        scores = X[i].dot(W)  #乘出的分数都是一行，有C个数字
        scores -= np.max(scores) #平移到最大为0
        exp_sum = np.sum(np.exp(scores)); #中间值，之后需要用到
        for j in range(num_classes): #对于这一行结果中的每一个数字
            if j == y[i]: #求导结果
                dW[:,y[i]] += -X[i] + X[i]*(np.exp(scores[y[i]])/exp_sum) # ∇Wj Li = -X[i] + X[i]*e^j/∑e^k + 2λWj (j≠yi)
            else:
                dW[:,j] += X[i]*(np.exp(scores[j])/exp_sum) # ∇Wj Li = X[i]*e^j/∑e^k + 2λWj (j≠yi)
        loss += -scores[y[i]]+np.log(exp_sum)
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  dW /= num_train
  dW +=  reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops. 非显示循环 #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  scores -= np.max(scores,axis = 1).reshape((num_train,1)) #平移到最大为0 ??????为什么留着这一句就不大对了 哦原来是需要reshape一下 学习了
  exp_scores = np.exp(scores) #把结果矩阵的每一个元素都放在指数上
  exp_correct_scores = exp_scores[np.arange(num_train),y]

  exp_sum = np.sum(exp_scores,axis = 1) #把每一行加起来，得到一个列向量
  exp_scores /= np.tile(exp_sum.reshape(num_train,1),(1,num_classes)) #把每一行的每一个数字都除以这个sum
#   exp_scores = np.array(np.exp(scores) / np.matrix(exp_sum).T)


#   loss = np.sum(-np.log(exp_correct_scores/exp_sum))
  loss = np.sum(-scores[range(num_train),y]) + np.sum(np.log(exp_sum))
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

    
  #算梯度dW
  exp_scores[np.arange(num_train),y] -= 1
  dW = X.T.dot(exp_scores)
  
  dW /= num_train
  dW +=  reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

