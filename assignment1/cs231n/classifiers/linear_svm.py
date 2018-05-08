# -*- encoding: utf-8 -*-
import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  - gradient with respect to weights W; an array of same shape as W 返回的是元祖 loss（float）、梯度dw（和W维度一样的的array）
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero 即要计算的梯度

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
        if j == y[i]:
            continue
        margin = scores[j] - correct_class_score + 1 # note delta = 1
        if margin > 0:
            loss += margin
            dW[:, j] += X[i, :]         #  根据公式： ∇Wj Li = xiT 1(xiWj - xiWyi +1>0) + 2λWj , (j≠yi)
            dW[:, y[i]] += -X[i, :]     #  根据公式：∇Wyi Li = - xiT(∑j≠yi1(xiWj - xiWyi +1>0)) + 2λWyi （最优化笔记（下））
                                        #  解释 如果margin > 0不仅会给第j列带来 +Xi 的增益，也会给第y[i]列带来 -Xi 的增益（所以一行里有多少个正数，就会给dW的y[i]行造成多少的 -X[i]）

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
# 计算损失函数的梯度，并把它存在dW里。相反，首先计算损失，然后计算导数，在计算损失的同时计算导数可能会更简单。。因此，您可能需要修改上面的一些代码来计算渐变。

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.实现结构化SVM损失的矢量化版本，将结果存储在损失中。                #
  #############################################################################
  dW = np.zeros(W.shape) # initialize the gradient as zero 即要计算的梯度

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  scores = X.dot(W)
  correct_class_scores = scores[np.arange(num_train),y]
  margin = np.maximum(0,scores - correct_class_scores[:, np.newaxis] + 1) #margin 与 scores 的size相同
  margin[np.arange(num_train),y] = 0.0 #在margin的第i行里选出index为y[i]的，并置为0
  loss = np.sum(margin, axis = 0) #算每一个图的loss Li
  loss = np.sum(loss)/num_train #算L （还没加上正则化）
  loss += 0.5 * reg * np.sum(W * W)
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  # 实现结构化SVM损失的矢量化梯度版本，将结果存储在dW中。                                                                          #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  X_mask = np.zeros(margin.shape)
  X_mask[margin > 0] = 1 #留下loss值大于0的，仅用于计数
  incorrect_num = np.sum(X_mask, axis = 1) # 把每一行加起来，成为一个列向量
  X_mask[np.arange(num_train),y] = -incorrect_num #X_mask第i行中，index为y[i]的，赋值成incorrect_num[i]
  dW = X.T.dot(X_mask) / num_train
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
