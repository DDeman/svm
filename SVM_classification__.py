#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '任晓光'
__mtime__ = '2020/4/3'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# np.set_printoptions(suppress=True)


class SVM__smo_simple():
    def __init__(self):
        pass


    def select_j(self, i, m):
        while True:
            j = int(np.random.uniform(0, m))
            if j != i:
                break
        return j

    def calcute_L_H(self, i, j, C, y_array, alpha):
        if y_array[i] != y_array[j]:
            L = max(0, alpha[j] - alpha[i])
            H = min(C, C + alpha[j] - alpha[i])
        else:
            L = max(0, alpha[i] + alpha[j] - C)
            H = min(C, alpha[i] + alpha[j])
        return L, H

    def cat_alpha(self, L, H, alpha):
        if alpha > H:
            return H
        elif alpha < L:
            return L
        else:
            return alpha

    def calcute_Ei(self,alpha,x_array,y_array,b):
        for idx in range(len(alpha)):
            gx = sum(np.dot(x_array,x_array[idx]) * y_array * alpha) + b
            self.E[idx] = gx - y_array[idx]

    def fit(self, data, C, toler, max_iters):
        x_pd = data.iloc[:, :-1]
        y_pd = data.iloc[:, -1]
        m, n = x_pd.shape
        alpha = np.zeros((m,))
        b = 0
        x_array = np.array(x_pd)
        y_array = np.array(y_pd)
        iters = 0
        max_iters = max_iters
        self.E = np.zeros((m,1))
        self.calcute_Ei(alpha,x_array,y_array,b)
        kk = 0

        while iters < max_iters:
            i = self.select_i(alpha, y_array, x_array, C, b)

            E_i = self.E[i]
            # print((E_i))
            # b = alpha[i]
            if ((y_array[i] * E_i < - toler) and (alpha[i] < C)) or ((y_array[i] * E_i > toler) and (alpha[i] > 0)):
                j = self.select_j(i, m)
                E_j = sum(np.dot(x_array, x_array[j]) * y_array * alpha) + b - y_array[j]
                L, H = self.calcute_L_H(i, j, C, y_array, alpha)
                if L == H:
                    continue
                eta = np.dot(x_array[i], x_array[i]) + np.dot(x_array[j], x_array[j]) - 2 * np.dot(x_array[i],
                                                                                                   x_array[j])
                if eta <= 0:
                    continue
                alpha_j_new_unc = alpha[j] + y_array[j] * (E_i - E_j) / eta

                alpha_j_new = self.cat_alpha(L, H, alpha_j_new_unc)

                alpha_i_new = alpha[i] + y_array[i] * y_array[j] * (alpha[j] - alpha_j_new)

                bi_new = b - E_i - y_array[i] * (alpha_i_new - alpha[i]) * np.dot(x_array[i], x_array[i]) - y_array[
                                                                                                                j] * (
                                                                                                            alpha_j_new -
                                                                                                            alpha[
                                                                                                                j]) * np.dot(
                    x_array[j], x_array[i])
                bj_new = b - E_j - y_array[i] * (alpha_i_new - alpha[i]) * np.dot(x_array[i], x_array[j]) - y_array[
                                                                                                                j] * (
                                                                                                            alpha_j_new -
                                                                                                            alpha[
                                                                                                                j]) * np.dot(
                    x_array[j], x_array[j])

                if 0 < alpha_i_new < C:
                    b = bi_new
                elif 0 < alpha_j_new < C:
                    b = bj_new
                else:
                    b = (bi_new + bj_new) / 2

                alpha[i], alpha[j] = alpha_i_new, alpha_j_new
                # print(alpha)
            iters += 1

            print('iters is ：', iters)
            # print(alpha)

        self.w = np.dot(alpha * y_array, x_array)
        j = None
        for i in range(m):
            if alpha[i] > 0:
                j = i
                continue
        self.b = y_array[j] - alpha * y_array * np.dot(x_array, x_array[j])
        return self.w, self.b

    def predict(self, x_array):
        pred = np.dot(x_array, self.w) + self.b
        return pred

    def select_i(self, alpha, y_array, x_array, C, b):
        for idx in range(len(alpha)):
            gx = sum(np.dot(x_array,x_array[idx]) * y_array * alpha) + b
            if alpha[idx] == 0:
                if y_array[idx] * gx < 1:
                    return idx
            elif 0 < alpha[idx] < C:
                if y_array[idx] * gx != 1:
                    return idx
            elif alpha[idx] == C:
                if y_array[idx] * gx > 1:
                    return idx
        i = np.random.uniform(0, len(alpha))
        return alpha[i]


if __name__ == '__main__':
    data = load_breast_cancer()
    x, y = data.data, data.target
    x_pd = pd.DataFrame(x, columns=data.feature_names)
    y_pd = pd.DataFrame(y, columns=['result']).replace([0, 1], [-1, 1])
    data_pd = pd.concat([x_pd, y_pd], axis=1)
    svm = SVM__smo_simple()
    w, b = svm.fit(data_pd, 0.6, 0.001, 5000)


    x_array = np.array(x_pd)
    y_pred = svm.predict(x_array)
    print(y_pred)

    y_p = []
    for i in y_pred:
        if i > 0:
            y_p.append(1)
        else:
            y_p.append(-1)
    from sklearn.metrics import accuracy_score

    print(y_p)
    print(np.array(y_pd).tolist())
    acc = accuracy_score(y_pd, y_p)
    print(acc)

