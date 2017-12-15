#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on Mar 30 15:45:19 2016

@author: gfmei
"""
import MySQLdb
import rem_noise
from svmutil import *
from pylab import *
try:
    conn = MySQLdb.connect(host='localhost', user='root', passwd='51desk', db='Place51desk', charset='utf8', port=3306)
    cur = conn.cursor()
    cur.execute("select id, label from ver_image where nu > 50000")
    results = cur.fetchall()
    conn.commit()
    cur.close()
    conn.close()
except Exception, e:
    print e.message
class1 = []
class2 = []
labels = []
test = []
t_labels = []

for row in results:
    pic = row[0] + '.jpg'
    im = array(rem_noise.get_image(pic))
    class1.append(im.reshape(-1, 1))
    labels.append(int(row[1]))


samples = map(list, array(class1).reshape(array(class1).shape[0], -1))
ex = array(rem_noise.get_image('captcha12.jpg')).reshape(1, -1)
prob = svm_problem(labels, samples)
# param = svm_parameter('-t 1 -d 5')
param = svm_parameter('-t 1 -d 3')
m = svm_train(prob, param)
print type(m)
p_label, p_acc, p_val = svm_predict([2], map(list, array(ex).reshape(array(ex).shape[0], -1)), m)

def ver_result():
    return int(p_label[0])