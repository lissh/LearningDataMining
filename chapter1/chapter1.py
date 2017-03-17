# -*- coding:utf-8 -*-

#输出主目录
import os,time
# print(os.path.expanduser("~"))
import numpy as np 
import csv

from sklearn.cross_validation import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import cross_val_score

from matplotlib import pyplot as plt

data_folder = '/Users/lish/Desktop'
data_filename = os.path.join(data_folder,"Ionosphere","ionosphere.data")
X = np.zeros((351,34),dtype = 'float')
y = np.zeros((351,),dtype = 'bool')


with open(data_filename,'r') as input_file:
	reader = csv.reader(input_file)

	for i , row in enumerate(reader):
		data = [float(datum) for datum in row[:-1]]
		X[i] = data
		y[i] = row[-1] == 'g.'

##这里只是进行了一次分割，降样本集分为测试样本和训练样本
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)
print("There are {} samples in the training dataset".format(X_train.shape[0]))
print("There are {} samples in the testing dataset".format(X_test.shape[0]))
print("Each sample has {} features".format(X_train.shape[1]))

estimator = KNeighborsClassifier()#K近邻分类器,KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_neighbors=5, p=2, weights='uniform')
estimator.fit(X_train, y_train)
y_predicted = estimator.predict(X_test)

accuracy = np.mean(y_test == y_predicted) * 100
print("The accuracy is {0:.1f}%".format(accuracy))

##接下来这里将真个大数据分为几个部分，最其中一部分执行以下操作：
#####将其中一部分作为当前的测试集
#####用剩余的部分训练算法
#####在当前测试集上测试算法
##记录每次得分及平均得分
##注：上述过程中，每条数据只能在测试集中出现一次，以减少运气成分
scores = cross_val_score(estimator, X, y, scoring='accuracy')
average_accuracy = np.mean(scores) * 100
print("The average accuracy is {0:.1f}%".format(average_accuracy))


##各K近邻器设置参数，n_neighbors值：为选取多少个近邻作为预测依据。
##我们这里给他一系列的n_neighbors值，重复试验，观察不同参数值所带来的结果之间的差异。
##把不同的n_neighbors值的得分和平均分保存起来，留作分析
avg_scores = []
all_scores = []

parameter_values = list(range(1, 21))  # Including 20
for n_neighbors in parameter_values:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator, X, y, scoring='accuracy')
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)


##这里我们为了更直观的观察分类效果，将不同n_neighbors值与分类正确率关系，用图表示出来
plt.figure(figsize=(32,20))
plt.plot(parameter_values, avg_scores, '-o', linewidth=5, markersize=24)
# plt.axis([0, max(parameter_values), 0, 1.0])
plt.show()





