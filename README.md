## MY&PY天池机器学习
### 第一天
额，对于其他人来说可能是直接就开始学内容了，可是我啊，是不是选错博客了，我选Github作为写博客的地方，但是却从未用过。
没想到用好Github也是一件不容易的事情呢，本来想下载一个模板再写的，可是几经波折后发现不是一两天内能够弄好的事情，能够用好Github这个工具也相当于一个训练营的内容了，嗯，一边学训练营的内容一边搞博客空间好了哈哈哈，前期博客就走简单路线了。 
TASK 1 的代码
```
Demo实战
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

x_fearures = np.array([-1, -2],[-2, -1],[-3, -2],[1, 3],[2, 1],[3,2]])
y_label = np,array([0, 0, 0, 1 , 1, 1])

Ir_clf = LogistincRegression()

Ir_clf = Ir_clf.fit(x_fearures,y_label)

print('the weight of Logistic Regression:',Ir_clf,coef_)

print('the intercept(w0) of Logistic Regression:',Ir_clf.intercept_)

plt.figure()
plt.scatter(x_fearures[:,0],x_fearures[:,1],c= y_label, s=50, cmap= 'viridis'
plt.title('Dataset')
plt.show()

plt.figure()
plt.scatter(x_fearures[:,0],x_fearures[:,1],c= y_label,s=50,cmap='viridis')
plt.title('Dataset')

nx,ny = 200,100
x_min, x_max = plt,xlim()
y_min, y_max = plt,ylim()
x_grid, y_grid = np.meshgrid(np.linspace(x_xin, x_max, nx),np.linspace(y_min, y_max, ny))

z_proba = Ir_clf.predict_proba(np.c_[x_grid.ravel(), y_grid.ravel()])
z_proba = z_proba[:, 1].reshape(x_grid.shape)
plt.contour(x_grid,y_grid, z_proba,[0.5], linewidths = 2.,colors ='blue')

plt.show()

plt.figure()

x_fearures_new1 = np.array([[0, -1]])
plt.scatter(x_fearures_new1[:, 0],x_fearures_new1[:.1]. s= 50 ,cmap = 'vididis')
plt.annotate(s= 'New point 1', xy=(0,-1),xytext= (-2,0),color = 'blue', arrowprops = dict(arrowstyle= '-|>',connectionstyle= 'arc3',color= 'red'))

x_fearures_new2 = np.array([[1,2]])
plt.scatter(x_fearures_new2[:,0],x_fearures_new2[:,1],s= 50, cmap= 'viridis')
plt.annotate(s='New point 2‘，xy= (1,2),xytext= (-1.5,2.5),color= 'red',arrowprops=dict(arrowstyle= '-|>',connectionstyle= 'arc3',color= 'red'))

plt.scatter(x_fearures[:,0],x_fearures[:,1],c= y_label,s=50,cmap='viridis')
plt.title('Dataset')

plt.contour(x_grid,y_grid,z_proba, [0.5],linewidths= 2.,color='bule')

plt.show()

y_label_new1_predict = Ir_clf.predict(x_fearures_new1)
y_label_new2_predict = Ir_clf.predict(x_fearures_new2)

print('The New point 1 predict class:\n',y_label_new1_predict)
print('The New point 2 predict class:\n',y_label_new2_predict)

y_label_new1_predict_proba = Ir_clf.predict_proba(x_fearures_new1)
y_label_new2_predict_proba = Ir_clf.predict_proba(x_fearures_new2)

print('The New pront 1 predict Probability of each class:\n', y_label_new1_predict_proba)
print('The New pront 2 predict Probability of each class:\n', y_label_new2_predict_proba)

基于鸢尾花数据集的逻辑回归分类实践 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.datasets import load_iris
data  load_iris()
iris_target = data.target
iris_features = pd.DataFrame(data=data.data, columns=data.feature_names)

iris_features.info()
iris_features.head()
iris_features.tail()
iris_target
pd.Series(iris_target).value_counts()
iris_features.describe()
iris_all+ iris_features.copy()
iris_all['target'] = iris_target
sna.pairplot(data= iris_all,diag_kind='hist', hue= 'target')
plt.show()

for col in iris_features.columns:
    sns.boxplot(x='target', y=col, saturation=0.5, palette='pastel'. data= iris_all)
    plt.title(col)
    plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize= (10,8))
ax = fig.add_subplot(111, projection='3d')

iris_all_class0 = iris_all[iris_all['target']==0].values
iris_all_class1 = iris_all[iris_all['target']==1].values
iris_all_class2 = iris_all[iris_all['target']==2].values

ax.scatter(iris_all_class0[:,0], iris_all_class0[:,1], iris_all_class0[:,2],label='setosa')
ax.scatter(iris_all_class1[:,0], iris_all_class0[:,1], iris_all_class0[:,2],label='versicolor')
ax.scatter(iris_all_class2[:,0], iris_all_class0[:,1], iris_all_class0[:,2],label='virginica')
plt.legend()

plt.show()

利用逻辑回归模型在二分类上进行训练和预测
from sklearn.model_selection import train_test_split

iris_features_part = iris_features.iloc[:100]
iris_target_part = iris_target[:100]

x_train, x_test, y_train, y_test = train_test_split(iris_featurs_part, isis_target_part, test_size = 0.2, random_state = 2020)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0 , solver = '1bfgs')
clf.fit(x_train, y_train)

print('the weight of Logistic Regression:' ,clf.coef_)

print('the intercept(w0) of Logistic Regression:', clf.intercept_)

train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)

from sklearn import metrics

print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_train,train_predict))
print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_test,test_predict))

coinfusion_matrix_result = metrics. confusion_matrix(test_predict. y_test)
print('The confusion matrix result:\n', confusion_matrix_result)

plt.figure(figsize= (8,6))
sns.heatmap(confusion_matrix_ressult, annot=True. cmap= 'Bules')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(iris_features, iris_target, test_size = 0.2, random_state = 2020)
clf = LogisticRegression(random_state=0, solver='1bfgs')
clf.fit(x_train, y_train)

print('the weight of Logistic Regression:\n', clf.coef_)
print('the intercept(w0) of Logistic Regression:\n', clf.intercept_)

train_predict = clf.predict(x_trrain)
test_predict = clf.predict(x_test)

train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)

train_predict_proba = clf. predict_proba(x_train)
test_predict_proba = clf.predict_proba(x_test)

print('the test predict Probability of each class:\n', test_predict_proba)
print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_train, train_predict))
print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_test,test_predict))

confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
print('The confusion matrix result:\n', confusion_matrix_result)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot= True, cmap= 'Bules'
plt.xlabel('Predicted labels')
plt.ylabel(True labels')
plt.show()

```
第二天用了一个小时把第一天没敲的代码敲上了 ，嗯，现在开始进行观察。。。

###第二天
逻辑回归是适合于二倍以上的分类方法，传统方法
突出点：模型简单，模型的可解释性强
由函数图像的对称点将其结果分为0和1

Demo
库：numpy matplotlib.pyplot seaborn 
逻辑回归模型函数： sklearn.liner_model
查看输入的x和y值所得出的w0,w的方法  intercept_ , coef_

模型数据可视化的4步
plt.
plt.
plt.
plt.
但是，x_fearures[:,1] 是什么意思呢？还有s=50又是指什么？？，为啥cmap='viridis'？？？


