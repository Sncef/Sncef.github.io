## MY&PY天池机器学习
### 第一天
额，对于其他人来说可能是直接就开始学内容了，可是我啊，是不是选错博客了，我选Github作为写博客的地方，但是却从未用过。<br>
没想到用好Github也是一件不容易的事情呢，本来想下载一个模板再写的，可是几经波折后发现不是一两天内能够弄好的事情，能够用好Github这个工具也相当于一个训练营的内容了，嗯，一边学训练营的内容一边搞博客空间好了哈哈哈，前期博客就走简单路线了。 <br>
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

train_predict = clf.predict(x_train)
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

### 第二天<br>
逻辑回归是适合于二倍以上的分类方法，传统方法<br>
突出点：模型简单，模型的可解释性强<br>
由函数图像的对称点将其结果分为0和1<br>

Demo<br>
库：numpy matplotlib.pyplot seaborn <br>
逻辑回归模型函数： sklearn.liner_model<br>
查看输入的x和y值所得出的w0,w的方法  intercept_ , coef_<br>

模型数据可视化的4步
plt.<br>
plt.<br>
plt.<br>
plt.<br>
但是，x_fearures[:,1] 是什么意思呢？还有s=50又是指什么？？，为啥cmap='viridis'？？？

### 第三天
哈哈，我醒了，今天下午4点就醒了，比昨天早两个小时，昨晚上班不忙睡了一会今天早点起开心！<br>

继续昨天的问题，s的意思，还有camp，必应了一下，发现简书的解释比较清晰，[lianjie](https://www.jianshu.com/p/53e49c02c469) <br>
但是，x_fearures两个数的意思还是想不明白，待我再看看。。

经过我测试![image](https://github.com/Sncef/Sncef.github.io/blob/main/photo/1.jpg)<br>![image](https://github.com/Sncef/Sncef.github.io/blob/main/photo/2.jpg)<br>
![image](https://github.com/Sncef/Sncef.github.io/blob/main/photo/3.jpg)<br>![image](https://github.com/Sncef/Sncef.github.io/blob/main/photo/4.jpg)
<br>
我将问题指向了ARRAY这个数组，应该是数组的用法，后面我查询了数组，嘿嘿，我懂了<br>![image](https://github.com/Sncef/Sncef.github.io/blob/main/photo/5.jpg)
python中数组和列表切片用法应该是相似而不同的，我用看列表切片的思维去看数组，导致我一直看不懂，哎。<br>

终于可以接下来看了。。
可视化决策边界的前置知识<br>
ylim(limits) 设置当前坐标轴或图表的 y 轴限制。将限制指定为窗体的两个元素向量 [ymin ymax], 其中 ymax 大于 ymin,<br>
语法：X,Y = numpy.meshgrid(x, y)<br>
输入的x，y，就是网格点的横纵坐标列向量（非矩阵）<br>
输出的X，Y，就是坐标矩阵。<br>
[meshgrid](https://blog.csdn.net/lllxxq141592654/article/details/81532855)<br>
linspace是Matlab中的均分计算指令，用于产生x1,x2之间的N点行线性的矢量。<br>
[predict_proba](https://blog.csdn.net/anan15151529/article/details/102632463)<br>
[reshape](https://blog.csdn.net/qq_28618765/article/details/78083895)<br>
[ravel](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html)<br>
可视化预测新样本的知识前置<br>
用一个箭头指向要注释的地方，再写上一段话的行为，叫做annotate<br>
[zhihu](https://zhuanlan.zhihu.com/p/32501335)<br>
训练样本，训练样本，我总觉得训练这两字怪怪的，哈哈<br>
<br>
en,我觉得这个边界线跟结尾函数的对称点差不多嘛~<br>

忽然对源代码有了点好奇，额，翻墙看了下是深渊，不止是坑那么简单的事了，先不管了,先掌握如何利用工具，而不探究工具本身，面向对象的好处不就是这样么，止住我的好奇心吧[?](https://scikit-learn.org/stable/modules/classes.html)<br>
看到这里，我觉得这里的预测，指的是利用算法，把需要结果的概率求出，便是这里的含义了。<br>
hhhh,到看花花的地方了，我也想一日看尽长安花的日子快点到来啊，要努力思考！<br>
基于花花的知识前置<br>
额，我觉得DSW里面讲的很详细了，每个都有备注哎，那么，这个画花花的人所要让我们知道的是啥呢？，不要只看别人让我看的，还要看他没有让我看的，才能收获更多<br>
一开始是用简单的数据来画图，这个是用较复杂的互相有联系的数据来画图，是使用数据的手法吗，组合数据来进行可视化，有个有意思的事情，数据浅拷贝，防止对于原始数据的修改。那有深拷贝？？老板来两斤生蚝，我要5成熟！！！<br>
我记得之前学列表引用也是一样的，新建一个列表，防止对原列表修改，是一样的手段。。<br>

箱型图 三维散点图 <br>
```
ax.scatter(iris_all_class0[:,0], iris_all_class0[:,1], iris_all_class0[:,2],label='setosa')
ax.scatter(iris_all_class1[:,0], iris_all_class0[:,1], iris_all_class0[:,2],label='versicolor')
ax.scatter(iris_all_class2[:,0], iris_all_class0[:,1], iris_all_class0[:,2],label='virginica')
```
现在知道意思了，前天敲代码的时候有点懵逼<br>
#### Step 5
我觉得这段话是重点，为了正确评估模型性能，将数据划分为训练集和测试集，并在训练集上训练模型，在测试集上验证模型性能。那么，如何实现呢，让我往下看。。
```

iris_target = data.target
iris_features = pd.DataFrame(data=data.data, columns=data.feature_names)
iris_features_part = iris_features.iloc[:100]
iris_target_part = iris_target[:100]
```
为啥第四行没有.iloc呢？？？
明天继续吧，没时间了，明天休息时间多~~~ 干巴爹<br>

### 第四天
在群里看到了我最近隐隐感受到还没凝聚成语言的话语。<br>
逻辑回归可以二分类，也可以多分类。他下面得到了三组参数，就证明做三分类的时候训练了三个分类器<br>
所以数据的作用是得出预测未来的参数咯~<br>
数据划分为两种，80%用于得出预测参数，20%用于验证准确性。<br>
嗯，应该是有一个数据转化为参数的流程可以具体归纳的，我看看啊，利用model_selection划分数据，然后用了fit将数据转化为预测参数，
clf.fit，这个是用逻辑回归的算法clf，那么是不是有用其他算法就有其他xxx.fit呢，哈哈，如何查看逻辑回归得出的参数这步我知道就好了，黑匣子黑到底，
预测居然把80%也搞进去了，这个有必要？如果算法已经被验证是正常的那么这步应该不用吧，直接代入那20%不就可以了？？<br>
在利用热力图对于结果进行可视化中，我没看到
```
train_predict = clf.predict(x_train)
print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_train,train_predict))
```
这两条代码对热力图的影响，是不是删除也不影响图呢？<br>
另外我搜了搜[model_selection](https://blog.csdn.net/qq_41861526/article/details/88617840?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-2.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-2.control） 加深了model_selection的split的理解。<br>
往上翻了下我才看出来二分和三分的代码区别，上面有个注释是这么说的，其对应的类别标签为，其中0，1，2分别代表
'setosa','versicolor',virginica'三种不同花的类别，每个类别数量那里也值得注意，2 50 1 50 0 50 ,也就解释了[:100]的意思了。<br>
二分类那里用的是iris_features_part<br>
三分类直接用iris_feature<br>
[混淆矩阵](https://baike.baidu.com/item/%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5/10087822?fr=aladdin)<br>
通过结果我们发现·····出现了一定的错误。我们从可视化的时候也可以发现，这里指的是上面那三维三点图吧，蓝色的点附近毛都没有，绿色和橙色点
有几乎重合的的，也就是边界模糊。<br>
##TASK 2
e，学代码先从敲代码开始
```
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn import datasets
from sklearn.navie_bayes import GaussianNB
from sklearn.model_selection import train_test_split

X,y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_tarin, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = GaussianNB(var_smoothing=1e-8)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = np.sun(y_test == y_pred) / X_test.shape[0]
print('Test Acc : %.3f'% acc)

y_proba = clf.predict_proba(X_test[:1])
print(clf.predict(X_test[:1]))
print('预计的概率值:', y_proba)


import random
import numpy as np 
from sklearn.naive_bayes import Categorica1NB
from sklearn.model_selection import train_test_split

rng = np.random.RandomState(1)
X = rng.randint(5, size=(600, 100))
y = np.array([1,2,3,4,5,6] * 100)
data = np.c_[X, y]
random.shuffle(data)
X = data[:,:-1]
y = data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=0)

clf = Categorica1NB(alpha=1)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print('Test Acc : %.3f % acc)

x = rng.randint(5, size= (1, 100))
print(clf.predict_proba(x))
print(clf.predict(x))
```
填个第三天结尾的疑问！
[jianshu](https://www.jianshu.com/p/732858f89a00)
[image](https://github.com/Sncef/Sncef.github.io/blob/main/photo/6.png)
[image](https://github.com/Sncef/Sncef.github.io/blob/main/photo/7.png)
iloc对二维数据的读取用，第三行是表格，第四行是列表，不用iloc来选择。。




