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
data = load_iris()
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
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

X,y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = GaussianNB(var_smoothing=1e-8)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = np.sum(y_test == y_pred) / X_test.shape[0]
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
![image](https://github.com/Sncef/Sncef.github.io/blob/main/photo/6.png)
![image](https://github.com/Sncef/Sncef.github.io/blob/main/photo/7.png)
iloc对二维数据的读取用，第三行是表格，第四行是列表，不用iloc来选择。。

### 第五天
TASK 2 先粗后细吧,还是先搜我不懂的内容<br>
[warnings](https://blog.konghy.cn/2017/12/16/python-warnings/)<br>
[shuffle](https://www.runoob.com/python/func-number-shuffle.html)<br>
```
X = data[:,:-1]
y = data[:, -1]
```
第二行少了个： 看不懂啊<br>
需要计算的两个概率：条件概率，先验概率<br>
[贝叶斯先验性](https://zhuanlan.zhihu.com/p/136791364)<br>
极大似然估计，只是一种概率论在统计学的应用，它是参数估计的方法之一。说的是已知某个随机样本满足某种概率分布，但是其中具体的参数不清楚，参数估计就是通过若干次试验，观察其结果，利用结果推出参数的大概值。极大似然估计是建立在这样的思想上：已知某个参数能使这个样本出现的概率最大，我们当然不会再去选择其他小概率的样本，所以干脆就把这个参数作为估计的真实值<br>

https://www.jianshu.com/p/9c153d82ba2d<br>
###### 我们可以看出就是对每一个变量的多加了一个频数alpha。当alpha=0时，就是极大似然估计。通常取值alpha=1，这就是拉普拉斯平滑，这又叫做贝叶斯估计，主要时因为如果使用极大似然估计，如果某个特征值在训练数据中没有出现，这时候会出现概率为0的情况，导致整个估计都为0，因此引入贝叶斯估计。
学到现在。总感到有些地方不清晰，我想了想，是因为2.3里面的引入的数据集被隐藏了具体数据造成的，在TASK 1 中运用了数据查看的手段来看，这里没有，于是，我返回了TASK 1 重新学习查看数据的手段。<br>
###### 好像先要将数据利用Pandas转化为DataFrame格式才能用下面的查看手段，我直接用报了个错AttributeError: 'numpy.ndarray' object has no attribute 'head',但是具体是不是真的还要实践。。。
TASK 1给出的方法有<br>
利用.info()查看数据的整体信息 iris_features.head() <br>
进行简单的数据查看，我们可以利用.head().tail() iris_features.head() <br>
利用value_counts函数查看每个类别数量 pd.Series(iris_target).value_counts()<br>
对于特征进行一些统计描述 iris_featues.describe() <br>
额，今天就到这里了，留个悬念给明天。。剩下的一点时间看看能不能优化下空间哈！

### 第六天
尝试掌握数据查看的方法。<br>
额，不容易啊，一开始打多了一个字母，iris_features = pd.DataFrame(data=data.data, columns=data.feature_names)等于号左边的有s，右边的没有
。估计是一种命名习惯之类的，还在探索。学会了群里前辈给出的查看的方法：print(data.key()).<br>
这时候，我忽然发现了-------------
```
print(data.key())
dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])
```
可是，我没设定target,DESCR啊，为啥会在keys里出现，莫非~~~~ 只能对这五个进行数据查看？为毛X_train看不了，然后，我仔细看了有关X_train诞生的代码，额，好像是随机数，可能，每次，不一样，所以，没法子看，只需要，知道，是，十分之八，就，好，了。。。看来，我还缺少很多能彻底看懂这三个task的前置知识啊，得多看书了。。。<br>

### 第7天
TASK 3<br>
不知不觉时间已过半，第一天照常敲代码走起！！！<br>
```
demoshujuji knn

import numpy as np
import matplotlib.pylot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

k_list = [1, 3, 5, 8, 10, 15]
h = .02
camp_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
camp_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

plt.figure(ifgsize=(15,14))

for ind,k in enumerate(k_list):
    clf = KNeighborsClassifier(k):
    clf.fit(X, y)
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() +1
    y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() +1
    xx, yy = np.meshgrid(np.arrange(x_min, x_max, h),
                        (np.arrange(y_min, y_max, h))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = Z.reshape(xx.shape)
    
    plt.subplot(321+ind)
    plt.pcolormesh(xx, yy, Z, cmpa=camp_light)
    plt.scatter(X[:, 0], X[:, 1], c= y, cmap=cmap_bold.
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i)"% k)
plt.show()
```
```
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = dataset.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski")
clf.fit(X_train, y_train)

X_pred = clf.predict(X_test)
acc = sum(X_pred == y_test) / X_pred.shape[0]
print("预测的准确率ACC： %.3f" % acc)
```
```
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsRegressor
np.random.seed(0)
X = np.sort(5*np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()
y[::5] += 1*(0.5 - np.random.rand(8))

n_neighbors = [1, 3, 5, 8, 10, 40]
plt.figure(figsize=(10,20))
for i,k in enumerate(n_neighbors):
    clf = KNeighborsRegressor(n_neighbors=k, p=2, mietric="minkowski")
    clf.fit(X, y)
    y_ = clf.predict(T)
    plt.subplot(6, 1, i + 1)
    plt.scatter(X, y, color='red', label='data')
    plt.plot(T, y_, color='navy', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i)" % (k))
plt. tight_layout()
plt.show()
```
```
!wget https://tianchi-media.oss-cn-beijing.aliyuncs.com/DSW/3K/horse-colic.csv
!wget https://tianchi-media.oss-cn-beijing.aliyuncs.com/DSW/3K/horse-colic.names

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X = [[1, 2 np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]
imputer = KNNImputer(n_neighbors=2, metric='nan_euclidean')
imputer.fit_transform(X)

nan_edclidean_distances([[np.nan, 6, 5], [3, 4, 3]], [[3, 4, 3], [1, 2, np.nan], [8, 8, 7]])

input_file = './horse-colic.csv'
df_data = pd.read_csv(input_file, header= None, na_values='?')
data = df_data.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = adta[:, ix], data[:, 23'

for i in range(df_data.shape[1]):
    n_miss df_data[[i]].isnull().sum()
    perc = n_miss / df_data,shape[0] *100
    if n_miss.values[0] > 0:
        print('>Feat: %d, Missing: %d, Missing ratio: (%.2f%%)' % (i, n_miss, perc))

print('KNNImputer before Missing: %d' % sum(np.isnan(X).flatten()))
imputer = KNNImputer()
imputer.fit(X)
Xtrans = imputer.transform(X)
print('KNNImputer after Missing: %d' % sum(np.isnan(Xtrans).flatten()))

results = list()
strategis = [str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,15, 16, 20, 21]]
for s in strategies:
pipe = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=int(s))), ('model', KNeighborsClassifier())])
scores = []
for k in range(20):
    X_train, X_test, y_train, y_test = train_test_split(Xtrans, y, test_size=0.2)
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    scores.append(score)
results.append(np.array(scores))
print('>k: %s, Acc Mean: %.3f, Std: %.3f' % (s, np.mean(scores), np.std(scores)))
boxplot(results, labels= strategies. showmeans=True)
show()
```
### 第八天
将我不懂的列出来<br>
[enumerate](https://www.runoob.com/python/python-func-enumerate.html) 可是前面的ind,k什么意思？我还想不明白。。<br>
[subplot](https://www.jianshu.com/p/de223a79217a) 看到这个，我想IND应该是跟辅助排序有关的吧。。。<br>
根据6张图的变化，可以看出画布颜色的变化与K值有关，但是是怎么影响的，为啥较小的k值就相当于用较小的领域中的训练实例进行预测？好想把函数中的代码挖出来看。<br>
[figsize](https://www.cnblogs.com/lijunjie9502/p/10327151.html)<br>
hhh,好像找到了！！！[edgecolor](https://www.cnblogs.com/OliverQin/p/7965435.html) 这里的K等价于edgecolor，这个值越大，边界轮廓越光滑！！！！，然后作者借用这特性，来进行K分类。。
