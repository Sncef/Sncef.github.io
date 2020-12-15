## MY&PY天池机器学习
### 第一天
额，对于其他人来说可能是直接就开始学内容了，可是我啊，是不是选错博客了，我选Github作为写博客的地方，但是却从未用过。
没想到用好Github也是一件不容易的事情呢，本来想下载一个模板再写的，可是几经波折后发现不是一两天内能够弄好的事情，能够用好Github这个工具也相当于一个训练营的内容了，嗯，一边学训练营的内容一边搞博客空间好了哈哈哈，前期博客就走简单路线了。 
TASK 1 的代码
```
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

```
先计划把代码全部敲一次，再细思其含义，但是没时间了，只能先敲一半，明天敲完，上班！第一天结束！！！


