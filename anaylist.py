import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

data_train= pd.read_csv('./train.csv')
#print(data_train)
#print(data_train.info())
#print(data_train.describe())
plt.rcParams['font.sans-serif']=['SimHei'] #用來正常顯示中文標籤
plt.rcParams['axes.unicode_minus']=False #用來正常顯示負號
fig=plt.figure()
fig.set(alpha=0.2)
plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title(u'獲救情形（1為獲救）')
plt.ylabel(u'人數')

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.title(u'乘客等級分佈')
plt.ylabel(u'人數')

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u'人數')
plt.grid(b=True, which='major', axis='y')
plt.title(u'按年齡看獲救分佈(1為獲救)')

plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass==1].plot(kind='kde')
data_train.Age[data_train.Pclass==2].plot(kind='kde')
data_train.Age[data_train.Pclass==3].plot(kind='kde')
plt.xlabel(u'年齡')
plt.ylabel(u'密度')
plt.title(u'各等級的乘客年齡分佈')
plt.legend((u'頭等艙',u'2等艙',u'3等艙'),loc='best')

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u'各登船口岸上船人數')
plt.ylabel(u'人數')
plt.show()


