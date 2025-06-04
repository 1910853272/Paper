
'''
# lesson-1
# NumPy 部分 Numerical Py
#【例2-1】从列表创建数组。依次输入以下代码，并观察输出结果。
import numpy as np

# tensor 张量  数组

arr1=np.array( [1,2,3,4,5] )	# 从列表创建一维数组
print(arr1)

arr2=np.array( [ [1,2,3],[4,5,6] ] )	# 从列表创建二维数组
print(arr2)

arr3=np.array([ [ [1,2,3],[4,5,6] ],[[4,0,4],[1,6,8] ] ])
print(arr3)

arr4=np.array([[ [ [1,2,3],[4,5,6] ],[[4,0,4],[1,6,8] ] ]])

print(arr1.shape,arr2.shape,arr3.shape,arr4.shape)


#【例2-2】利用特定函数创建数组。依次输入以下代码，并观察输出结果。
import numpy as np

zeros_arr=np.zeros((3,3))		# 创建全为零的数组
ones_arr=np.ones((2,2))			# 创建全为一的数组
print(zeros_arr)

full_arr=np.full((2,3),7)	# 创建指定形状的常数数组
print(full_arr)

eye_arr=np.eye(3)				# 创建单位矩阵
print(eye_arr)

diag_arr=np.diag([1,2,3])		# 创建对角矩阵
print(diag_arr)


range_arr=np.arange(0,10,2)	# 创建一维数组，范围是[0,10)，步长为2
print(range_arr)

linspace_arr=np.linspace(0,1,5)
print(linspace_arr)								# 创建一个在[0,1]范围内均匀分布的数组，共5个元素



rand_arr=np.random.rand(2,3)	# 创建一个形状为(2,3)的随机数组 服从0-1均匀分布
print(rand_arr)
randn_arr=np.random.randn(2,3)		# 创建一个形状为(2,3)的随机数组  服从标准正态分布
print(randn_arr)


#【例2-3】数组的索引与切片应用示例。依次输入以下代码，并观察输出结果。
import numpy as np

arr=np.array([1,2,3,4,5])
# 一维数组的索引
print(arr[0])		# 输出：1
print(arr[-1])		# 输出：5
# 一维数组的切片
print(arr[1:4])		# 输出：[2 3 4]


arr=np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9]])
print(arr)

# 多维数组的索引
print(arr[0,0])			# 输出：1
print(arr[1,2])			# 输出：6
# 多维数组的切片
print(arr[0:2,1:3])		# 输出：[[2 3]
							#       [5 6]]


arr=np.array([1,2,3,4,5])
# 布尔索引
mask=arr>2
print(arr[mask])		# 输出：[3 4 5]


arr=np.array([1,2,3,4,5])
# 花式索引
indices=[0,2,4]
print(arr[indices])		# 输出：[1 3 5]

arr=np.array([1,2,3,4,5])
print(arr[:3])			# 输出：[1 2 3]
print(arr[2:])			# 输出：[3 4 5]
print(arr[::2])			# 输出：[1 3 5]

# 数组的型状变换
#【例2-4】数组的变换应用示例。依次输入以下代码，并观察输出结果。

import numpy as np					# 导入 NumPy 库，并使用别名 np
arr=np.arange(9)					# 创建一个包含0到8的一维数组
print(arr)


reshaped_arr=arr.reshape((3,3))	# 将一维数组重塑为3x3的二维数组
print(reshaped_arr)

flattened_arr=reshaped_arr.flatten()			# 将多维数组变为一维数组
print(flattened_arr)

arrT = np.random.rand(2,3)
print(arrT)

transposed_arr1=arrT.transpose()	# 将数组进行转置操作
print(transposed_arr1)

transposed_arr2=arrT.T				# 简洁方式，使用 .T 属性进行转置操作
print(transposed_arr2)

# axis 轴 维度demension
arr=np.array([1,2,3])

print(arr.shape)
expanded_arr1=arr[:,np.newaxis]
expanded_arr2=arr[np.newaxis,:]
print(expanded_arr1,expanded_arr1.shape)
print(expanded_arr2,expanded_arr2.shape)
# np.newaxis的作用就是在这一位置增加一个一维，这一位置指的是np.newaxis所在的位置
# 用途： 通常用它将一维的数据转换成一个矩阵，这样就可以与其他矩阵进行相乘。


# concat 拼接  add 加法

arr1=np.array([[1,2],[3,4]])
print(arr1)
arr2=np.array([[5,6]])
print(arr2)

concatenated_arr=np.concatenate((arr1,arr2),axis=0)
# axis=0:在第一维拼接：其他维度的需要是相同的维度； 行维度
# axis=1:在第二维拼接：其他维度的需要是相同的维度；  列维度
# axis=-1:最后一维拼接：其他维度的需要是相同的维度；  第三个维度
print(concatenated_arr)

#利用split()函数库将数组分裂为多个子数组。
arr = np.array([3,1,7])

subarrays=np.split(arr,3)
print(subarrays)

repeated_arr=np.repeat(arr,3)
print(repeated_arr)

tiled_arr=np.tile(arr,(2,3))
print(tiled_arr)

sorted_arr=np.sort(arr)
print(sorted_arr)

#【例2-5】基本运算示例。依次输入以下代码，并观察输出结果。
import numpy as np
arr1=np.array([1,2,3])
arr2=np.array([4,5,6])

sum_arr=arr1+arr2	# 加法运算，同np.add(arr1,arr2)
print(sum_arr)
diff_arr=arr1-arr2				# 减法运算，同np.subtract(arr1,arr2)
prod_arr=arr1*arr2				# 乘法运算，同np.multiply(arr1,arr2)
div_arr=arr1/arr2				# 除法运算，同np.divide(arr1,arr2)

arr=np.array([True,False,True])
arr2=np.array([False,True,False])

and_arr=arr & arr2				# 逻辑与，同np.logical_and(arr,arr2)
or_arr=arr | arr2				# 逻辑或，同np.logical_or(arr,arr2)
not_arr=~arr					# 逻辑非，同np.logical_not(arr)

arr=np.array([[1,2,3],[0,6,9]])

squared_arr=np.square(arr)		# 平方
sqrt_arr=np.sqrt(arr)
print(sqrt_arr)# 开方
exp_arr=np.exp(arr)				# 指数函数
log_arr=np.log(arr)				# 对数函数

'''






























# lesson-2
# pandas部分 两种数据结构 series与dataframe
# numpy
#【例2-6】Series的构造方法示例。依次输入以下代码，并观察输出结果。

# pip install package-name

import pandas as pd
data=[1,2,3,4,5]
series=pd.Series(data)	# 从列表创建Series
print(series)

import numpy as np
data_np=np.array([1,2,3,4,5])
series_np=pd.Series(data_np)				# 从NumPy数组创建Series

series_default_index=pd.Series(data)		# 使用默认索引

custom_index=['a','b','c','d','e']			# 自定义索引
series_custom_index=pd.Series(data,index=custom_index)		# 使用自定义索引

print(series[0])							# 通过位置访问数据，输出：1
print(series_custom_index['b'])			# 通过标签访问数据，输出：2

result=series*2								# 算术运算
result=np.sqrt(series)						# 应用数学函数
mask=series>3								# 布尔运算

# 创建带有缺失值的 Series
data_with_nan=[1,2,np.nan,4,5]
series_with_nan=pd.Series(data_with_nan)

mask_nan = series_with_nan.isnull()	# 检测缺失值
print(mask_nan)
series_filled = series_with_nan.fillna(0)	# 填充缺失值
print(series_filled)


series_with_name = pd.Series(data,name='MySeries')		# 设置名称

index=series.index							# 获取索引
dtype=series.dtype							# 获取数据类型
























# mpg数据
#【例2-7】创建DataFrame数据示例。依次输入以下代码，并观察输出结果。
#（1）创建DataFrame
import pandas as pd
import numpy as np

# 从字典创建 DataFrame
data={'Name':['Alice','Bob','Charlie'],
       'Age':[25,30,35],
       'Score':[90,85,88]}
df = pd.DataFrame(data)

# 从列表创建 DataFrame
data=[['Alice',25,90],['Bob',30,85],['Charlie',35,88]]
df=pd.DataFrame(data,columns=['Name','Age','Score'])
print(df)









# 从外部数据源读取，可以包含路径
df  =pd.read_csv(r'S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mpg.csv')
print(df)

#（2）查看DataFrame
print(df.head())			# 查看头部几行数据
print(df.tail())			# 查看尾部几行数据
print(df.shape)				# 查看数据的维度
print(df.columns)			# 查看列名
print(df.index)				# 查看索引
print(df.describe())		# 查看数据摘要统计信息
print(df.dtypes)			# 查看数据类型



#（3）数据访问
print(df['horsepower'])			# 访问列
print(df.iloc[2,1])			# 使用整数位置进行 行访问
print(df.loc[2])			# 使用标签进行 行访问
print(df.at[7,'acceleration'])		# 使用标签访问特定行列的值
print(df.iat[2,1])			# 使用整数位置访问特定行列的值

#（4）数据操作  增 删 改 查
df['mpg']=np.ones((398,1))					# 添加列
print(df.head())

df.drop('mpg',axis=1,inplace=True)	# 删除列
print(df.head())

df.at[0,'cylinders'] = 16							# 修改数据
print(df.head())

selected_data=df[df['cylinders']>3]				# 根据条件选择数据
df.sort_values(by='cylinders',ascending=False,inplace=True)		# 对数据进行排序
print(df.head())



#（5）数据分析   统计  均值 总和 最值 中位数 方差 标准差
df=pd.read_csv(r'S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mpg.csv')
print(df['horsepower'].mean())	# 计算平均值

grouped=df.groupby('mpg')
print(grouped.head())			# 分组统计

print(df['mpg'].sum())			# 总和
print(df['horsepower'].max())			# 最大值
print(df['mpg'].min())			# 最小值
print(df['mpg'].median())		# 中位数
























#【例2-8】数据类型分类示例。依次输入以下代码，并观察输出结果。
import pandas as pd

s=pd.Series([1,2,3,4,5])					# 创建一个整数Series
print(s.dtype)								# 输出：int64

s=pd.Series([1.0,2.5,3.7,4.2,5.9])			# 创建一个浮点数Series
print(s.dtype)		# 输出 float64

s=pd.Series(['apple','banana','grape','kiwi'])		# 创建一个字符串Series
print(s.dtype)		# 输出object

s=pd.Series([True,False,True,False])		# 创建一个布尔Series
print(s.dtype)								# 输出bool

# 创建一个日期时间 Series
s=pd.Series(['2024-03-10','2024-06-15','2025-02-11'])
print(s.dtype)


s=pd.to_datetime(s)
print(s.dtype)
# 输出datetime64[ns]
print(s[1]-s[0])

# 创建一个分类 Series
s=pd.Series(['male','female','female','male','female','1'])
s=s.astype('category')

print(s.dtype)		# 输出 category
print(s)

print('ended')




















#【例2-9】分类（categorical）数据类型示例。依次输入以下代码，并观察输出结果。
import pandas as pd

data=['A','B','C','C','B','C','D','A']
s=pd.Series(data)						# 创建一个Series
cat_series=pd.Categorical(s)			# 将Series转换为categorical类型

print(cat_series)

print(cat_series.categories)			# 查看分类的唯一值列表
print(cat_series.codes)					# 查看每个元素在分类中的位置索引

# 指定分类变量的顺序
cat_series=pd.Categorical(s,categories=['D','C','B','A'],ordered=True)
print(cat_series)
cat_series=cat_series.sort_values()	# 使用categorical类型进行排序
print(cat_series)















#【例2-10】数据类型转换示例。依次输入以下代码，并观察输出结果。
#（1）创建DataFrame
import pandas as pd
data={'Name':['Alice','Bob','Charlie'],
       'Age':[25,30,35],
       'Score':[90,88,88],
      'date':['2022-03-14','2023-08-11','2024-05-16']}
df=pd.DataFrame(data)
print(df)

df['Age']=df['Age'].astype(float)		# 将整数转换为浮点数
print(df['Age'])

# 将字符串转换为日期时间类型
df['date']=pd.to_datetime(df['date'])
print(df['date'])

# 将日期时间类型转换为字符串
df['date']=df['date'].astype(str)
print(df['date'])


# 以此类推
# 将分类数据类型转换为字符串类型
# df['category_column']=df['category_column'].astype(str)

# 将分类数据类型转换为整数类型
# df['category_column']=df['category_column'].astype(int)

# 将整数或字符串列转换为分类数据类型
df['Score']=df['Score'].astype('category')
print(df['Score'])


# 将布尔类型转换为整数类型
# df['bool_column']=df['bool_column'].astype(int)

# 将字符串列转换为分类数据类型并指定顺序
# df['string_column']=df['string_column'].astype('category',ordered=True,categories=['low','medium','high'])

print('ended!!!!!!!!!!!!!!!!!!!!!')


