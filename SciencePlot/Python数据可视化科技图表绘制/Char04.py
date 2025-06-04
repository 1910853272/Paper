#四、类别比较数据可视化

#  类别比较数据可视化是一种用于呈现和分析离散或分类变量的数据可视化方法。类别数
#据（也称为离散数据）是一种具有有限数量的可能取值的数据类型，它表示了不同的类别、
#类型或标签。类别数据可视化的目标是展示不同类别之间的关系、频率分布以及类别的比较，
#帮助更好地理解数据中的模式、趋势和关联信息。


# 1 柱状图 包括 单一、分组、堆积、百分比、均值、不等宽，有序
#     柱状图（Bar Chart）是一种用于显示不同类别或组之间的比较或分布情况。它由一系列
# 垂直的矩形柱组成，每个柱子的高度表示对应类别或组的数值大小。当柱水平排列时又称为
# 条形图。

# 要素：
# （1）X 轴（水平轴）：用于表示不同的类别或组。每个柱子通常对应于一个类别。
# （2）Y 轴（垂直轴）：用于表示数值的大小或数量。Y 轴可以表示各种度量，如计数、
# 百分比、频率等，具体取决于数据类型和分析目的。
# （3）柱子（矩形条）：每个柱子的高度代表相应类别或组的数值大小。柱子的宽度可以
# 是固定的或可以调整。
# （4）填充颜色：柱子可以使用不同的填充颜色来区分不同的类别或组。颜色选择可以
# 根据需要进行调整，以提高可读性或强调特定的类别。

# 涉及到的库：Matplotlib、Seaborn等

#【例4-1】采用5个类别及对应的值创建单一柱状图。输入代码如下：
import pandas as pd  # 导入pandas库并简写为pd
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块并简写为plt

# 自定义数据集
data = pd.DataFrame({'category': ["A", "B", "C", "D", "E"],
                     'value': [10, 15, 7, 12, 8]})  # 创建一个包含类别和值的DataFrame
# 查看数据结构
print("数据结构：")
print(data)

# 创建单一柱状图
plt.figure(figsize=(6, 4))  # 创建图形对象，并设置图形大小
plt.bar(data['category'], data['value'], color='steelblue')
# 绘制柱状图，指定x轴为类别，y轴为值，柱状颜色为钢蓝色
plt.xlabel('Category')  # 设置x轴标签
plt.ylabel('Value')  # 设置y轴标签
plt.title('Single Bar Chart')  # 设置图表标题

# 添加网格线，采用虚线，设置为灰色，透明度为0.5
plt.grid(linestyle='-', color='gray', alpha=0.5)
plt.show()












#【例4-2】创建包含5个类别和4个对应的数值列的分组柱状图。输入代码如下：
import pandas as pd  # 导入pandas库并简写为pd
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块并简写为plt

# 自定义一个包含多列数据的数据框DataFrame，包含类别和多列值
data = pd.DataFrame({'category': ["A", "B", "C", "D", "E"],
                     'value1': [10, 15, 7, 12, 8], 'value2': [6, 9, 5, 8, 4],
                     'value3': [3, 5, 2, 4, 6], 'value4': [9, 6, 8, 3, 5]})
# 查看数据框
print("Data Structure：")
print(data)  # 输出如图43所示

# 创建分组柱状图
data.plot(x='category', kind='bar', figsize=(6, 4))
# 使用DataFrame的plot方法绘制分组柱状图
# 指定x轴为'category'列，图表类型为'bar'，图形大小为(6,4)
plt.xlabel('Category')  # 设置x轴标签
plt.xticks(rotation=0)  # 旋转x轴文本，使其水平显示
plt.ylabel('Value')  # 设置y轴标签
plt.title('Grouped Bar Chart')  # 设置图表标题
plt.legend(title='Values')  # 添加图例，并设置标题为'Values'
plt.show()









#【例4-3】创建包含5个类别和4个对应的数值列的堆积柱状图。输入代码如下：
# 续上例，将'category' 列设置为索引，并创建堆积柱状图

data.set_index('category').plot(kind='bar', stacked=True, figsize=(6, 4))
# 使用DataFrame的plot方法绘制堆积柱状图
# 设置索引为'category'列，图表类型为'bar'，堆积模式为True，图形大小为(6,4)

plt.xlabel('Category')  # 设置x轴标签
plt.ylabel('Value')  # 设置y轴标签
plt.title('Stacked Bar Chart')  # 设置图表标题
plt.xticks(rotation=0)  # 旋转x轴文本，使其水平显示

# 添加图例，并设置标题为'Values'，并放置在图的右侧
plt.legend(title='Values', loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()














#【例4-4】创建包含5个类别和4个对应的数值列的百分比柱状图。输入代码如下：
# 续上例，创建百分比柱状状图
# 复制数据集到新的DataFrame以便进行百分比计算
data_percentage = data.copy()

# 计算每个数值列的百分比，除以每行的总和并乘以100
data_percentage.iloc[:, 1:] = data_percentage.iloc[:, 1:].div(
    data_percentage.iloc[:, 1:].sum(axis=1), axis=0) * 100

data_percentage.set_index('category').plot(kind='bar',
                                           stacked=True, figsize=(6, 4))
# 创建百分比堆叠柱状图，设置索引为'category'列，
# 图表类型为'bar'，堆积模式为True，图形大小为(6,4)

plt.xlabel('Category')  # 设置x轴标签
plt.ylabel('Percentage')  # 设置y轴标签
plt.title('Percentage Stacked Bar Chart')  # 设置图表标题
plt.xticks(rotation=0)  # 旋转x轴文本，使其水平显示

# 添加图例，并设置标题为'Values'，并放置在图的右侧
plt.legend(title='Values', loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()




















#【例4-5】创建包含5个类别和4个对应的数值列的均值柱状图。输入代码如下：
import pandas as pd  # 导入pandas库并简写为pd
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块并简写为plt
import seaborn as sns  # 导入seaborn库并简写为sns

# 创建创建一个包含类别、值和标准差的DataFrame数据集
data = pd.DataFrame({'category': ["A", "B", "C", "D", "E"],
                     'value': [10, 15, 7, 12, 8], 'std': [1, 2, 1.5, 1.2, 2.5]})

# 计算每个类别的均值和标准差
mean_values = data['value']
std_values = data['std']

colors = sns.color_palette("Set1", n_colors=len(data))  # 创建颜色调色板
# 创建均值柱状图
plt.figure(figsize=(6, 4))  # 创建图形对象，并设置图形大小
bars = plt.bar(data['category'], mean_values, color=colors)
# 绘制柱状图，指定x轴为类别，y轴为均值，柱状颜色为颜色调色板中的颜色

# 添加误差线
for i, (bar, std) in enumerate(zip(bars, std_values)):
    plt.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 # 在柱状图的中心位置添加误差线
                 yerr=std, fmt='none', color='black', ecolor='gray',
                 # 设置误差线的样式和颜色
                 capsize=5, capthick=2)  # 设置误差线的帽子大小和线宽
# 添加标题和标签
plt.xlabel('Category')  # 设置x轴标签
plt.ylabel('Mean Value')  # 设置y轴标签
plt.title('Mean Bar Chart with Error Bars')  # 设置图表标题

# 设置网格线的样式、颜色和透明度
plt.grid(axis='both', linestyle='-', color='gray', alpha=0.5)
plt.show()




















#【例4-6】创建包含5个类别和5个对应的值及宽度值的不等宽柱状图。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt

# 创建数据集
data = pd.DataFrame({'category': ["A", "B", "C", "D", "E"],
                     'value': [10, 15, 7, 12, 8],
                     'width': [0.8, 0.4, 1.0, 0.5, 0.9]})
print("数据结构："), print(data)  # 查看数据框，如图48所示

# 自定义颜色列表，每个柱子使用不同的配色
colors = ['red', 'green', 'blue', 'orange', 'purple']
# 创建不等宽柱状图
plt.figure(figsize=(6, 4))
for i in range(len(data)):
    plt.bar(data['category'][i], data['value'][i],
            width=data['width'][i], color=colors[i])

# 添加标题和标签
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Unequal Width Bar Chart')

# 设置网格线
plt.grid(axis='both', linestyle='-', color='gray', alpha=0.5)
plt.show()





























#【例4-7】基于mpg_ggplot2.csv数据集创建有序柱状图，以显示不同制造商制造的汽车的城市里程。
# 并在图表上方添加度量标准的值。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

df_raw = pd.read_csv(r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mpg_ggplot2.csv")  # 读取原始数据
# 按制造商分组，并计算每个制造商的平均城市里程
df = df_raw[['cty', 'manufacturer']].groupby('manufacturer').apply(
    lambda x: x.mean())

# 按城市里程排序数据
df.sort_values('cty', inplace=True)  # 按城市里程排序数据
df.reset_index(inplace=True)  # 重置索引

# 绘图
# 创建图形和坐标轴对象
fig, ax = plt.subplots(figsize=(10, 6), facecolor='white', dpi=80)

# 使用vlines绘制垂直线条，代表城市里程
ax.vlines(x=df.index, ymin=0, ymax=df.cty, color='firebrick',
          alpha=0.7, linewidth=20)

# 添加文本注释
# 在每个条形的顶部添加数值标签
for i, cty in enumerate(df.cty):
    ax.text(i, cty + 0.5, round(cty, 1), horizontalalignment='center')

# 设置标题、标签、刻度和y轴范围
ax.set_title('Bar Chart for Highway Mileage',
             fontdict={'size': 18})  # 设置标题
ax.set(ylabel='Miles Per Gallon', ylim=(0, 30))  # 设置y轴标签和范围
plt.xticks(df.index, df.manufacturer.str.upper(), rotation=60,
           horizontalalignment='right', fontsize=8)  # 设置x轴标签

# 添加补丁以为X轴标签着色
# 创建两个补丁对象，用于着色X轴标签的背景
p1 = patches.Rectangle((.57, -0.005), width=.33, height=.13, alpha=.1,
                       facecolor='green', transform=fig.transFigure)  # 创建绿色补丁
p2 = patches.Rectangle((.124, -0.005), width=.446, height=.13, alpha=.1,
                       facecolor='red', transform=fig.transFigure)  # 创建红色补丁
# 将补丁对象添加到图形上
fig.add_artist(p1)  # 添加绿色补丁
fig.add_artist(p2)  # 添加红色补丁
plt.show()





























# 2 条形图
# 条形图也是一种常用的数据可视化图表，用于显示不同类别或组之间的比较或分布情况。
# 与柱状图类似，条形图使用水平或垂直的矩形条（条形）来表示数据。

# 要素：
# （1）Y 轴（垂直轴）：用于表示不同的类别或组。每个条形通常对应于一个类别。
# （2）X 轴（水平轴）：用于表示数值的大小或数量。X 轴可以表示各种度量，如计数、
# 百分比、频率等，具体取决于数据类型和分析目的。
# （3）条形（矩形条）：每个条形的长度代表相应类别或组的数值大小。条形的宽度可以
# 是固定的或可以调整。
# （4）填充颜色：条形可以使用不同的填充颜色来区分不同的类别或组。颜色选择可以
# 根据需要进行调整，以提高可读性或强调特定的类别。


#【例4-9】基于mpg_ggplot2.csv数据集，对制造商进行分组统计，并绘制条形图，
# 展示了每个制造商的汽车数量。条形图的颜色随机选择，并为其添加数值标签。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt
import random

df_raw = pd.read_csv("S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mpg_ggplot2.csv")  # 导入数据
# 准备数据
df = df_raw.groupby('manufacturer').size().reset_index(name='counts')
# 按制造商分组并计算每个制造商的数量
n = df['manufacturer'].unique().__len__() + 1  # 获取唯一制造商的数量
all_colors = list(plt.cm.colors.cnames.keys())  # 获取所有可用的颜色
random.seed(100)  # 设置随机种子，确保每次运行生成的颜色相同
c = random.choices(all_colors, k=n)  # 从颜色列表中随机选择 n 个颜色

# 绘制条形图
plt.figure(figsize=(10, 6), dpi=80)  # 设置图形大小
plt.barh(df['manufacturer'], df['counts'], color=c,
         height=.5)  # 绘制水平条形图，X轴为counts，Y轴为manufacturer
for i, val in enumerate(df['counts'].values):  # 遍历每个条形并在右侧添加数值标签
    plt.text(val, i, float(val), horizontalalignment='left',
             verticalalignment='center', fontdict={'fontweight': 500, 'size': 12})

# 添加修饰
plt.gca().invert_yaxis()  # 反转Y轴，确保顺序正确显示
plt.title("Number of Vehicles by Manufacturers",
          fontsize=18)  # 设置标题和字体大小
plt.xlabel('# Vehicles')  # 设置x轴标签
plt.xlim(0, 45)  # 设置x轴的范围
plt.show()






















#【例4-10】基于mpg_ggplot2.csv数据集，创建发散条形图展示汽车的里程数据，
# 其中条形的颜色根据数据的标准化值而变化，正值使用绿色，负值使用红色。输入代码如下：
#说明：散型条形图适应于根据单个指标查看项目的变化情况，并可视化此差异的顺序和数量。
# 它有助于快速区分数据中组的性能，并且非常直观，并且可以立即传达这一点。
# 导入必要的库
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mtcars1.csv")  # 读取数据

# 提取'mpg'列作为x变量，并计算其标准化值
x = df.loc[:, ['mpg']]
df['mpg_z'] = (x - x.mean()) / x.std()
# 根据'mpg_z'列的值确定颜色
df['colors'] = ['red' if x < 0 else 'green' for x in df['mpg_z']]

df.sort_values('mpg_z', inplace=True)  # 根据'mpg_z'列的值对数据进行排序
df.reset_index(inplace=True)  # 重置索引

# 绘制图形 ①
plt.figure(figsize=(10, 8), dpi=80)
plt.hlines(y=df.index, xmin=0, xmax=df.mpg_z, color=df.colors,
           alpha=0.4, linewidth=5)

# 图形修饰
plt.gca().set(ylabel='$Model$', xlabel='$Mileage$')  # 设置y轴和x轴标签
plt.yticks(df.index, df.cars, fontsize=12)  # 设置y轴刻度标签和字体大小
plt.title('Diverging Bars of Car Mileage',
          fontdict={'size': 20})  # 设置标题和字体大小
plt.grid(linestyle='--', alpha=0.5)  # 添加网格线
plt.show()


























#【例4-11】基于上例，在每个条形上添加了标签来表示'mpg_z'的值

plt.figure(figsize=(10, 8), dpi=80)
plt.hlines(y=df.index, xmin=0, xmax=df.mpg_z)

# 在条形上添加标签
for x, y, tex in zip(df.mpg_z, df.index, df.mpg_z):
    t = plt.text(x, y, round(tex, 2),
                 horizontalalignment='right' if x < 0 else 'left',
                 verticalalignment='center',
                 fontdict={'color': 'red' if x < 0 else 'green', 'size': 12})

# 图形修饰
plt.gca().set(ylabel='$Model$', xlabel='$Mileage$')  # 设置y轴和x轴标签
plt.yticks(df.index, df.cars, fontsize=12)  # 设置y轴刻度标签和字体大小
plt.title('Diverging Bars of Car Mileage',
          fontdict={'size': 20})  # 设置标题和字体大小
plt.grid(linestyle='--', alpha=0.5)  # 添加网格线
plt.xlim(-2.5, 2.5)  # 设置x轴范围
plt.show()


















# 3 棒棒糖图
#     棒棒糖图（Lollipop Chart），也称为火柴棒图（Stick Chart）、标志线图（Flag Chart）、茎
# 图（Stem Chart），是一种用于可视化数据的图表类型，结合了柱状图和折线图的元素。它以
# 一条垂直线（“棒棒糖”）和一个标记点（“棒棒糖头”）的形式来表示数据的分布和取值。
# 在棒棒糖图中，通常使用垂直线段表示数值变量，而水平的点表示分类变量。数值变量
# 可以是平均值、中位数或其他统计量，而分类变量则表示不同的类别或分组。
# 棒棒糖图的优点是可以同时显示数据的数值和范围，通过垂直线和标记点的组合，使得
# 数据的分布和差异更直观地呈现。它特别适用于比较多个类别或分组的数据，并突出显示数
# 据的关键数值。
# 【注意】：棒棒糖图适用于表示有序的、离散的数据集，而不适用于表示连续的数据。当数
# 据集较大或类别较多时，棒棒糖图可能会显得拥挤和混乱，因此在使用时应根据数据的特点
# 和数量进行调整，以确保图表的可读性和准确性。


#【例4-12】使用Matplotlib库创建不同的图形来展示如何绘制和定制棒棒糖图。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
np.random.seed(19781101)  # 固定随机种子，以便结果可复现
values = np.random.uniform(size=40)  # 生成40个0到1之间的随机数
positions = np.arange(len(values))  # 生成与values长度相同的位置数组

plt.figure(figsize=(10, 6))  # 创建图形窗口大小

# 绘制没有标记的图形
plt.subplot(2, 2, 1)  # 创建一个2x2的子图矩阵，并选择第1个子图
plt.stem(values, markerfmt=' ')  # 绘制棒棒糖图，没有标记
plt.title("No Markers")  # 设置子图标题

# 改变颜色、形状、大小和边缘
plt.subplot(2, 2, 2)  # 选择第2个子图
(markers, stemlines, baseline) = plt.stem(values)  # 获取棒棒糖图的组件
plt.setp(markers, marker='D', markersize=6,
         markeredgecolor="orange", markeredgewidth=2)  # 设置标记属性
plt.title("Custom Markers")  # 设置子图标题

# 绘制没有标记的图形（水平展示）
plt.subplot(2, 2, 3)  # 选择第3个子图
plt.hlines(y=positions, xmin=0, xmax=values, color='skyblue')  # 绘制水平线
plt.plot(values, positions, ' ')  # 绘制数据点
plt.title("Horizontal No Markers")  # 设置子图标题

# 改变颜色、形状、大小和边缘进行水平展示
plt.subplot(2, 2, 4)  # 选择第4个子图
plt.hlines(y=positions, xmin=0, xmax=values, color='skyblue')  # 绘制水平线
plt.plot(values, positions, 'D', markersize=6,
         markeredgecolor="orange", markerfacecolor="orange",
         markeredgewidth=2)  # 绘制数据点，并设置属性
plt.title("Horizontal Custom Markers")  # 设置子图标题

plt.tight_layout()  # 自动调整子图布局
plt.show()

























#【例4-13】基于mpg_ggplot2.csv数据集，绘制棒棒糖图显示每个制造商的平均城市里程。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt

df_raw = pd.read_csv(r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mpg_ggplot2.csv")  # 读取原始数据
# 按制造商分组，并计算每个制造商的平均城市里程
df = df_raw[['cty', 'manufacturer']].groupby('manufacturer').apply(
    lambda x: x.mean())

# 按城市里程排序数据
df.sort_values('cty', inplace=True)
df.reset_index(inplace=True)

# 绘图
fig, ax = plt.subplots(figsize=(12, 8), dpi=80)

# 使用vlines绘制垂直线条，代表城市里程的起始点
ax.vlines(x=df.index, ymin=0, ymax=df.cty, color='firebrick',
          alpha=0.7, linewidth=2)

# 使用scatter绘制lollipop的圆点
ax.scatter(x=df.index, y=df.cty, s=75, color='firebrick', alpha=0.7)

# 设置标题、标签、刻度和y轴范围
ax.set_title('Lollipop Chart for Highway Mileage',
             fontdict={'size': 20})  # 设置标题
ax.set_ylabel('Miles Per Gallon')  # 设置y轴标签
ax.set_xticks(df.index)  # 设置x轴刻度位置
ax.set_xticklabels(df.manufacturer.str.upper(), rotation=60,
                   fontdict={'horizontalalignment': 'right', 'size': 12})  # 设置x轴刻度标签
ax.set_ylim(0, 30)  # 设置y轴范围

# 添加注释
# 使用for循环遍历DataFrame的每一行，并在每个lollipop的顶部添加城市里程的数值
for row in df.itertuples():
    ax.text(row.Index, row.cty + 0.5, s=round(row.cty, 2),
            horizontalalignment='center', verticalalignment='bottom',
            fontsize=14)
plt.show()





















#【例4-14】绘制带基线的棒棒糖图示例一。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.linspace(0, 2 * np.pi, 100)  # 生成100个从0到2π的等间隔数据
y = np.sin(x) + np.random.uniform(size=len(x)) - 0.2
# 根据正弦函数生成y值，并添加一些随机噪声
my_color = np.where(y >= 0, 'orange', 'skyblue')  # 根据y的正负确定颜色

plt.vlines(x=x, ymin=0, ymax=y, color=my_color, alpha=0.4)  # 绘制垂直柱状图
plt.scatter(x, y, color=my_color, s=1, alpha=1)  # 绘制散点图

# 添加标题和坐标轴标签
plt.title("Evolution of the value of ...", loc='left')  # 设置标题左对齐
plt.xlabel('Value of the variable')  # x轴标签
plt.ylabel('Group')  # y轴标签
plt.show()




























#【例4-15】绘制带基线的棒棒糖图示例二。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
np.random.seed(19781101)  # 固定随机种子，以便结果可复现
values = np.random.uniform(size=80)  # 生成80个0到1之间的随机数
positions = np.arange(len(values))  # 生成与values长度相同的位置数组

plt.figure(figsize=(10, 6))  # 创建图形窗口大小

# 使用`bottom`参数自定义位置
plt.subplot(2, 2, 1)  # 创建一个2x2的子图矩阵，并选择第1个子图
plt.stem(values, markerfmt=' ', bottom=0.5)  # 绘制棒棒糖图，设置基线位置
plt.title("Custom Bottom")  # 设置子图标题

# 隐藏基线
plt.subplot(2, 2, 2)  # 选择第2个子图
(markers, stemlines, baseline) = plt.stem(values)  # 获取stem图的组件
plt.setp(baseline, visible=False)  # 隐藏基线
plt.title("Hide Baseline")  # 设置子图标题

# 隐藏基线-第二种方法
plt.subplot(2, 2, 3)  # 选择第3个子图
plt.stem(values, basefmt=" ")  # 绘制棒棒糖图，设置基线格式为空
plt.title("Hide Baseline-Method 2")  # 设置子图标题

# 自定义基线的颜色和线型
plt.subplot(2, 2, 4)  # 选择第4个子图
(markers, stemlines, baseline) = plt.stem(values)  # 获取棒棒糖图的组件
plt.setp(baseline, linestyle="-", color="grey",
         linewidth=6)  # 设置基线的颜色、线型和线宽
plt.title("Custom Baseline Color and Style")  # 设置子图标题

plt.tight_layout()  # 自动调整子图布局
plt.show()






















#【例4-16】绘制带标记的棒棒糖图。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

df = pd.read_csv(r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mtcars1.csv")  # 读取数据

# 提取'mpg'列作为x变量，并计算其标准化值
x = df.loc[:, ['mpg']]
df['mpg_z'] = (x - x.mean()) / x.std()
df['colors'] = 'black'  # 设置所有点的颜色为黑色

# 为'Fiat X1-9'设置不同的颜色
df.loc[df.cars == 'Fiat X1-9', 'colors'] = 'darkorange'

# 根据'mpg_z'列的值对数据进行排序
df.sort_values('mpg_z', inplace=True)
df.reset_index(inplace=True)

plt.figure(figsize=(14, 12), dpi=80)  # 绘制图形
plt.hlines(y=df.index, xmin=0, xmax=df.mpg_z, color=df.colors,
           alpha=0.4, linewidth=1)  # 绘制水平线

# 绘制散点图，并为'Fiat X1-9'设置不同的大小
plt.scatter(df.mpg_z, df.index, color=df.colors,
            s=[600 if x == 'Fiat X1-9' else 300 for x in df.cars], alpha=0.6)

plt.yticks(df.index, df.cars)  # 设置y轴刻度标签
plt.xticks(fontsize=12)  # 设置x轴刻度字体大小
# 添加注释
plt.annotate('Mercedes Models', xy=(0.0, 11.0), xytext=(1.0, 11),
             xycoords='data', fontsize=15, ha='center', va='center',
             bbox=dict(boxstyle='square', fc='firebrick'),
             arrowprops=dict(arrowstyle='-[,widthB=2.0,lengthB=1.5',
                             lw=2.0, color='steelblue'), color='white')
# 添加补丁
p1 = patches.Rectangle((-2.0, -1), width=0.3, height=3, alpha=0.2,
                       facecolor='red')
p2 = patches.Rectangle((1.5, 27), width=0.8, height=5, alpha=0.2,
                       facecolor='green')
plt.gca().add_patch(p1)
plt.gca().add_patch(p2)

# 图形修饰
plt.title('Diverging Bars of Car Mileage',
          fontdict={'size': 20})  # 设置标题
plt.grid(linestyle='--', alpha=0.5)  # 添加网格线
plt.show()


























#【例4-17】利用自创数据绘制哑铃图示例。输入如下代码。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 创建一个DataFrame
value1 = np.random.uniform(size=20)  # 生成第一组随机数
value2 = value1 + np.random.uniform(size=20) / 4
# 生成第二组随机数，基于第一组数据并加上一定随机量
df = pd.DataFrame({
    'group': list(map(chr, range(65, 85))),  # 创建从A到T的组标签
    'value1': value1,
    'value2': value2
})

# 按照第一组值的大小对DataFrame进行排序
ordered_df = df.sort_values(by='value1')
my_range = range(1, len(df.index) + 1)  # 创建一个范围，用于y轴坐标

# 使用hlines函数绘制水平线图
plt.hlines(y=my_range, xmin=ordered_df['value1'],
           xmax=ordered_df['value2'],
           color='grey', alpha=0.4)  # 绘制水平线
plt.scatter(ordered_df['value1'], my_range, color='skyblue', alpha=1,
            label='value1')  # 绘制value1的散点图
plt.scatter(ordered_df['value2'], my_range, color='green', alpha=0.4,
            label='value2')  # 绘制value2的散点图
plt.legend()  # 显示图例

# 添加标题和坐标轴名称
plt.yticks(my_range, ordered_df['group'])  # 设置y轴标签和坐标
plt.title("Comparison of the value 1 and the value 2", loc='left')
# 设置标题
plt.xlabel('Value of the variables')  # 设置x轴标签
plt.ylabel('Group')  # 设置y轴标签

plt.show()




























#【例4-18】基于health.csv数据集绘制哑铃图。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  # 导入线段模块

df = pd.read_csv("S:\Desktop_new\caxbook_python\python_plot_202405\PyData\health.csv")  # 导入数据

# 按 2014 年的数据进行排序
df.sort_values('pct_2014', inplace=True)
df.reset_index(inplace=True)


# 绘制线段的函数
def newline(p1, p2, color='black'):
    ax = plt.gca()  # 获取当前坐标轴
    l = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]],
                      color='skyblue')  # 创建线段对象
    ax.add_line(l)  # 添加线段到坐标轴
    return l


# 创建图形和轴
fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor='#f7f7f7', dpi=80)

# 绘制垂直的参考线
# 绘制垂直的参考线，用于标记百分比位置
ax.vlines(x=.05, ymin=0, ymax=26, color='black', alpha=1,
          linewidth=1, linestyles='dotted')
ax.vlines(x=.10, ymin=0, ymax=26, color='black', alpha=1,
          linewidth=1, linestyles='dotted')
ax.vlines(x=.15, ymin=0, ymax=26, color='black', alpha=1,
          linewidth=1, linestyles='dotted')
ax.vlines(x=.20, ymin=0, ymax=26, color='black', alpha=1,
          linewidth=1, linestyles='dotted')

# 绘制 2013 年和 2014 年的点
# 使用散点图绘制 2013 年和 2014 年的数据点
ax.scatter(y=df['index'], x=df['pct_2013'], s=50, color='#0e668b',
           alpha=0.7, label='2013')  # 绘制 2013 年数据点
ax.scatter(y=df['index'], x=df['pct_2014'], s=50, color='#a3c4dc',
           alpha=0.7, label='2014')  # 绘制 2014 年数据点

# 绘制线段
# 使用for循环遍历DataFrame的每一行，并绘制相应的线段
for i, p1, p2 in zip(df['index'], df['pct_2013'], df['pct_2014']):
    newline([p1, i], [p2, i])  # 调用函数绘制线段

# 图形修饰
# 设置图形的背景颜色和标题
ax.set_facecolor('#f7f7f7')
ax.set_title("Dumbbell Chart:Pct Change-2013 vs 2014",
             fontdict={'size': 16})  # 设置标题
ax.set(xlim=(0, .25), ylim=(-1, 27), ylabel='Index',
       xlabel='Percentage')  # 设置坐标轴的范围和标签
ax.set_xticks([.05, .1, .15, .20])  # 设置x轴刻度位置
ax.set_xticklabels(['5%', '10%', '15%', '20%'])  # 设置x轴刻度标签
ax.legend()  # 显示图例
plt.show()





























# 4 包点图
#     包点图（Dot Plot）是一种用来展示数据分布的图表类型，它使用点来表示数据点的位置
# 和数量。在包点图中，每个数据点都用一个小圆点或小方块来表示，通常是在一条水平或垂
# 直的轴上进行排列。数据点的位置表示其数值，而数据点的数量则可以通过点的大小、颜色
# 或形状来表示。
#     包点图通常用于展示分类数据的分布情况，特别是当数据量较少或数据点之间存在重叠
# 时。通过将数据点以点的形式直接绘制在轴上，包点图可以清晰地展示数据的集中程度和分
# 布形态，同时也能够直观地比较不同类别之间的差异。
#     包点图的优点包括简洁直观、易于理解和阅读，同时也可以有效地展示异常值和集中趋
# 势。然而，当数据点数量较大或重叠较多时，包点图可能会变得混乱不清，此时可以考虑使
# 用其他类型的图表来更好地展示数据。


#【例4-19】利用包点图显示不同制造商的平均城市里程。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt

df_raw = pd.read_csv("S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mpg_ggplot2.csv")  # 读取数据

# 按制造商分组，并计算每个制造商的平均城市里程
df = df_raw[['cty', 'manufacturer']].groupby('manufacturer').apply(
    lambda x: x.mean())

# 按城市里程排序数据
df.sort_values('cty', inplace=True)
df.reset_index(inplace=True)

fig, ax = plt.subplots(figsize=(8, 5), dpi=80)  # 绘图

# 使用hlines绘制水平线条，代表每个制造商
ax.hlines(y=df.index, xmin=11, xmax=26, color='gray', alpha=0.7,
          linewidth=1, linestyles='dashdot')

# 使用scatter绘制点，点的位置表示城市里程
ax.scatter(y=df.index, x=df.cty, s=75, color='firebrick', alpha=0.7)

# 设置标题、标签、刻度和x轴范围
ax.set_title('Dot Plot for Highway Mileage',
             fontdict={'size': 16})  # 设置标题
ax.set_xlabel('Miles Per Gallon')  # 设置横轴标签
ax.set_yticks(df.index)  # 设置纵轴刻度
ax.set_yticklabels(df.manufacturer.str.title(), fontdict={
    'horizontalalignment': 'right'})  # 设置纵轴标签
ax.set_xlim(10, 27)  # 设置x轴范围

plt.show()

























#【例4-20】创建一个包点图，其中点的大小和颜色都是基于mpg_z列的值进行设置，并在每个点上添加mpg_z的值作为标签。输入代码如下：
# 导入必要的库
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mtcars1.csv")  # 读取数据

# 提取'mpg'列作为x变量，并计算其标准化值
x = df.loc[:, ['mpg']]
df['mpg_z'] = (x - x.mean()) / x.std()

# 根据'mpg_z'列的值确定颜色
df['colors'] = ['red' if x < 0 else 'darkgreen' for x in df['mpg_z']]

df.sort_values('mpg_z', inplace=True)  # 根据'mpg_z'列的值对数据进行排序
df.reset_index(inplace=True)  # 重置索引

# 绘制图形
plt.figure(figsize=(14, 12), dpi=80)
plt.scatter(df.mpg_z, df.index, s=450, alpha=0.6, color=df.colors)

# 在每个点上添加'mpg_z'的值作为标签
for x, y, tex in zip(df.mpg_z, df.index, df.mpg_z):
    t = plt.text(x, y, round(tex, 1),
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontdict={'color': 'white'})

# 轻化边框
plt.gca().spines["top"].set_alpha(0.3)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)
plt.gca().spines["left"].set_alpha(0.3)

plt.yticks(df.index, df.cars)  # 设置y轴刻度标签
plt.title('Diverging Dotplot of Car Mileage',
          fontdict={'size': 20})  # 设置标题
plt.xlabel('$Mileage$')  # 设置x轴标签
plt.grid(linestyle='--', alpha=0.5)  # 添加网格线
plt.xlim(-2.5, 2.5)  # 设置x轴范围
plt.show()



























# 5 雷达图
#     雷达图（Radar Chart），也称为蜘蛛图（Spider Chart）或星形图（Star Plot），是一种用于
#可视化多维数据的图表类型。它以一个多边形的形式来表示数据的不同维度或变量，并通过
#将每个变量的取值连接起来，形成一个闭合的图形来展示数据之间的关系和相对大小。
#     雷达图可以同时显示多个变量的相对大小和差异，能够直观地比较不同维度之间的差异
# 和模式。特别适用于评估和比较多个方面或属性的性能、能力、优劣或优先级。通过雷达图，
# 可以更好地理解数据的多维特征，并发现其中的模式、异常或趋势。
#     由于雷达图在处理大量维度或数据量较大时可能会变得复杂和混乱。另外，雷达图在数
# 据分布不平衡或缺失维度时也可能存在一定的局限性。因此，在使用雷达图时，应仔细选择
# 和处理数据，确保图表清晰易读，并结合其他图表类型进行综合分析和解读。


#【例4-21】绘制雷达图示例一。输入代码如下：
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置数据
df = pd.DataFrame({'group': ['A', 'B', 'C', 'D', 'E'],  # 五组数据
                   'var1': [38, 1.5, 30, 4, 29], 'var2': [29, 10, 9, 34, 18],
                   'var3': [8, 39, 23, 24, 19], 'var4': [7, 31, 33, 14, 33],
                   'var5': [28, 15, 32, 14, 22]})  # 每组数据的变量

# ①处
# 获取变量列表
categories = list(df.columns[1:])
N = len(categories)

# 通过复制第1个值来闭合雷达图
values = df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
# 计算每个变量的角度
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# 初始化雷达图
fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
# 绘制每个变量的轴，并添加标签
plt.xticks(angles[:-1], categories, color='grey', size=8)

# 添加y轴标签
ax.set_rlabel_position(0)
plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=7)
plt.ylim(0, 40)

ax.plot(angles, values, linewidth=1, linestyle='solid')  # 绘制数据
ax.fill(angles, values, 'b', alpha=0.1)  # 填充区域
plt.show()























#【例4-22】绘制雷达图示例二。输入代码如下：
# 将上例①处后的代码替换为以下代码。
# ------- 第一部分:创建背景
categories = list(df)[1:]  # 列出除了'group'列之外的所有列名
N = len(categories)  # 变量的数量

# 计算每个轴在图中的角度（将图分成等份，每个变量对应一个角度）
angles = [n / float(N) * 2 * 3.14 for n in range(N)]  # 计算角度
angles += angles[:1]  # 为了闭合图形，将第1个角度再次添加到列表末尾

# 初始化雷达图
ax = plt.subplot(111, polar=True)
# 第1个轴在图的顶部
ax.set_theta_offset(3.14 / 2)
ax.set_theta_direction(-1)

# 为每个变量添加标签
plt.xticks(angles[:-1], categories)  # 设置x轴标签

# 添加y轴标签
ax.set_rlabel_position(0)
plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=7)  # 设置y轴刻度
plt.ylim(0, 40)  # 设置y轴范围

# ------- 第二部分:添加绘图
# 绘制第1个个体
values = df.loc[0].drop('group').values.flatten().tolist()  # 获取第一组的值
values += values[:1]  # 为了闭合图形，将第1个值再次添加到列表末尾
ax.plot(angles, values, linewidth=1, linestyle='solid',
        label="group A")  # 绘制线条
ax.fill(angles, values, 'b', alpha=0.1)  # 填充颜色

# 绘制第2个个体
values = df.loc[1].drop('group').values.flatten().tolist()  # 获取第二组的值
values += values[:1]  # 为了闭合图形，将第1个值再次添加到列表末尾
ax.plot(angles, values, linewidth=1, linestyle='solid',
        label="group B")  # 绘制线条
ax.fill(angles, values, 'r', alpha=0.1)  # 填充颜色
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))  # 添加图例
plt.show()
































#【例4-23】创建一个雷达图，用于可视化给定数据集中的每个分组的变量值。输入代码如下：
import matplotlib.pyplot as plt
import pandas as pd
# 设置数据
df = pd.DataFrame({'group': ['A', 'B', 'C', 'D'],
                   'var1': [38, 1.5, 30, 4], 'var2': [29, 10, 9, 34],
                   'var3': [8, 39, 23, 24], 'var4': [7, 31, 33, 14],
                   'var5': [28, 15, 32, 14]})
# ------- 第一部分:定义一个函数来绘制数据集中的每一行！
def make_spider(row, title, color):
    # 变量的数量
    categories = list(df)[1:]
    N = len(categories)
    # 计算每个轴在图中的角度
    angles = [n / float(N) * 2 * 3.14 for n in range(N)]
    angles += angles[:1]
    ax = plt.subplot(1, 4, row + 1, polar=True)  # 初始化雷达图
    # 如果希望第1个轴在顶部：
    ax.set_theta_offset(3.14 / 2)
    ax.set_theta_direction(-1)
    # 为每个变量添加标签
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    # 添加y轴标签
    ax.set_rlabel_position(0)
    plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=7)
    plt.ylim(0, 40)
    # 绘制数据
    values = df.loc[row].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    plt.title(title, size=11, color=color, y=1.1)  # 添加标题


# ------- 第二部分:将函数应用到所有数据
# 初始化图形
my_dpi = 96
plt.figure(figsize=(1000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)

my_palette = plt.colormaps.get_cmap("Set2")  # 创建颜色调色板

# 循环绘制雷达图
for row in range(0, len(df.index)):
    make_spider(row=row, title='group ' + df['group'][row],
                color=my_palette(row))
plt.show()























# 6 径向柱状图
#      径向柱状图（Radial Bar Chart），也被称为圆环图（Circular Bar Plot），是一种以圆环形
#式展示数据的柱状图。其优点是可以同时展示多个类别或分组的数据，以及它们的相对大小
#和差异。通过径向布局，可以更容易地比较不同类别之间的数据大小和趋势。
#      在径向柱状图中，每个数据类别或实体被表示为一个从圆心向外伸展的条形。每个条形
#的长度表示该类别或实体的数值大小。整个圆环被等分为多个扇区，每个扇区代表一个数据
#类别或实体。
#      由于径向柱状图在数据较多或柱状条形重叠时可能会显得混乱。因此在使用径向柱状图
#时，应根据数据的特点和数量进行调整，以确保图表的可读性和准确性。
#      利用 Python 和 Matplotlib 库构建径向柱状图时需要使用极坐标，而非笛卡尔坐标。这种
#表示形式可通过 subplot 函数的 polar 参数实现，基本过程如下。

#【例4-24】创建基础径向柱状图。输入如下代码。
# 导入所需的库
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 创建数据集，第一列为数据集各项的名称。第二列各项的数值
df = pd.DataFrame(
    {'Name': ['Ding ' + str(i) for i in list(range(1, 51))],
     'Value': np.random.randint(low=10, high=100, size=50)}
)
df.head(3)  # 显示前 3 行数据，输出略

# ①
plt.figure(figsize=(20, 10))  # 设置图形大小
ax = plt.subplot(111, polar=True)  # 绘制极坐标轴
plt.axis('off')  # 移除网格线

upperLimit = 100  # 设置坐标轴的上限
lowerLimit = 30  # 设置坐标轴的下限
max_value = df['Value'].max()  # 计算数据集中的最大值

# 计算每个条形图的高度，它们是在新坐标系中将每个条目值转换的结果
# 数据集中的0转换为lowerLimit(30)，最大值被转换为upperLimit(100)
slope = (max_value - lowerLimit) / max_value
heights = slope * df.Value + lowerLimit

width = 2 * np.pi / len(df.index)  # 计算每个条形图的宽度，共有2*Pi=360°

# 计算每个条形图中心的角度：
indexes = list(range(1, len(df.index) + 1))
angles = [element * width for element in indexes]

# ②
# 绘制条形图
bars = ax.bar(x=angles, height=heights, width=width,
              bottom=lowerLimit, linewidth=2, edgecolor="white")

#将②后绘制条形图的所有代码更换为如下代码，可以在径向柱状图上创建标签。
# 绘制条形图
bars = ax.bar(x=angles, height=heights, width=width,
              bottom=lowerLimit, linewidth=2, edgecolor="white", color="#61a4b2", )

labelPadding = 4  # 条形图和标签之间的小间距
# 添加标签
for bar, angle, height, label in zip(bars, angles, heights, df["Name"]):
    rotation = np.rad2deg(angle)  # 标签被旋转，旋转必须以度数为单位指定
    # 翻转部分标签使其朝下
    alignment = ""
    if angle >= np.pi / 2 and angle < 3 * np.pi / 2:
        alignment = "right"
        rotation = rotation + 180
    else:
        alignment = "left"
    # 添加标签
    ax.text(x=angle, y=lowerLimit + bar.get_height() + labelPadding,
            s=label, ha=alignment, va='center',
            rotation=rotation, rotation_mode="anchor")
#在①处添加如下代码。可以绘制排序的径向柱状图
df = df.sort_values(by=['Value'])  # 重新按值对数据集进行排序
plt.show()

































#【例4-25】绘制分组径向柱状图。输入代码如下：

#（1）绘制基础径向柱状图。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rng = np.random.default_rng(123)  # 确保随机数的可重现性
# 构建数据集
df = pd.DataFrame({
    "name": [f"Ding {i}" for i in range(1, 51)],
    "value": rng.integers(low=30, high=100, size=50),
    "group": ["A"] * 10 + ["B"] * 20 + ["C"] * 12 + ["D"] * 8
})
# ①处 
df.head(5)  # 显示前5行数据

# 辅助函数，用于标签的旋转和对齐
def get_label_rotation(angle, offset):
    """
    根据给定的角度和偏移量计算文本标签的旋转角度和对齐方式
    参数:
        angle (float):标签的角度。
        offset (float):开始角度的偏移量。
    返回:
        rotation (float):标签文本的旋转角度。
        alignment (str):文本对齐方式（'left' 或 'right'）。
    """
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else:
        alignment = "left"
    return rotation, alignment

# 辅助函数，用于添加标签
def add_labels(angles, values, labels, offset, ax):
    """
    添加文本标签到极坐标图。
    参数:
        angles (array-like):每个柱的角度
        values (array-like):每个柱的值。
        labels (array-like):每个柱的标签。
        offset (float):开始角度的偏移量。
        ax (matplotlib.axes._subplots.PolarAxesSubplot):极坐标图的轴对象。
    """
    padding = 4
    # 遍历角度、值和标签，以添加
    for angle, value, label, in zip(angles, values, labels):
        angle = angle

        # 获取文本的旋转角度和对齐方式
        rotation, alignment = get_label_rotation(angle, offset)

        # 添加文本
        ax.text(x=angle, y=value + padding, s=label,
                ha=alignment, va="center",
                rotation=rotation, rotation_mode="anchor")

ANGLES = np.linspace(0, 2 * np.pi, len(df), endpoint=False)
# 生成角度值，确定每个柱的位置
VALUES = df["value"].values  # 获取数据集中的数值列作为柱的高度
LABELS = df["name"].values  # 获取数据集中的名称列作为柱的标签

# 确定每个柱的宽度，一周为'2*pi'，将总宽度除以柱的数量
WIDTH = 2 * np.pi / len(VALUES)
# 确定第1个柱的位置。默认从0开始（第1个柱是水平的），指定从pi/2(90度)开始
OFFSET = np.pi / 2

# ***********************************modified
# ②处 
# 初始化图和轴
fig, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": "polar"})

ax.set_theta_offset(OFFSET)  # 指定偏移量
ax.set_ylim(-100, 100)  # 设置径向（y）轴的限制。负的下限创建了中间的空洞。
ax.set_frame_on(False)  # 删除所有脊柱
# 删除网格和刻度标记
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])
# 添加柱
ax.bar(ANGLES, VALUES, width=WIDTH, linewidth=2,
       color="#61a4b2", edgecolor="white")
add_labels(ANGLES, VALUES, LABELS, OFFSET, ax)  # 添加标签
#（2）在圆环中添加缺口。
#将②后的所有代码更换为如下代码，可以在圆环中添加缺口。
# 添加3个空柱
PAD = 3  # 设置额外空白柱的数量
ANGLES_N = len(VALUES) + PAD  # 计算包含额外空白柱的角度数量
# 计算新的角度数组，用于包含额外的空白柱，不包括终点
ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
WIDTH = (2 * np.pi) / len(ANGLES)  # 计算每个柱的宽度

IDXS = slice(0, ANGLES_N - PAD)  # 确定非空柱的索引范围

# 图和轴的设置与上述相同
fig, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": "polar"})

ax.set_theta_offset(OFFSET)
ax.set_ylim(-100, 100)
ax.set_frame_on(False)
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# 添加柱，使用仅包含非空柱的角度子集
ax.bar(ANGLES[IDXS], VALUES, width=WIDTH, color="#61a4b2",
       edgecolor="white", linewidth=2)
add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)  # 添加标签
#将②后的所有代码更换为如下代码，可以在圆环中添加缺口。
GROUP = df["group"].values  # 获取分组值
PAD = 3  # 向每个分组的末尾添加三个空柱

# 计算包含额外空白柱的角度数量，每个分组都会添加 PAD 个额外空白柱
ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
# 计算新的角度数组，用于包含额外的空白柱，不包括终点
ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
WIDTH = (2 * np.pi) / len(ANGLES)  # 计算每个柱的宽度

GROUPS_SIZE = [len(i[1]) for i in df.groupby("group")]  # 获取每个分组的大小

# 现在获取正确的索引有点复杂
offset = 0
IDXS = []
for size in GROUPS_SIZE:
    IDXS += list(range(offset + PAD, offset + size + PAD))
    offset += size + PAD

# 图和轴的设置与上述相同
fig, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": "polar"})

ax.set_theta_offset(OFFSET)
ax.set_ylim(-100, 100)
ax.set_frame_on(False)
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# 为每个分组使用不同的颜色！
GROUPS_SIZE = [len(i[1]) for i in df.groupby("group")]
COLORS = [f"C{i}" for i, size in enumerate(GROUPS_SIZE) for _ in range(size)]

# 最后添加柱。注意再次使用'ANGLES[IDXS]'来删除一些角度，以便在柱之间留下空间。
ax.bar(ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS,
       edgecolor="white", linewidth=2)
add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)  # 添加标签

#（3）在组内排序。
df_sorted = (df
             .groupby(["group"])
             .apply(lambda x: x.sort_values(["value"], ascending=False))
             .reset_index(drop=True))
df = df_sorted

plt.show()


























# 7 词云图
#       词云图（Word Cloud）是一种可视化工具，用于展示文本数据中单词的频率或重要性。
# 它通过在一个图形区域内根据单词的频率或重要性来调整单词的大小，并将这些单词以一种
# 修饰性的方式呈现，形成一个图形化的云状图。
#       词云图通过直观地展示单词的频率或重要性，帮助我们快速了解文本数据的主题、关键
# 词或热点。词云图常用于可视化文本摘要、主题分析、舆情分析等领域。然而，词云图并不
# 能提供详细的语义信息，因此在进行深入分析时，可能需要结合其他文本分析技术或图表。
#       此外，为了准确地反映文本的特征，词云图的制作过程需要注意合理选择预处理方法和
# 调整词频或重要性的计算方式。


#【例4-26】绘制词云图示例。输入代码如下：
from os import path  # 导入操作系统路径模块中的path函数，用于处理文件路径
from PIL import Image  # 导入PIL中的Image类，用于读取、处理图像
import numpy as np  # 导入NumPy库，用于处理数组和矩阵
import matplotlib.pyplot as plt  # 导入pyplot模块，用于绘制图形
import os  # 导入操作系统相关的模块，用于获取当前工作目录等操作
from wordcloud import WordCloud, STOPWORDS

# 导入词云库中的WordCloud类和STOPWORDS集合，用于生成和处理词云图

# 获取当前文件的目录
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
# 读取整个文本文件的内容
text = open(path.join(d, r'S:\Desktop_new\caxbook_python\python_plot_202405\PyData\alice.txt')).read()

# 读取蒙版图像（掩码图像）
# 蒙版图像用于控制词云形状，这里使用了一个Alice in Wonderland的蒙版图像
alice_mask = np.array(Image.open(
    path.join(d, r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\alice_mask.png")))

# 设置停用词（在词云中不显示的词语）
stopwords = set(STOPWORDS)
stopwords.add("said")  # 添加一个额外的停用词

# 创建词云对象，指定背景颜色、最大词数、蒙版图像、停用词等参数
wc = WordCloud(background_color="white", max_words=2000,
               mask=alice_mask, stopwords=stopwords,
               contour_width=3, contour_color='steelblue')

wc.generate(text)  # 生成词云图
# 将词云图保存为图片文件
wc.to_file(path.join(d, r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\alice.png"))
# 显示词云图
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")  # 不显示坐标轴
plt.show()

# 显示蒙版图像（仅为了对比效果）
plt.figure()
plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")  # 不显示坐标轴
plt.show()































#【例4-27】创建词云图示例1。输入代码如下：
import numpy as np  # 导入NumPy库，用于处理数组和矩阵
from PIL import Image  # 导入PIL中的Image类，用于读取、处理图像
from os import path  # 导入操作系统路径模块中的path函数，用于处理文件路径
import matplotlib.pyplot as plt  # 导入pyplot模块，用于绘制图形
import os  # 导入操作系统相关的模块，用于获取当前工作目录等操作
import random  # 导入random模块，用于生成随机数
from wordcloud import WordCloud, STOPWORDS


# 导入词云库中的WordCloud类和STOPWORDS集合，用于生成和处理词云图

# 定义自定义颜色函数，用于生成词云中词语的颜色
def grey_color_func(word, font_size, position, orientation,
                    random_state=None, **kwargs):
    # 使用随机的亮度值来生成灰度颜色，使词云图更加丰富多彩
    return "hsl(0,0%%,%d%%)" % random.randint(60, 100)


# 获取当前文件的目录
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

# 读取蒙版图像（掩码图像）
# 这里使用了一个星球大战中的Stormtrooper的蒙版图像
mask = np.array(Image.open(
    path.join(d, r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\stormtrooper_mask.png")))

# 读取星球大战电影《新希望》的剧本文本
text = open(path.join(d, r'S:\Desktop_new\caxbook_python\python_plot_202405\PyData\a_new_hope.txt')).read()

# 对文本进行一些预处理，例如替换文本中的特定词语
text = text.replace("HAN", "Han")
text = text.replace("LUKE'S", "Luke")

# 添加剧本特定的停用词
stopwords = set(STOPWORDS)
stopwords.add("int")
stopwords.add("ext")

# 创建词云对象，并生成词云图
wc = WordCloud(max_words=1000, mask=mask, stopwords=stopwords,
               margin=10, random_state=1).generate(text)

default_colors = wc.to_array()  # 保存默认颜色的词云图

# 显示自定义颜色的词云图
plt.title("Custom colors")
plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3),
           interpolation="bilinear")
plt.axis("off")
plt.show()

# 显示默认颜色的词云图
plt.figure()
plt.title("Default colors")
plt.imshow(default_colors, interpolation="bilinear")
plt.axis("off")
plt.show()



























# 8 玫瑰图
#       玫瑰图（Rose Plot），也称为极坐标直方图（Polar Histogram），是一种在极坐标系统下显
# 示数据分布的图表类型。它以一个圆形或半圆形的坐标系来表示数据，其中数据的频率或计
# 数通过半径的长度来表示，角度则代表不同的类别或区间。
#       玫瑰图常用于显示具有周期性或方向性特征的数据，例如风向分布、季节性数据、方位
# 角分布等。它可以帮助我们直观地理解数据的分布情况和主要趋势，同时提供了一种有效的
# 可视化方式，将多个类别或区间的数据进行比较。



#【例4-28】绘制玫瑰图示例1。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19781101)  # 固定随机种子，以便结果可复现
# 计算扇形柱
N = 20
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = 10 * np.random.rand(N)
width = np.pi / 4 * np.random.rand(N)
colors = plt.cm.viridis(radii / 10.)

# 创建极坐标子图
ax = plt.subplot(projection='polar')
# 绘制柱状图
ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.5)
plt.show()


























#【例4-29】绘制玫瑰图示例2。输入代码如下：
import numpy as np
from windrose import WindroseAxes
from matplotlib import cm
import matplotlib.pyplot as plt

# 生成随机的风速和风向数据
N = 500
ws = np.random.random(N) * 6  # 随机生成风速数据（0到6之间）
wd = np.random.random(N) * 360  # 随机生成风向数据（0到360之间）

# 创建风玫瑰图对象，并绘制归一化柱状图
ax = WindroseAxes.from_ax()
ax.bar(wd, ws, normed=True, opening=0.8, edgecolor="white")  # 绘制柱状图
ax.set_legend()  # 添加图例
plt.show()

# 创建风玫瑰图对象，并绘制箱线图
ax = WindroseAxes.from_ax()
ax.box(wd, ws, bins=np.arange(0, 8, 1))  # 绘制箱线图
ax.set_legend()  # 添加图例
plt.show()

# 创建风玫瑰图对象，并绘制填充等高线图
ax = WindroseAxes.from_ax()
ax.contourf(wd, ws, bins=np.arange(0, 8, 1), cmap=cm.hot)  # 绘制填充等高线图
ax.set_legend()  # 添加图例
plt.show()

# 创建风玫瑰图对象，并绘制等高线图
ax = WindroseAxes.from_ax()
ax.contour(wd, ws, bins=np.arange(0, 8, 1), cmap=cm.hot, lw=1)  # 绘制等高线图
ax.set_legend()  # 添加图例
plt.show()

































#【例4-30】绘制玫瑰图示例1。输入代码如下：
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# 导入 windrose 库中的 WindroseAxes 和 plot_windrose 函数
from windrose import WindroseAxes, plot_windrose

# 创建包含风速、风向和月份的 DataFrame
wind_data = pd.DataFrame({
    "ws": np.random.random(1200) * 6,  # 随机生成风速数据（0到6之间）
    "wd": np.random.random(1200) * 360,  # 随机生成风向数据（0到360之间）
    "month": np.repeat(range(1, 13), 100),  # 重复月份数据100次
})


def plot_windrose_subplots(data, *, direction, var, color=None, **kwargs):
    """封装函数，用于在子图中绘制风玫瑰图"""
    ax = plt.gca()  # 获取当前轴对象
    ax = WindroseAxes.from_ax(ax=ax)  # 创建风玫瑰图对象，并在当前轴上添加
    # 调用 plot_windrose 函数绘制风玫瑰图
    plot_windrose(direction_or_df=data[direction],
                  var=data[var], ax=ax, **kwargs)


# 使用 FacetGrid 创建子图结构
g = sns.FacetGrid(data=wind_data, col="month",  # 按月份创建子图列
                  col_wrap=4,  # 每行最多显示4个子图
                  subplot_kws={"projection": "windrose"},  # 使用风玫瑰图投影
                  sharex=False, sharey=False,  # 不共享 x、y 轴
                  despine=False,  # 不去掉轴线外侧边框
                  height=3.5, )  # 子图高度

# 在每个子图上调用 plot_windrose_subplots 函数绘制风玫瑰图
g.map_dataframe(plot_windrose_subplots,
                direction="wd",  # 风向数据列名
                var="ws",  # 风速数据列名
                normed=True,  # 对频率进行归一化
                bins=(0.1, 1, 2, 3, 4, 5),  # 设置频率区间
                calm_limit=0.1,  # 静风的限制值
                kind="bar",  # 使用柱状图形式
                )

# 设置每个子图的图例和径向网格线
y_ticks = range(0, 17, 4)  # 设置径向网格线范围
for ax in g.axes:
    ax.set_legend(title="$m \cdot s^{-1}$",  # 图例标题
                  bbox_to_anchor=(1.15, -0.1),  # 图例位置
                  loc="lower right",  # 图例位置
                  )
    plt.show()
ax.set_rgrids(y_ticks, y_ticks)  # 设置径向网格线
plt.subplots_adjust(wspace=-0.2)  # 调整子图之间的间距