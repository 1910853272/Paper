# 五、数值关系数据可视化
#     数值关系数据可视化是数据科学和数据分析中至关重要的一部分。通过将数据可视化，
# 可以将数据转化为图形或图表，使得数据的模式、关联和趋势变得更加清晰和直观。本章将
# 介绍常见的数值关系数据可视化的实现方法，包括散点图、气泡图、等高线图等。通过本章
# 的学习可以掌握数值关系数据可视化的知识，并能在实际应用中灵活运用。


# 5.1 散点图
#     散点图（Scatter Plot）是一种用于可视化两个连续变量之间关系的图表类型。它以坐标
# 系中的点的位置来表示数据的取值，并通过点的分布来展示两个变量之间的相关性、趋势和
# 离散程度。
#     散点图可以展示两个变量之间的分布模式和趋势，帮助观察变量之间的关系和可能的相
# 关性。通过散点图，可以发现数据的聚集、离散、趋势、异常值等特征。
#     当数据点较多时，散点图可能会出现重叠，导致点的形状和分布难以辨认，此时可以使
# 用透明度、颜色编码等方式来区分和凸显不同的数据子集。
#     在python 中构建散点图，最简单的是利用seaborn，而matplotlib 允许更高级别的定制。
# 如果需要构建交互式图表，可以使用plotly。

#【例5-1】使用seaborn库绘制鸢尾花（iris）数据集中萼片长度（sepal_length）与萼片宽度（sepal_width）之间的散点图。输入代码如下：
import seaborn as sns  # 导入seaborn库并简写为sns
import matplotlib.pyplot as plt
df = sns.load_dataset('iris',data_home='seaborn',cache=True)  # 加载 iris 数据集
# 绘制散点图
sns.regplot(x=df["sepal_length"], y=df["sepal_width"])
# 绘制散点图（默认进行线性拟合），如图所示

sns.regplot(x=df["sepal_length"], y=df["sepal_width"], fit_reg=False)
plt.show()















# 绘制散点图（不进行线性拟合），如图所示
#【例5-2】利用midwest_filter.csv数据集绘制散点图，展示了中西部地区的面积与人口之间的关系。输入代码如下：
# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

midwest = pd.read_csv('S:\Desktop_new\caxbook_python\python_plot_202405\PyData\midwest_filter.csv')  # 导入数据集
categories = np.unique(midwest['category'])  # 获取数据集中的唯一类别
# 生成与唯一类别数量相同的颜色
# 使用 matplotlib 的颜色映射函数来生成颜色
colors = [plt.cm.tab10(i / float(len(categories) - 1)) for i in
          range(len(categories))]

# 绘制每个类别的图形
# 设置图形的大小、分辨率和背景色
plt.figure(figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')

# 遍历每个类别，并使用 scatter 函数绘制散点图
for i, category in enumerate(categories):
    # 使用 loc 方法筛选出特定类别的数据，并绘制散点图
    plt.scatter('area', 'poptotal',
                data=midwest.loc[midwest.category == category, :],
                s=20, c=colors[i], label=str(category))

# 图形修饰
plt.gca().set(xlim=(0.0, 0.1), ylim=(0, 90000),
              xlabel='Area', ylabel='Population')  # 设置 x 轴和 y 轴的范围、标签

# 设置 x 轴和 y 轴的刻度字体大小
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.title("Scatterplot ", fontsize=22)  # 设置图形标题
plt.legend(fontsize=12)  # 添加图例
plt.show()























#【例5-3】使用seaborn的regplot函数在鸢尾花（iris）数据集上绘制添加线性回归的散点图。输入代码如下：
import pandas as pd  # 导入pandas库并简写为pd
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块并简写为plt
import seaborn as sns  # 导入seaborn库并简写为sns

df = sns.load_dataset('iris')  # 加载iris数据集
# 绘制散点图，并添加红色线性回归线
fig, ax = plt.subplots(figsize=(8, 6))  # 创建图形和子图对象，并设置图形大小
sns.regplot(x=df["sepal_length"], y=df["sepal_width"],
            # 绘制散点图，指定横纵轴数据
            line_kws={"color": "r"}, ax=ax)  # 设置线性回归线颜色为红色
plt.show()

# 绘制散点图，并添加半透明的红色线性回归线
fig, ax = plt.subplots(figsize=(8, 6))  # 创建图形和子图对象，并设置图形大小
sns.regplot(x=df["sepal_length"], y=df["sepal_width"],
            # 绘制散点图，指定横纵轴数据
            line_kws={"color": "r", "alpha": 0.4}, ax=ax)
# 设置线性回归线颜色为红色，透明度为0.4
plt.show()

# 绘制散点图，并自定义线性回归线的线宽、线型和颜色
fig, ax = plt.subplots(figsize=(8, 6))  # 创建图形和子图对象，并设置图形大小
sns.regplot(x=df["sepal_length"], y=df["sepal_width"],
            # 绘制散点图，指定横纵轴数据
            line_kws={"color": "r", "alpha": 0.4, "lw": 5, "ls": "--"}, ax=ax)
# 设置线性回归线的颜色、透明度、线宽和线型
plt.show()























#【例5-4】使用seaborn的regplot函数在鸢尾花（iris）数据集上绘制散点图，请尝试采用不同的点形状。输入代码如下：
fig, ax = plt.subplots(figsize=(8, 6))  # 创建图形和子图对象，并设置图形大小
# 绘制散点图，不添加回归线，标记形状为"+"
sns.regplot(x=df["sepal_length"], y=df["sepal_width"],
            marker="+", fit_reg=False, ax=ax)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))  # 创建图形和子图对象，并设置图形大小
# 绘制散点图，不添加回归线，设置散点标记颜色为暗红色，透明度为0.3，标记大小为200
sns.regplot(x=df["sepal_length"], y=df["sepal_width"], fit_reg=False,
            scatter_kws={"color": "darkred", "alpha": 0.3, "s": 200}, ax=ax)
plt.show()
























#【例5-5】使用seaborn的regplot函数在鸢尾花（iris）数据集上绘制散点图，每幅散点图均按照不同的方式（使用分类变量）对数据子集进行着色和标记。输入代码如下：
import seaborn as sns  # 导入seaborn库
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库

df = sns.load_dataset('iris')  # 加载iris数据集
# 使用'hue'参数提供一个因子变量，并绘制散点图
sns.lmplot(x="sepal_length", y="sepal_width",
           data=df, fit_reg=False, hue='species', legend=False)
# 将图例移动到图形中的一个空白部分
plt.legend(loc='lower right')
plt.show()

# 绘制散点图，并指定每个数据子集的标记形状
sns.lmplot(x="sepal_length", y="sepal_width", data=df,
           fit_reg=False, hue='species', legend=False, markers=["o", "x", "1"])
# 将图例移动到图形中的一个空白部分
plt.legend(loc='lower right')
plt.show()

# 使用调色板来着色不同的数据子集
sns.lmplot(x="sepal_length", y="sepal_width",
           data=df, fit_reg=False, hue='species',
           legend=False, palette="Set2")
# 将图例移动到图形中的一个空白部分
plt.legend(loc='lower right')
plt.show()

# 控制每个数据子集的颜色
sns.lmplot(x="sepal_length", y="sepal_width",
           data=df, fit_reg=False, hue='species', legend=False,
           palette=dict(setosa="blue", virginica="red",
                        versicolor="green"))
# 将图例移动到图形中的一个空白部分
plt.legend(loc='lower right')
plt.show()


























#【例5-6】绘制抖动散点图。输入代码如下：
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mpg_ggplot2.csv")  # 导入数据

# 绘制Stripplot
fig, ax = plt.subplots(figsize=(12, 6), dpi=80)
sns.stripplot(x='cty', y='hwy', hue='class', data=df, jitter=0.25, size=6,
              ax=ax, linewidth=.5)
# 修饰
plt.title('Use jittered plots to avoid overlapping of points', fontsize=22)
plt.show()

























# 5.2 边际图
#     边际图（Margin Diagram）通常用于表示某种现象或数据集中的边际分布。它是一种统
# 计图表，用于显示数据的边际分布情况。边际分布指的是在多个变量中只关注其中一个变量
# 时的分布情况。
#     边际图通常用于多维数据的可视化，特别是在探索性数据分析阶段。它能够帮助观察数
# 据的边际关系，即在其他变量保持不变的情况下，某一变量的分布情况。边际图的绘制方式
# 可以因数据类型和目的而异，但一般来说，它们常常包括以下几种形式：

#     （1）边际直方图：在边际图中，通过绘制直方图来表示某个变量的分布情况。在多维
# 数据中，可以选择其中一个变量，绘制其直方图，并在其边缘显示其他变量的密度或频率。
#     （2）边际密度图：与边际直方图类似，边际密度图也是用来表示某个变量的分布情况，
# 但是采用的是核密度估计等连续密度估计方法，以平滑地显示概率密度。
#     （3）边际箱线图：边际箱线图可以用来展示某个变量的分布情况，并在其边缘显示其
# 他变量的分布情况，通过箱线图的上下界和中位数等统计量可以观察到数据的分布情况和离
# 群值。
#     （4）边际散点图：当数据是二维的时候，可以绘制边际散点图来显示两个变量的边际
# 分布情况。通常在散点图的边缘添加直方图或密度图，以展示每个变量的边际分布。

#     边际图的优点在于可以同时展示多个变量之间的边际关系，有助于发现变量之间的相互
# 作用以及变量的单独影响。在数据分析和可视化过程中，边际图是一种非常有用的工具，可
# 以帮助人们更好地理解数据的特征和结构。使用seaborn的jointplot（）函数绘制边际散点图。

#【matplotlib == 3.7.5!】
#【例5-7】使用Seaborn库绘制不同类型的边际图，包括带有散点图、六边形图、核密度估计图等。输入代码如下：
import seaborn as sns  # 导入Seaborn库并简写为sns
import matplotlib.pyplot as plt  # 导入Matplotlib.pyplot库并简写为plt

df = sns.load_dataset('iris')  # 从Seaborn中加载iris数据集
fig, axs = plt.subplots(1, 3, figsize=(12, 10))  # 创建一个2x2的网格布局

# 创建带有散点图的边际图
sns.jointplot(x=df["sepal_length"], y=df["sepal_width"], kind='scatter')
# 创建带有六边形图的边际图
sns.jointplot(x=df["sepal_length"], y=df["sepal_width"], kind='hex')
# 创建带有核密度估计图的边际图
sns.jointplot(x=df["sepal_length"], y=df["sepal_width"], kind='kde')
plt.show()

# 自定义联合图中的散点图
sns.jointplot(x=df["sepal_length"], y=df["sepal_width"],
              kind='scatter', s=200, color='m',
              edgecolor="skyblue", linewidth=2)

# 自定义颜色
sns.set_theme(style="white", color_codes=True)
sns.jointplot(x=df["sepal_length"], y=df["sepal_width"], kind='kde',
              color="skyblue")
plt.show()

# 自定义直方图
sns.jointplot(x=df["sepal_length"], y=df["sepal_width"], kind='hex',
              marginal_kws=dict(bins=30, fill=True))
plt.show()

# 无间隔
sns.jointplot(x=df["sepal_length"], y=df["sepal_width"], kind='kde',
              color="blue", space=0)
# 大间隔
sns.jointplot(x=df["sepal_length"], y=df["sepal_width"], kind='kde',
              color="blue", space=3)
# 调整边际图比例
sns.jointplot(x=df["sepal_length"], y=df["sepal_width"],
              kind='kde', ratio=2)
plt.show()




print('5-7 ended!')





















'''
# matplotlib == low version than 3.7.5 such as 3.3.0
#【例5-8】绘制边缘为直方图的边际图。边际图具有沿X和Y轴变量的直方图，用于可视化X和Y之间的关系以及单独的X和Y的单变量分布，常用于探索性数据分析（EDA）。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mpg_ggplot2.csv")  # 导入数据
# 创建图形和网格布局
fig = plt.figure(figsize=(12, 8), dpi=80)
grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

# 定义坐标轴
ax_main = fig.add_subplot(grid[:-1, :-1])
ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

# 主图上的散点图
ax_main.scatter('displ', 'hwy', s=df.cty * 4,
                c=df.manufacturer.astype('category').cat.codes,
                alpha=.9, data=df, cmap="tab10", edgecolors='gray', linewidths=.5)

# 右侧的直方图
ax_bottom.hist(df.displ, 40, histtype='stepfilled', orientation='vertical',
               color='blue', alpha=0.8)
ax_bottom.invert_yaxis()

# 底部的直方图
ax_right.hist(df.hwy, 40, histtype='stepfilled', orientation='horizontal',
              color='blue', alpha=0.8)

# 图形修饰
ax_main.set(title='Scatterplot with Histograms \n displ vs hwy',
            xlabel='displ', ylabel='hwy')
ax_main.title.set_fontsize(20)
for item in ([ax_main.xaxis.label, ax_main.yaxis.label] +
             ax_main.get_xticklabels() + ax_main.get_yticklabels()):
    item.set_fontsize(14)

xlabels = ax_main.get_xticks().tolist()
ax_main.set_xticklabels(xlabels)
plt.show()
'''

























#【例5-9】边缘箱线图与边缘直方图具有相似的用途。箱线图有助于精确定位X和Y的中位数、第25和第75百分位数。输入代码如下：
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mpg_ggplot2.csv")  # 导入数据

# 创建图形和网格布局
fig = plt.figure(figsize=(12, 8), dpi=80)
grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

# 定义坐标轴
ax_main = fig.add_subplot(grid[:-1, :-1])  # 主图的位置
ax_right = fig.add_subplot(grid[:-1, -1],
                           xticklabels=[], yticklabels=[])  # 右侧箱线图的位置
ax_bottom = fig.add_subplot(grid[-1, 0:-1],
                            xticklabels=[], yticklabels=[])  # 底部箱线图的位置

# 主图上的散点图
ax_main.scatter('displ', 'hwy', s=df.cty * 5,
                c=df.manufacturer.astype('category').cat.codes,  # 根据制造商进行着色
                alpha=.8, data=df, cmap="Set1", edgecolors='black', linewidths=.5)
sns.boxplot(df.hwy, ax=ax_right, orient="v")  # 在右侧添加箱线图
sns.boxplot(df.displ, ax=ax_bottom, orient="h")  # 在底部添加箱线图

# 图形修饰
ax_bottom.set(xlabel='')  # 移除箱线图的x轴名称
ax_right.set(ylabel='')  # 移除箱线图的y轴名称

# 主标题、X轴和Y轴标签
ax_main.set(title='Scatterplot with Histograms \n displ vs hwy',
            xlabel='displ', ylabel='hwy')

# 设置字体大小
ax_main.title.set_fontsize(20)  # 设置主标题的字体大小
for item in ([ax_main.xaxis.label, ax_main.yaxis.label] +
             ax_main.get_xticklabels() + ax_main.get_yticklabels()):
    item.set_fontsize(14)  # 设置其他组件的字体大小
plt.show()

























'''
# matplotlib == 3.3.0!!!!
#【例5-10】利用proplot绘制边际图。输入代码如下：
import proplot as pplt
import numpy as np

# 创建数据
N = 500
state = np.random.RandomState(51423)
x = state.normal(size=(N,))
y = state.normal(size=(N,))
bins = pplt.arange(-3, 3, 0.25)  # 设置直方图的区间范围和步长

# 创建带有边际分布的直方图
fig, axs = pplt.subplots(ncols=2, refwidth=2.3)  # 创建一个包含两个子图的图形
axs.format(abc='A.', abcloc='l', titleabove=True,  # 设置图形标签和标题的样式
           ylabel='y axis',
           suptitle='Histograms with marginal distributions'  # 设置y轴标签和总标题
           )
colors = ('indigo9', 'red9')  # 设置直方图颜色
titles = ('Group 1', 'Group 2')  # 设置子图标题
for ax, which, color, title in zip(axs, 'lr', colors, titles):
    # 绘制2D直方图
    ax.hist2d(x, y, bins, vmin=0, vmax=10, levels=50,  # 绘制二维直方图
              cmap=color, colorbar='b',
              colorbar_kw={'label': 'count'}  # 设置颜色映射和颜色条
              )
    color = pplt.scale_luminance(color, 1.5)  # 调整直方图颜色的亮度
    # 添加边际直方图
    px = ax.panel(which, space=0)  # 创建边际直方图所在的面板
    px.histh(y, bins, color=color, fill=True, ec='k')  # 绘制y轴边际直方图
    px.format(grid=False, xlocator=[],
              xreverse=(which == 'l'))  # 格式化面板，设置网格线和x轴刻度
    px = ax.panel('t', space=0)  # 创建顶部边际直方图所在的面板
    px.hist(x, bins, color=color, fill=True, ec='k')  # 绘制x轴边际直方图
    px.format(grid=False, ylocator=[], title=title,
              titleloc='l')  # 格式化面板，设置网格线、y轴刻度和标题位置
'''








print('5-10 ended')















# 5.3 曼哈顿图
#     曼哈顿图是一种特定类型的散点图，多用于全基因组关联研究（GWAS），其每个点代表
# 一种基因变异，X轴值表示它在染色体上的位置，y轴值表示它与某一性状的关联程度。
#     曼哈顿图（Manhattan plot）通常用于展示基因组关联研究（GWAS）等研究中的统计显
# 著性结果，但也可用于其他类型的数据。这种图表的名称源自于纽约曼哈顿的天际线，图表
# 中的垂直线条和水平线条形成了类似建筑物和街道的感觉。
#     曼哈顿图的特点是横轴代表基因组的染色体位置，纵轴代表某种统计指标（例如-p 值、
# 负对数-p 值、FDR 等），而每个数据点则表示某个单个基因或基因组区域的统计显著性。通
# 常，显著性较高的数据点会在图上显示为较高的柱状，而较不显著的数据点则显示为较短的
# 柱状。
#     曼哈顿图的主要用途之一是帮助研究人员快速发现在基因组中具有显著关联的区域或
# 基因，尤其是在大规模数据集中进行关联分析时。通过观察曼哈顿图，研究人员可以迅速识
# 别出具有统计学上显著的基因变异或其他特征，这有助于进一步的生物信息学分析和实验验
# 证。
#     曼哈顿图在研究中的使用已经成为了一个标准实践，因为它提供了一种直观的方式来可
# 视化大规模基因组数据的统计结果，并且能够帮助研究人员迅速确定值得进一步研究的区域
# 或基因。
'''
#【例5-11】绘制曼哈顿图示例1。输入代码如下：
from bioinfokit import analys, visuz  # 导入 analys 和 visuz 模块

df = analys.get_data('mhat').data  # 加载数据集为 pandas DataFrame
df.head(2)  # 输出略

color = ("#a7414a", "#696464", "#00743f", "#563838", "#6a8a82",
         "#a37c27", "#5edfff", "#282726", "#c0334d", "#c9753d")  # 设置颜色列表

# 创建默认参数的曼哈顿图，如图（a）所示
visuz.marker.mhat(df=df, chr='chr', pv='pvalue')

# 默认情况下，线将绘制在P=5E-08处，如图（b）所示
visuz.marker.mhat(df=df, chr='chr', pv='pvalue', color=color,
                  gwas_sign_line=True)  # 添加全基因组显著性线

# 更改全基因组显著性线的位置，根据需要更改该值，如图（c）所示
visuz.marker.mhat(df=df, chr='chr', pv='pvalue', color=color,
                  gwas_sign_line=True, gwasp=5E-06)

# 根据'gwasp'定义的显著性，为SNPs添加注释（文本框文本），如图（d）所示
visuz.marker.mhat(df=df, chr='chr', pv='pvalue', color=color,
                  gwas_sign_line=True, gwasp=5E-06,
                  markernames=True, markeridcol='SNP', gstyle=2)
'''

























#【例5-12】绘制曼哈顿图示例2。输入代码如下：
from pandas import DataFrame  # 导入DataFrame模块
from scipy.stats import uniform  # 从scipy.stats模块导入uniform分布
from scipy.stats import randint  # 从scipy.stats模块导入randint分布
import numpy as np  # 导入numpy库，并简称为np
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，并简称为plt

# 生成样本数据
df = DataFrame({'gene': ['gene-%i' % i for i in np.arange(10000)],
                # 创建基因名字的序列
                'pvalue': uniform.rvs(size=10000),  # 生成服从均匀分布的p值数据
                'chromosome': ['ch-%i' % i for i in randint.rvs(0, 12, size=10000)]})
# 生成随机的染色体编号

# 计算-log10(pvalue)
df['minuslog10pvalue'] = -np.log10(df.pvalue)
# 计算-p值的负对数，用于曼哈顿图的纵轴
df.chromosome = df.chromosome.astype('category')  # 将染色体列转换为分类类型
df.chromosome = df.chromosome.cat.set_categories(
    ['ch-%i' % i for i in range(12)], ordered=True)  # 对染色体进行排序
df = df.sort_values('chromosome')  # 根据染色体排序

# 准备绘制曼哈顿图
df['ind'] = range(len(df))  # 为数据集添加索引列
df_grouped = df.groupby(('chromosome'), observed=False)  # 按染色体分组

# 绘制曼哈顿图
fig = plt.figure(figsize=(14, 8))  # 设置图形大小
ax = fig.add_subplot(111)  # 添加子图
colors = ['darkred', 'darkgreen', 'darkblue', 'gold']  # 定义颜色列表
x_labels = []  # 初始化x轴标签列表
x_labels_pos = []  # 初始化x轴标签位置列表
for num, (name, group) in enumerate(df_grouped):  # 遍历分组后的数据
    group.plot(kind='scatter', x='ind', y='minuslog10pvalue',
               color=colors[num % len(colors)], ax=ax)  # 绘制散点图，并按染色体着色
    x_labels.append(name)  # 添加染色体名到标签列表
    x_labels_pos.append((group['ind'].iloc[-1] -
                         (group['ind'].iloc[-1] - group['ind'].iloc[0]) / 2))  # 添加染色体标签的位置
ax.set_xticks(x_labels_pos)  # 设置x轴刻度位置
ax.set_xticklabels(x_labels)  # 设置x轴刻度标签

ax.set_xlim([0, len(df)])  # 设置x轴范围
ax.set_ylim([0, 3.5])  # 设置y轴范围
ax.set_xlabel('Chromosome')  # 设置x轴标签
plt.show()
























# 5.4 气泡图
#     气泡图（Bubble Chart）是一种用于可视化三个变量之间关系的图表类型。气泡图基本上
# 类似于一个散点图，它通过在坐标系中以点的形式表示数据，并使用不同大小的气泡（圆形）
# 来表示第3个变量的数值。
#     气泡图能够同时展示三个变量之间的关系，通过点的位置和气泡的大小，可以观察两个
# 变量之间的相关性、趋势，并展示第3个变量的相对大小。
#     注意：绘图时需要考虑气泡的大小范围，确保气泡大小的差异在图表中明显可见。可以
# 根据数据的范围和分布进行适当的调整。如果数据集中有多个类别或分组，可以考虑使用不
# 同的颜色或形状来区分和表示不同的类别，以增加图表的多样性和可读性。

#【例5-13】基于gapminder数据集绘制基础气泡图，展示人均GDP（GDP
#per
#Capita）与预期寿命（Life
#Expectancy）间的关系，并且根据大洲（continent）对数据进行分类。输入代码如下：
import matplotlib.pyplot as plt  # 导入Matplotlib库并简写为plt
import seaborn as sns  # 导入Seaborn库并简写为sns
from gapminder import gapminder  # 导入数据集

plt.rcParams['figure.figsize'] = [8, 8]  # 设置笔记本中的图形大小
data = gapminder.loc[gapminder.year == 2007]  # 从数据集中选择特定年份的数据

# 使用scatterplot函数绘制气泡地图
sns.scatterplot(data=data, x="gdpPercap", y="lifeExp", size="pop",
                legend=False, sizes=(20, 1600))
plt.show()

sns.set_style("darkgrid")  # 设置Seaborn主题为"darkgrid"
# 使用scatterplot函数绘制气泡地图
sns.scatterplot(data=data, x="gdpPercap", y="lifeExp", size="pop",
                hue="continent", palette="viridis",
                edgecolor="blue", alpha=0.5, sizes=(10, 1600))

# 添加标题（主标题和轴标题）
plt.xlabel("Gdp per Capita")  # x轴标题
plt.ylabel("Life Expectancy")  # y轴标题

# 将图例放置在图形外部
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=15)
plt.show()
























#【例5-14】基于gapminderData数据集绘制气泡图，展示人均GDP与预期寿命间的关系。输入代码如下：
import pandas as pd  # 导入Pandas库并简写为pd
import matplotlib.pyplot as plt  # 导入Matplotlib库并简写为plt

data = pd.read_csv('S:\Desktop_new\caxbook_python\python_plot_202405\PyData\gapminderData.csv')  # 读取数据
data.head(2)  # 检查前两行数据

# 将分类列（continent）转换为数值型分组（group1->1,group2->2...）
data['continent'] = pd.Categorical(data['continent'])

plt.figure(figsize=(8, 6))  # 设置图形大小
data1952 = data[data.year == 1952]  # 选取1952年的数据子集

# 绘制散点图
plt.scatter(
    x=data1952['lifeExp'],  # x轴为预期寿命
    y=data1952['gdpPercap'],  # y轴为人均GDP
    s=data1952['pop'] / 50000,  # 气泡大小与人口数量相关
    c=data1952['continent'].cat.codes,  # 根据大洲分类编码设置气泡颜色
    cmap="Accent",  # 使用Accent调色板
    alpha=0.6,  # 设置透明度
    edgecolors="white",  # 设置气泡边缘颜色
    linewidth=2)  # 设置气泡边缘线宽度

# 添加标题（主标题和轴标题）
plt.yscale('log')  # 设置y轴为对数尺度
plt.xlabel("Life Expectancy")  # x轴标题
plt.ylabel("GDP per Capita")  # y轴标题
plt.title("Year 1952")  # 主标题
plt.ylim(0, 50000)  # 设置y轴范围
plt.xlim(30, 75)  # 设置x轴范围
plt.show()













print('5-14 ended')










#【例5-15】基于midwest_filter数据集绘制带边界的气泡图。从应该环绕的数据框中获取记录，并用encircle()
#来使边界显示出来。输入代码如下：
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# 步骤 1：准备数据
midwest = pd.read_csv("S:\Desktop_new\caxbook_python\python_plot_202405\PyData\midwest_filter.csv")  # 导入数据

# 每个唯一 midwest['category']对应一个颜色
categories = np.unique(midwest['category'])  # 获取唯一的类别
colors = [plt.cm.tab10(i / float(len(categories) - 1)) for i in
          range(len(categories))]  # 为每个类别选择颜色

# 步骤 2：绘制散点图，每个类别使用唯一颜色
fig = plt.figure(figsize=(10, 6), dpi=80, facecolor='w',
                 edgecolor='k')  # 创建图形

for i, category in enumerate(categories):
    plt.scatter('area', 'poptotal',
                data=midwest.loc[midwest.category == category, :], s='dot_size',
                c=colors[i], label=str(category), edgecolors='black',
                linewidths=.5, alpha=0.5)  # 绘制气泡图


# 步骤 3：绘制围绕数据点的多边形
def encircle(x, y, ax=None, **kw):
    if not ax: ax = plt.gca()
    p = np.c_[x, y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices, :], **kw)
    ax.add_patch(poly)


# 选择要绘制圈的数据
midwest_encircle_data = midwest.loc[midwest.state == 'IN', :]

# 绘制围绕数据点的多边形
encircle(midwest_encircle_data.area, midwest_encircle_data.poptotal,
         ec="k", fc="gold", alpha=0.1)
encircle(midwest_encircle_data.area, midwest_encircle_data.poptotal,
         ec="firebrick", fc="none", linewidth=1.5)

# 步骤 4：修饰图形
plt.gca().set(xlim=(0.0, 0.1), ylim=(0, 90000),
              xlabel='Area', ylabel='Population')  # 设置坐标轴的范围和标签

plt.xticks(fontsize=12);
plt.yticks(fontsize=12)  # 设置刻度的字体大小
plt.title("Scatterplot of Midwest Area vs Population",
          fontsize=18)  # 设置标题
plt.legend(fontsize=12)  # 显示图例
plt.show()






















#【例5-16】绘制极坐标下的气泡图。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19781101)  # 固定随机种子，以便结果可复现
# 计算面积和颜色
N = 150
r = 2 * np.random.rand(N)
theta = 2 * np.pi * np.random.rand(N)
area = 200 * r ** 2
colors = theta

# 创建第1个图形
fig = plt.figure()
ax = fig.add_subplot(projection='polar')
c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)

# 创建第2个图形
fig = plt.figure()
ax = fig.add_subplot(projection='polar')
c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)
# 设置极坐标原点的位置
ax.set_rorigin(-2.5)
ax.set_theta_zero_location('W', offset=10)
plt.show()  # 显示第1个图形

# 创建第3个图形
fig = plt.figure()
ax = fig.add_subplot(projection='polar')
c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)
# 设置极坐标角度的范围
ax.set_thetamin(45)
ax.set_thetamax(135)






















# 5.5 等高线图
#     等高线图（Contour Plot），也称为等值线图或等高图，是一种用于可视化二维数据的图
# 表类型。它通过绘制等高线来表示数据的变化和分布，将相同数值的数据点连接起来形成曲
# 线，以展示数据的等值线和梯度。
#     等高线图能够直观地显示二维数据的变化和分布，帮助观察数据的轮廓、梯度和峰值。
# 它可以揭示数据的高低区域、变化趋势以及相邻区域之间的差异。

#【例5-17】绘制等高线图示例1。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

levels = np.linspace(-1, 1, 20)  # 定义等高线水平
plt.contour(X, Y, Z, levels=levels)  # 创建等高线图，并指定等级

# 添加标签和标题
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.title('指定等级的等高线图')

plt.show()






















#【例5-18】绘制等高线图示例2。输入代码如下：
import numpy as np
import matplotlib.pyplot as plt

delta = 0.0125  # 减小delta值，增加网格的间距
x = np.arange(-3.0, 3.0, delta)  # 定义x范围为-3到3，间隔为delta
y = np.arange(-2.0, 2.0, delta)  # 定义y范围为-2到2，间隔为delta
X, Y = np.meshgrid(x, y)  # 生成网格坐标矩阵

Z1 = np.exp(-X ** 2 - Y ** 2)  # 生成第1个二元正态分布数据
Z2 = np.exp(-(X - 1.5) ** 2 - (Y - 0.5) ** 2)  # 生成第2个二元正态分布数据

# 高斯差分
Z = 10.0 * (Z2 - Z1)  # 计算两个正态分布之间的差分值乘以10

plt.figure()  # 创建新的图形窗口
CS = plt.contour(X, Y, Z)  # 绘制等高线图
plt.clabel(CS, inline=1, fontsize=10)  # 在等高线上添加标签，默认位置
plt.title('Simplest default with labels')  # 设置标题
plt.show()






















#【例5-19】绘制等高线图示例3。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri

np.random.seed(19781101)  # 固定随机种子，以便结果可复现
npts = 200
ngridx = 100
ngridy = 200
x = np.random.uniform(-2, 2, npts)
y = np.random.uniform(-2, 2, npts)
z = x * np.exp(-x ** 2 - y ** 2)

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6, 4), )

# 在网格上进行插值。通过在网格上进行插值，绘制不规则数据坐标的等高线图。
# 首先创建网格值。
xi = np.linspace(-2.1, 2.1, ngridx)
yi = np.linspace(-2.1, 2.1, ngridy)

# 在由 (xi,yi)定义的网格上线性插值数据 (x,y)。
triang = tri.Triangulation(x, y)
interpolator = tri.LinearTriInterpolator(triang, z)
Xi, Yi = np.meshgrid(xi, yi)
zi = interpolator(Xi, Yi)

# 注意，scipy.interpolate 提供了在网格上进行数据插值的方法。
# 下面的代码是对上面四行代码的替代写法：
# from scipy.interpolate import griddata
# zi=griddata((x,y),z,(xi[None,:],yi[:,None]),method='linear')

ax1.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
cntr1 = ax1.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")

fig.colorbar(cntr1, ax=ax1)
ax1.plot(x, y, 'ko', ms=2)
ax1.set(xlim=(-2, 2), ylim=(-2, 2))
ax1.set_title('grid and contour (%d points,%d grid points）' %
              (npts, ngridx * ngridy))

# 三角剖分等高线图
# 直接将无序的、不规则间隔的坐标提供给tricontour
ax2.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
cntr2 = ax2.tricontourf(x, y, z, levels=14, cmap="RdBu_r")

fig.colorbar(cntr2, ax=ax2)
ax2.plot(x, y, 'ko', ms=2)
ax2.set(xlim=(-2, 2), ylim=(-2, 2))
ax2.set_title('tricontour (%d points' % npts)

plt.subplots_adjust(hspace=0.5)
plt.show()





















'''
# matplotlib == 3.3.0!!!
#【例5-20】使用proplot库创建等高线图。输入代码如下：
import xarray as xr
import numpy as np
import pandas as pd
import proplot as pplt

# DataArray
state = np.random.RandomState(51423)
# 生成20*20的随机数据
linspace = np.linspace(0, np.pi, 20)
data = 50 * state.normal(1, 0.2, size=(20, 20)) * (np.sin(linspace * 2) ** 2
                                                   * np.cos(linspace + np.pi / 2)[:, None] ** 2)
# 创建纬度数据
lat = xr.DataArray(np.linspace(-90, 90, 20), dims=('lat',),
                   attrs={'units': '\N{DEGREE SIGN}N'})  # 添加属性，单位为°N
# 创建压力数据
plev = xr.DataArray(np.linspace(1000, 0, 20), dims=('plev',),
                    attrs={'long_name': 'pressure',
                           'units': 'hPa'})  # 添加属性，长名为pressure，单位为hPa
# 创建DataArray
da = xr.DataArray(data, name='u',  # DataArray的名称
                  dims=('plev', 'lat'),  # 维度
                  coords={'plev': plev, 'lat': lat},  # 坐标
                  attrs={'long_name': 'zonal wind',
                         'units': 'm/s'})  # 添加属性，长名为zonal wind，单位为m/s

# 数据框
data = state.rand(12, 20)  # 生成12*20的随机数据
# 对数据进行累积和运算，并取反
df = pd.DataFrame((data - 0.4).cumsum(axis=0).cumsum(axis=1)[::1, ::-1],
                  index=pd.date_range('2000-01', '2000-12',
                                      freq='MS'))  # 设置索引为2000年1月到12月
df.name = 'temperature (\N{DEGREE SIGN}C)'  # DataFrame的名称，单位为°C
df.index.name = 'date'  # 设置索引名称为date
df.columns.name = 'variable (units)'  # 设置列名称为variable (units)

# 创建图形
fig = pplt.figure(refwidth=2.5, share=False,
                  suptitle='Automatic subplot formatting')

# 绘制DataArray
cmap = pplt.Colormap('PuBu', left=0.05)  # 设置颜色映射
ax = fig.subplot(121, yreverse=True)  # 在图形中添加子图，y轴反向
ax.contourf(da, cmap=cmap, colorbar='t', lw=0.7, ec='k')  # 绘制填充等值线图

# 绘制DataFrame
ax = fig.subplot(122, yreverse=True)  # 在图形中添加子图，y轴反向
ax.contourf(df, cmap='YlOrRd', colorbar='t', lw=0.7, ec='k')  # 绘制填充等值线图
ax.format(xtickminor=False, yformatter='%b',
          ytickminor=False)  # 设置格式，禁用次要刻度，y轴日期格式为月份的英文缩写
plt.show()
'''






















#【例5-21】利用bokeh包绘制等高线图，展示在极坐标网格上的二维正弦波数据。输入代码如下：
import numpy as np
from bokeh.palettes import Cividis
from bokeh.plotting import figure, show

# 创建极坐标网格上的二维正弦波数据
radius, angle = np.meshgrid(np.linspace(0, 1, 20),
                            np.linspace(0, 2 * np.pi, 120))
x = radius * np.cos(angle)
y = radius * np.sin(angle)
z = 1 + np.sin(3 * angle) * np.sin(np.pi * radius)

p = figure(width=550, height=400)  # 创建Bokeh图表对象
levels = np.linspace(0, 2, 11)  # 设置等高线的级别

# 绘制等高线并指定填充颜色、填充图案、线条颜色、线条样式和线条宽度
contour_renderer = p.contour(
    x=x, y=y, z=z, levels=levels,
    fill_color=Cividis,  # 设置填充颜色为Cividis调色板
    hatch_pattern=["x"] * 5 + [" "] * 5,  # 设置填充图案
    hatch_color="white",  # 设置填充图案的颜色
    hatch_alpha=0.5,  # 设置填充图案的透明度
    line_color=["white"] * 5 + ["black"] + ["red"] * 5,  # 设置线条颜色
    line_dash=["solid"] * 6 + ["dashed"] * 5,  # 设置线条样式
    line_width=[1] * 6 + [2] * 5, )  # 设置线条宽度

# 构建颜色条并添加到图表中
colorbar = contour_renderer.construct_color_bar(title="Colorbar title")
p.add_layout(colorbar, "right")
show(p)







print('5-21 ended')












# 5.6 三元相图
#     三元相图（Ternary Plot），也称为三角图，是一种用于可视化三个相互关联的变量之间的
# 比例、组合或分布关系的图表类型。它使用一个等边三角形作为坐标系，每个顶点代表一个
# 变量，而内部的点表示各个变量之间的相对比例或组合。
#     三元相图的优点是能够直观地显示三个变量之间的比例关系、组合关系或分布模式。它
# 可以帮助观察数据在三个维度上的相对权重、特征差异或聚集情况。三元相图可以通过R的
# ggtern、vcd、grid、ggplot2 等包绘制。
#     三元相图适用于比例性或组合性的数据，不适用于连续变量或离散变量。由于三元相图
# 的坐标轴是固定的，数据点的位置受到限制，因此需要注意数据点的范围和分布，以确保数
# 据能够充分展示在图表中。

#【例5-22】使用mpltern库创建三元相图，用于显示不同参数设置下的Dirichlet分布的概率密度函数（PDF）。输入代码如下：
import matplotlib.pyplot as plt
from mpltern.datasets import get_dirichlet_pdfs

# 创建一个大图，并设置子图之间的间距
fig = plt.figure(figsize=(10.8, 8.8))
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9,
                    wspace=0.5, hspace=0.5, )

# 设置不同的Dirichlet分布参数
alphas = ((1.5, 1.5, 1.5), (5.0, 5.0, 5.0), (1.0, 2.0, 2.0), (2.0, 4.0, 8.0))

# 在每个子图中绘制对应参数设置下的Dirichlet分布
for i, alpha in enumerate(alphas):
    # 添加子图，使用三角图的投影
    ax = fig.add_subplot(2, 2, i + 1, projection="ternary")
    # 获取Dirichlet分布的PDF数据
    t, l, r, v = get_dirichlet_pdfs(n=61, alpha=alpha)
    # 绘制填充颜色表示PDF
    cmap = "Blues"
    shading = "gouraud"
    cs = ax.tripcolor(t, l, r, v, cmap=cmap, shading=shading, rasterized=True)
    # 绘制等高线以更清晰地显示PDF的形状
    ax.tricontour(t, l, r, v, colors="k", linewidths=0.5)
    # 设置轴标签
    ax.set_tlabel("$x_1$")
    ax.set_llabel("$x_2$")
    ax.set_rlabel("$x_3$")
    # 将轴标签放在三角图的内部
    ax.taxis.set_label_position("tick1")
    ax.laxis.set_label_position("tick1")
    ax.raxis.set_label_position("tick1")
    # 设置子图标题，显示参数设置
    ax.set_title("${\\mathbf{\\alpha}}$=" + str(alpha))
    # 添加颜色条，显示PDF的颜色对应的数值
    cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
    colorbar = fig.colorbar(cs, cax=cax)
    colorbar.set_label("PDF", rotation=270, va="baseline")
plt.show()





print('5-22 ended!')















# 5.7 瀑布图
#     瀑布图（Waterfall Chart）是一种用于可视化数据的累积效果和变化情况的图表类型。它
# 通过一系列的矩形条表示数据的起始值、各个增减项和最终值，以展示数据在不同阶段的增
# 减和总体变化。
#     瀑布图能够直观地显示数据在不同阶段的增减和累积效果。它可以帮助观察数据的变化
# 趋势、识别主要增减项，并对总体变化进行可视化。
#     注意：根据数据的特点和目的，可以使用不同的颜色来表示正增减项和负增减项，以增
# 加图表的对比度和可读性。确保矩形条的起始位置正确表示前一阶段的累积值，高度准确表
# 示增减的数值大小。

#【例5-23】绘制瀑布图。输入代码如下：
import plotly.graph_objects as go

# 示例 1:创建一个简单的瀑布图
fig = go.Figure(go.Waterfall(
    name="20",  # 瀑布图的名称
    orientation="v",  # 方向为垂直
    measure=["relative", "relative", "total", "relative", "relative",
             "total"],  # 每个数据的类型：增量、总计等
    x=["Sales", "Consulting", "Net revenue", "Purchases", "Other expenses",
       "Profit before tax"],  # x轴标签
    textposition="outside",  # 文本位置
    text=["+60", "+80", "", "-40", "-20", "Total"],  # 显示在瀑布图上的文本
    y=[60, 80, 0, -40, -20, 0],  # y轴数据
    connector={"line": {"color": "rgb(63,63,63)"}},  # 连接线的样式
))
fig.update_layout(title="Profit and loss statement 2018",  # 设置图表标题
                  showlegend=True)  # 显示图例
fig.show()

# 示例 2:创建一个分组瀑布图
fig = go.Figure()

# 添加第1个瀑布图
fig.add_trace(go.Waterfall(
    x=[["2016", "2017", "2017", "2017", "2017", "2018", "2018", "2018", "2018"],
       ["initial", "q1", "q2", "q3", "total", "q1", "q2", "q3", "total"]],
    measure=["absolute", "relative", "relative", "relative", "total",
             "relative", "relative", "relative", "total"],
    y=[1, 2, 3, -1, None, 1, 2, -4, None],
    base=1000))

# 添加第2个瀑布图
fig.add_trace(go.Waterfall(
    x=[["2016", "2017", "2017", "2017", "2017", "2018", "2018", "2018", "2018"],
       ["initial", "q1", "q2", "q3", "total", "q1", "q2", "q3", "total"]],
    measure=["absolute", "relative", "relative", "relative", "total",
             "relative", "relative", "relative", "total"],
    y=[1.1, 2.2, 3.3, -1.1, None, 1.1, 2.2, -4.4, None],
    base=1000))
fig.update_layout(waterfallgroupgap=0.5, )  # 分组瀑布图之间的间隔
fig.show()

# 示例 3:创建一个水平方向的瀑布图
fig = go.Figure(go.Waterfall(
    name="2018",  # 瀑布图的名称
    orientation="h",  # 方向为水平
    measure=["relative", "relative", "relative", "relative",
             "total", "relative", "relative", "relative", "relative",
             "total", "relative", "relative", "total",
             "relative", "total"],  # 每个数据的类型：增量、总计等
    y=["Sales", "Consulting", "Maintenance", "Other revenue",
       "Net revenue", "Purchases", "Material expenses", "Personnel expenses",
       "Other expenses", "Operating profit", "Investment income",
       "Financial income", "Profit before tax", "Income tax (15%)",
       "Profit after tax"],  # y轴标签
    x=[375, 128, 78, 27, None, -327, -12, -78, -12, None,
       32, 89, None, -45, None],  # x轴数据
    connector={"mode": "between", "line": {"width": 4, "color": "rgb(0,0,0)",
                                           "dash": "solid"}}  # 连接线的样式
))
fig.update_layout(title="Profit and loss statement 2018")  # 设置图表标题
fig.show()




















# 5.8 生存曲线图
#     生存曲线图（Survival Curve）是用于描述在一段时间内生存下来的个体或实体的比例。
# 它通常用于生存分析领域，如医学、流行病学和可靠性工程等。在医学领域中，生存曲线常
# 用于分析患者的生存时间，比如研究特定治疗方法对患者生存率的影响。
#     生存曲线图的横轴表示经过的时间，纵轴表示生存率（或存活率）。曲线的形状和趋势
# 可以告诉我们在不同时间点上个体或实体的生存情况。在生存曲线图中，经常会看到以下两
# 种曲线：
#     （1）Kaplan-Meier 曲线：Kaplan-Meier 曲线是用来估计生存函数的非参数方法。它根据
# 样本中存活时间的数据绘制生存曲线，没有假设数据分布的情况下估计生存率。
#     （2）Cox 比例风险模型曲线：Cox 比例风险模型是用来评估某个因素对生存率的影响
# 的方法。其曲线表示了在考虑其他因素影响下，某一因素对生存率的影响程度。
#     通常情况下，生存曲线图中还会显示置信区间，以反映估计的不确定性范围。在Python
# 中，使用Plotly库可以绘制生存曲线图，可以直观地展示不同类别下的生存率。

#【例5-24】绘制生存曲线图。输入代码如下：
import plotly.graph_objects as go  # 导入绘图工具包
import plotly.express as px  # 导入绘图工具包
import numpy as np  # 导入数值计算工具包
import pandas as pd  # 导入数据处理工具包
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型
from sklearn.metrics import roc_curve, roc_auc_score

# 导入ROC曲线相关的评估指标

np.random.seed(0)  # 固定随机种子，以便结果可复现

# 人为地添加噪音，使任务更加困难
df = px.data.iris()  # 加载鸢尾花数据集
samples = df.species.sample(n=50, random_state=0)
# 从'species'列中随机抽取50个样本
np.random.shuffle(samples.values)  # 打乱样本的顺序
df.loc[samples.index, 'species'] = samples.values
# 将打乱后的样本顺序赋值给数据集

# 定义输入和输出
X = df.drop(columns=['species', 'species_id'])  # 提取特征
y = df['species']  # 提取目标变量

# 拟合模型
model = LogisticRegression(max_iter=200)  # 创建逻辑回归模型
model.fit(X, y)  # 对模型进行训练
y_scores = model.predict_proba(X)  # 预测概率

# 对标签进行独热编码以便绘图
y_onehot = pd.get_dummies(y, columns=model.classes_)

# 创建一个空的图形，并在每次计算新类别时添加新线
fig = go.Figure()  # 创建一个绘图对象
fig.add_shape(type='line', line=dict(dash='dash'),
              x0=0, x1=1, y0=0, y1=1)

for i in range(y_scores.shape[1]):
    y_true = y_onehot.iloc[:, i]
    y_score = y_scores[:, i]

    fpr, tpr, _ = roc_curve(y_true, y_score)  # 计算ROC曲线的假阳率和真阳率
    auc_score = roc_auc_score(y_true, y_score)  # 计算AUC值

    name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name,
                             mode='lines'))  # 绘制ROC曲线

fig.update_layout(
    xaxis_title='False Positive Rate',  # 设置X轴标题
    yaxis_title='True Positive Rate',  # 设置Y轴标题
    yaxis=dict(scaleanchor="x", scaleratio=1),  # 设置Y轴与X轴的比例相同
    xaxis=dict(constrain='domain'),  # 约束X轴范围在0到1之间
    width=500, height=500,  # 设置图形的宽度和高度
    # legend=dict(x=0.5,y=0.1,bgcolor='rgba(255,255,255,0.5)')
    # 将图例移动到图的内部，并设置背景色为半透明白色
)
fig.show()





















# 5.9 火山图
#     火山图（Volcano Plot）是一种用于展示两组样本之间的差异性和统计显著性。它常用于
# 基因表达分析和差异分析等领域。火山图的主要特点是将差异度量和统计显著性结合在一个
# 图中，以直观地显示基因或变量之间的差异程度和显著性水平。它的名称源自其外观形状类
# 似于火山喷发的形状。
#     火山图的横轴表示差异度量，通常是基因表达水平或其他衡量指标的对数倍数变化（如
# log fold change）。纵轴表示统计显著性，常用的度量是调整的p-value（经过多重检验校正后
# 的p-value）或其他显著性指标。每个基因或变量在图中以一个点的形式表示。
#     在火山图中，显著差异的基因或变量通常位于图的两侧，并且在图中表现为离中心轴较
# 远的点。这意味着它们在差异度量上有较大的变化，并且具有较低的统计显著性。与之相反，
# 非显著的基因或变量通常集中在中心轴附近。
#     绘制火山图的过程通常涉及对差异度量和统计显著性进行计算，并使用适当的软件或编
# 程工具来生成图形。

#【例5-25】绘制火山图。输入代码如下：
from bioinfokit import analys, visuz  # 导入bioinfokit库中的analys和visuz

# 从 pandas dataframe 中加载数据集
df = analys.get_data('volcano').data
df.head(2)  # 输出数据集的前两行，以确认数据加载成功

# 绘制火山图，保存为 volcano.png
visuz.GeneExpression.volcano(df=df, lfc='log2FC', pv='p-value')
# 如果想直接显示图像而不是保存，设置show=True参数

# 添加图例，并指定位置和锚点
visuz.GeneExpression.volcano(df=df, lfc='log2FC', pv='p-value',
                             plotlegend=True, legendpos='upper right', legendanchor=(1.46, 1))

# 更改颜色映射，指定折点和显著性阈值，并添加图例
visuz.GeneExpression.volcano(df=df, lfc='log2FC', pv='p-value',
                             lfc_thr=(1, 2), pv_thr=(0.05, 0.01), plotlegend=True,
                             color=("#00239CFF", "grey", "#E10600FF"), legendpos='upper right',
                             legendanchor=(1.46, 1))

# 指定透明度，并绘制火山图
visuz.GeneExpression.volcano(df=df, lfc='log2FC', pv='p-value',
                             color=("#00239CFF", "grey", "#E10600FF"), valpha=0.5)

# 添加基因自定义标签，并绘制火山图
visuz.GeneExpression.volcano(df=df, lfc="log2FC", pv="p-value",
                             geneid="GeneNames",
                             genenames=("LOC_Os09g01000.1", "LOC_Os01g50030.1",
                                        "LOC_Os06g40940.3", "LOC_Os03g03720.1"))
# 如果想要标记所有差异表达基因 (DEGs)，设置 genenames='deg'

# 添加基因自定义标签，并绘制火山图，指定标签样式、阈值线、坐标轴范围等参数
visuz.GeneExpression.volcano(df=df, lfc="log2FC", pv="p-value",
                             geneid="GeneNames",
                             genenames=({"LOC_Os09g01000.1": "EP", "LOC_Os01g50030.1": "CPuORF25",
                                         "LOC_Os06g40940.3": "GDH", "LOC_Os03g03720.1": "G3PD"}),
                             gstyle=2, sign_line=True, xlm=(-6, 6, 1), ylm=(0, 61, 5), figtype='svg',
                             axtickfontsize=10, axtickfontname='Verdana')
