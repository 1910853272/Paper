# 十、多维数据可视化
#     多维数据可视化是一种数据分析和数据呈现的方法，多维数据通常包含多个变量或维度，
# 可能是数值型、分类型、时间序列等不同类型的数据。本章将介绍多热图、矩阵散点图和平
# 行坐标图等多维数据可视化的关键方法，旨在帮助读者更清晰地理解数据中的关系、趋势和
# 模式。通过本章的学习，读者可以掌握这些可视化方法，并将其应用于实际数据分析中，以
# 更深入地理解和解释复杂的多维数据集。

# 10.1 热图
#     热图（Heatmap）是一种用于可视化矩阵数据的图表类型。它通过使用颜色编码来表示
# 数据的大小或值，以便在二维空间中显示数据的模式、趋势或关联性。
#     热图的优点是能够直观地显示数据的模式和趋势，并帮助观察数据的关联性和相似性。
# 它常用于分析多变量数据、基因表达数据、市场趋势分析等。
#     在绘制和解读热图时，需确保颜色编码的准确并合理，以避免造成误解。当矩阵数据较
# 大时，可以使用矩阵聚类和排序方法，以便更好地展示数据的模式和关联性；当矩阵数据具
# 有缺失值时，可以使用适当的填充或插值方法进行处理，以确保图表的完整性和可靠性。

#【例10-1】利用Pandas、Seaborn和Matplotlib库，对汽车数据集进行了处理和可视化分析。输入代码如下：
import pandas as pd  # 导入pandas库，用于数据处理和分析
import seaborn as sns  # 导入seaborn库，用于数据可视化
import matplotlib.pyplot as plt  # 导入matplotlib库用于绘图

# 导入数据集
df = pd.read_csv(r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mtcars.csv")
# 读取"mtcars.csv"的数据文件，并将数据存储在df中

# 删除非数值型的列
df_numeric = df.select_dtypes(include=['float64', 'int64'])
# 选择数据集中的数值型列（包括float64和int64类型），并存储在df_numeric中

# 绘制热力图，显示数值型列之间的相关性，使用'viridis'颜色映射，将相关性系数标注在图上
plt.figure(figsize=(10, 5), dpi=200)  # 创建图形对象
sns.heatmap(df_numeric.corr(), xticklabels=df_numeric.corr().columns,
            yticklabels=df_numeric.corr().columns,
            cmap='viridis', center=0, annot=True)  # 使用Seaborn绘制热力图

# 添加修饰
plt.title('Correlogram of mtcars', fontsize=18)  # 设置图形标题
plt.xticks(fontsize=12)  # 设置x轴标签的字体大小为12
plt.yticks(fontsize=12, rotation=0)  # 设置y轴标签的字体大小为12
plt.show()































#【例10-2】使用Seaborn库绘制聚类热图，展示脑网络之间的相关性，并使用分类调色板来标识不同的网络。输入代码如下：
import pandas as pd
import seaborn as sns

sns.set_theme()

# 加载示例数据集
df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)
# 选择网络的子集
used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
used_columns = (df.columns.get_level_values("network")
                .astype(int)
                .isin(used_networks))
df = df.loc[:, used_columns]

# 创建一个分类调色板以识别网络
network_pal = sns.husl_palette(8, s=.45)
network_lut = dict(zip(map(str, used_networks), network_pal))

# 将调色板转换为向量，将在矩阵的侧面绘制
networks = df.columns.get_level_values("network")
network_colors = pd.Series(networks, index=df.columns).map(network_lut)

# 绘制完整的图形
g = sns.clustermap(df.corr(), center=0, cmap="vlag",
                   row_colors=network_colors, col_colors=network_colors,
                   dendrogram_ratio=(.1, .2),
                   cbar_pos=(.02, .32, .03, .2),
                   linewidths=.75, figsize=(12, 13))
g.ax_row_dendrogram.remove()
































#【例10-3】利用mlxtend绘制热图。输入代码如下：
from mlxtend.plotting import heatmap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(19781101)  # 固定随机种子，以便结果可复现
some_array = np.random.random((15, 20))
heatmap(some_array, figsize=(20, 10))
plt.show()

heatmap(some_array, figsize=(15, 8), cell_values=False)
plt.show()































#【例10-5】利用mlxtend.plotting库中的heatmap函数和Matplotlib库来绘制热图，展示房价数据集中选定特征之间的相关性。输入代码如下：
from mlxtend.plotting import heatmap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 加载房价数据集，并设置列名
df = pd.read_csv(r'S:\Desktop_new\caxbook_python\python_plot_202405\PyData\housing.data.txt',
                 header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head(2)  # 输出略

from matplotlib import cm  # 从Matplotlib库中导入颜色映射

cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']  # 选择感兴趣的列

# 计算所选列的相关系数矩阵
corrmat = np.corrcoef(df[cols].values.T)
# 绘制热图并指定行名和列名
fig, ax = heatmap(corrmat, column_names=cols, row_names=cols, cmap=cm.PiYG)

# 将颜色条范围设置为-1到1
for im in ax.get_images():
    im.set_clim(-1, 1)
plt.show()































#【例10-6】使用Matplotlib库绘制热图，展示不同农场主种植的蔬菜收成量。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np

vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

fig, ax = plt.subplots(figsize=(5, 5))  # 创建图形和轴对象
im = ax.imshow(harvest)  # 绘制热图

# 显示所有刻度，并使用对应的列表条目进行标注
ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)

# 旋转刻度标签并设置对齐方式
plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
         rotation_mode="anchor")

# 循环遍历数据维度并创建文本注释
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j], ha="center", va="center", color="w")

# 设置标题和调整布局
ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()
plt.show()





























# 10.2 矩阵散点图
#     矩阵散点图（Matrix Scatter Plot）是一种用于可视化多个变量之间的关系的图表。它通
# 过在一个矩阵中绘制多个散点图的组合来展示变量之间的相互作用和相关性。
#     矩阵散点图的优点是可以同时展示多个变量之间的关系，帮助我们观察和发现不同变量
# 之间的模式、趋势和相关性。通过矩阵散点图，我们可以更全面地了解变量之间的相互作用，
# 发现潜在的关联和趋势。
#     然而，当变量数量较多时，矩阵散点图可能会变得复杂且难以解读。因此，在使用矩阵
# 散点图时，应谨慎选择变量数量，并根据可视化目的选择合适的变量和展示方式，以确保图
# 表的可读性和准确传达变量之间的关系。
#     在矩阵散点图中，每个变量都会沿着轴的方向占据一行或一列，图中的每个小格子代表
# 一个变量对。对角线上的小格子通常展示的是该变量自身的分布情况，而其他位置的小格子
# 则展示了两个变量之间的关系，即散点图。通过观察这些散点图，可以直观地了解各个变量
# 之间的相关性、趋势以及异常值。
#     矩阵散点图对于探索性数据分析非常有用，因为它能够同时显示多个变量之间的关系，
# 帮助研究人员发现变量之间的相互作用和规律。在数据集中存在多个变量时，使用矩阵散点
# 图可以更全面地了解数据的特征，并为进一步的分析提供线索。
#     利用seaborn 库中的pairplot()函数可以绘制矩阵散点图。


#【例10-7】绘制一个散点矩阵图，展示iris数据集中各个特征两两之间的关系，并根据花的种类进行了着色区分。
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块
import seaborn as sns  # 导入seaborn模块

df = sns.load_dataset('iris')  # 加载iris数据集
sns.pairplot(df, kind="reg")  # 绘制带回归线的矩阵散点图
plt.show()

sns.pairplot(df, kind="scatter")  # 绘制不带回归线的矩阵散点图
plt.show()

# 绘制左边图：矩阵散点图，并按照'species'列进行着色
sns.pairplot(df, kind="scatter", hue="species",  # hue定义数据子集的变量
             markers=["o", "s", "D"],  # markers标记形状列表
             palette="Set2")  # palette用于映射色调变量的颜色集
plt.show()

# 绘制右边图：矩阵散点图，并按照'species'列进行着色，同时设置绘图参数
sns.pairplot(df, kind="scatter", hue="species",
             plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
# plot_kws用于修改情节的关键字参数字典
plt.show()

sns.pairplot(df, diag_kind="kde")  # 绘制密度图矩阵
sns.pairplot(df, diag_kind="hist")  # 绘制直方图矩阵
# 将其自定义为密度图或直方图
sns.pairplot(df, diag_kind="kde",
             diag_kws=dict(shade=True, bw_adjust=.05, vertical=False))
# diag_kind设置对角子图的类型（包括'auto'、'hist'、'kde'、None）
plt.show()































#【例10-8】绘制一个散点矩阵图，展示iris数据集中各个特征两两之间的关系，并根据花的种类进行了着色区分。
import matplotlib.pyplot as plt
from mlxtend.data import iris_data
from mlxtend.plotting import scatterplotmatrix

X, y = iris_data()
scatterplotmatrix(X, figsize=(10, 8))
plt.tight_layout()
plt.show()

names = ['sepal length [cm]', 'sepal width [cm]',
         'petal length [cm]', 'petal width [cm]']

fig, axes = scatterplotmatrix(X[y == 0], figsize=(10, 8), alpha=0.5)
fig, axes = scatterplotmatrix(X[y == 1], fig_axes=(fig, axes), alpha=0.5)
fig, axes = scatterplotmatrix(X[y == 2], fig_axes=(fig, axes), alpha=0.5,
                              names=names)
plt.tight_layout()
plt.show()




























# 10.3 平行坐标图
#     平行坐标图（Parallel Coordinate Plot）是一种用于可视化多个连续变量之间关系的图表。
# 它通过在一个平行的坐标系中绘制多条平行的线段来表示每个数据点在各个变量上的取值。
# 通过观察线段之间的交叉和趋势，可以揭示变量之间的关系。
#     平行坐标图方便观察和比较不同变量之间的趋势和关系，适用于可视化较多的连续变量。
# 然而，当变量数量较多时，图形可能变得复杂且难以解读。因此，在使用平行坐标图时，应
# 选择合适的变量和展示方式，以确保图表的可读性和有效地传达变量之间的关系。
#     在平行坐标图中，每个连续变量被表示为一个垂直的轴线，连接各个轴线的折线表示每
# 个数据点。通过观察折线的形状和走势，可以推断出不同变量之间的关系和趋势。
#     平行坐标图适用于比较多个连续变量之间的关系和差异，观察变量之间的交叉效应和相
# 互影响，检测异常值和离群点，以及识别数据聚类和模式。它具有可以同时可视化多个连续
# 变量的优点，提供更全面的数据视图，发现变量之间的相关性和趋势，支持比较不同数据点
# 之间的差异和相似性。

#【例10-9】利用matplotlib包绘制平行坐标图1。输入代码如下：
# 导入绘制平行坐标图所需的库
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt

# 导入鸢尾花数据集
from sklearn import datasets
import pandas as pd

# 载入鸢尾花数据集
iris = datasets.load_iris()
# 将数据集转换为DataFrame格式，仅保留前四个特征，并为列命名
X = pd.DataFrame(iris.data[:, :4],
                 columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
X['class'] = iris.target  # 添加鸢尾花的类别列

parallel_coordinates(X, 'class')  # 绘制平行坐标图
plt.show()































#【例10-10】利用matplotlib包绘制平行坐标图2。输入代码如下：
from pandas.plotting import parallel_coordinates
import pandas as pd
import matplotlib.pyplot as plt

df_final = pd.read_csv(r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\diamonds_filter.csv")  # 导入数据

# 绘制平行坐标图
plt.figure(figsize=(10, 6), dpi=200)
parallel_coordinates(df_final, 'cut', colormap='Dark2')

# 设置边框透明度
plt.gca().spines["top"].set_alpha(0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(0)
plt.gca().spines["left"].set_alpha(.3)

plt.title('Parallel Coordinated of Diamonds', fontsize=18)  # 设置标题
plt.grid(alpha=0.3)  # 设置网格线透明度
plt.xticks(fontsize=12)  # 设置x轴刻度标签字体大小
plt.yticks(fontsize=12)  # 设置y轴刻度标签字体大小
plt.show()































#【例10-11】利用altair包绘制平行坐标图1。输入代码如下：
import altair as alt
from vega_datasets import data
from altair import datum

source = data.iris()  # 加载鸢尾花数据集
# 创建图表，并进行数据转换
alt.Chart(source).transform_window(
    index='count()'  # 在数据集中添加一个索引列，用于绘制多条线
).transform_fold(
    ['petalLength', 'petalWidth', 'sepalLength',
     'sepalWidth']  # 将特征列展开成长格式，以便后续处理
).transform_joinaggregate(
    min='min(value)',  # 计算每个特征的最小值
    max='max(value)',  # 计算每个特征的最大值
    groupby=['key']  # 按照特征名称分组
).transform_calculate(
    minmax_value=(datum.value - datum.min) / (datum.max - datum.min),
    # 计算每个特征的最小-最大归一化值
    mid=(datum.min + datum.max) / 2  # 计算每个特征的中间值
).mark_line().encode(
    x='key:N',  # X轴为特征名称
    y='minmax_value:Q',  # Y轴为最小-最大归一化值
    color='species:N',  # 颜色编码为鸢尾花的类别
    detail='index:N',  # 详细编码为索引列，用于绘制多条线
    opacity=alt.value(0.5)  # 设置线条透明度
).properties(width=500)  # 设置图表宽度为500像素


































#【例10-12】利用altair包绘制平行坐标图2。输入代码如下：
import altair as alt
from vega_datasets import data

source = data.iris()  # 载入数据集
# 创建图表，进行窗口转换
alt.Chart(source).transform_window(
    index='count()'  # 计算索引
).transform_fold(
    ['petalLength', 'petalWidth', 'sepalLength', 'sepalWidth']  # 折叠数据
).mark_line().encode(
    x='key:N',  # x轴
    y='value:Q',  # y轴
    color='species:N',  # 颜色编码
    detail='index:N',  # 详细信息编码
    opacity=alt.value(0.5)  # 设置透明度
).properties(width=500)  # 设置图表宽度





























# 10.4 安德鲁斯曲线
#     安德鲁斯曲线（Andrews Curves）是一种用于可视化多维数据的技术，通过将每个数据
# 点映射到一个多维空间中，并将每个维度的值视为一个正弦曲线的振幅，然后将这些曲线叠
# 加在一起，从而形成安德鲁曲线。这样每个样本的曲线都会在同一个图上绘制，因此可以直
# 观地比较不同样本之间的相似性和差异性。
#     绘制安德鲁斯曲线时，对于给定的数据集，每个样本的特征值被视为一组函数的系数。
# 定义一组三角函数（通常是正弦或余弦函数）。对于每个样本，将其特征值与三角函数相乘，
# 然后将结果相加，得到一个关于角度的函数。将每个样本的角度函数绘制在同一个图上，形
# 成安德鲁斯曲线图。
#     如果两个样本在特征空间中相似，则它们的安德鲁斯曲线图在图上会有相似的形状；如
# 果两个样本在特征空间中差异较大，则它们的安德鲁斯曲线图在图上会有明显的区别。
#     安德鲁斯曲线通常用于可视化多维特征的数据集，以便直观地比较样本之间的相似性和
# 差异性。在聚类分析和异常检测等任务中，帮助理解数据的结构和分布情况。
#     在绘制安德鲁斯曲线时，可以根据特征之间的重要性和关联性，对不同的特征值进行不
# 同的缩放和权重设置，以更好地反映样本之间的差异。


#【例10-13】绘制安德鲁斯曲线。输入代码如下：
from pandas.plotting import andrews_curves
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据
df = pd.read_csv(r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mtcars1.csv")
df.drop(['cars', 'carname'], axis=1, inplace=True)

# 绘制 Andrews 曲线
plt.figure(figsize=(8, 4), dpi=80)
andrews_curves(df, 'cyl', colormap='Set1')

# 设置边框透明度
plt.gca().spines["top"].set_alpha(0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(0)
plt.gca().spines["left"].set_alpha(.3)

plt.title('Andrews Curves of mtcars', fontsize=18)  # 设置标题
plt.xlim(-3, 3)  # 设置 x 轴范围
plt.grid(alpha=0.3)  # 设置网格线透明度
plt.xticks(fontsize=12)  # 设置x轴刻度标签字体大小
plt.yticks(fontsize=12)  # 设置y轴刻度标签字体大小
plt.show()