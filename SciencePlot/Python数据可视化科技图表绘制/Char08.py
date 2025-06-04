# 八、分布式数据可视化
#     本章聚焦于分布式数据的可视化，数据的分布性质对于统计分析、模型建立以及风险评
# 估都至关重要。通过本章的学习，可以掌握如何在Python中选择适当的可视化工具和技术，
# 以更好地理解数据的分布特性，并从中获取有价值的信息。本章介绍常见的分布数据可视化，
# 包括直方图、箱线图、密度图、小提琴图、脊线图等。读者通过学习可以掌握分布式数据可
# 视化的Python实现方法。
#





# 8.1 直方图
#     直方图（Histogram）是一种用于可视化连续变量的分布情况的统计图表。直方图的主要
# 特点是通过柱状图展示连续变量在每个区间内的观测频数。横轴表示连续变量的取值范围，
# 纵轴表示频数或频率（频数除以总数）。每个柱子的高度表示该区间内的观测频数。
#     直方图常用于观察数据的分布情况，包括集中趋势、离散程度和偏态。它可以帮助我们
# 识别数据的峰值、模式、异常值以及数据的整体形态。直方图是数据探索和分析的常见工具，
# 为我们提供了对数据分布的直观认识，从而有助于做出推断和决策。
#     连续变量直方图显示给定变量的频率分布。基于类型变量对频率条进行分组，可以更好
# 地了解连续变量和类型变量。类型变量直方图显示该变量的频率分布。通过对条形图进行着
# 色，可以将分布与表示颜色的另一个类型变量相关联。

#【例8-1】使用 Matplotlib 创建不同类型的直方图。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

# 使用固定的种子创建随机数生成器，以便结果可复现
rng=np.random.default_rng(19781101)
N_points=100000
n_bins=20
# 生成两个正态分布
dist1=rng.standard_normal(N_points)
dist2=0.4*rng.standard_normal(N_points)+5

# 创建具有共享y轴的子图
fig,axs=plt.subplots(1,2,figsize=(8,4),sharey=True,tight_layout=True)

# 使用 *bins* 关键字参数设置每个子图的箱数
axs[0].hist(dist1,bins=n_bins)
axs[1].hist(dist2,bins=n_bins)

# 创建新的图形，具有共享y轴的子图
fig,axs=plt.subplots(1,2,figsize=(8,4),tight_layout=True)

# N 是每个箱中的计数，bins 是每个箱的下限
N,bins,patches=axs[0].hist(dist1,bins=n_bins)

# 通过高度对颜色进行编码，但您可以使用任何标量
fracs=N/N.max()

# 将数据归一化为 0 到 1 的范围，以适应色彩映射的完整范围
norm=colors.Normalize(fracs.min(),fracs.max())

#  遍历对象，并相应地设置每个对象的颜色
for thisfrac,thispatch in zip(fracs,patches):
    color=plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
# 将输入归一化为总计数
axs[1].hist(dist1,bins=n_bins,density=True)
# 格式化y轴以显示百分比
axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))

# 创建新的图形（2D直方图），具有共享x轴和y轴的子图
fig,axs=plt.subplots(1,3,figsize=(15,5),sharex=True,sharey=True,
					      tight_layout=True)
axs[0].hist2d(dist1,dist2,bins=40)			# 增加每个轴上的箱数

# 定义颜色的归一化
axs[1].hist2d(dist1,dist2,bins=40,norm=colors.LogNorm())
# 为每个轴定义自定义数量的箱
axs[2].hist2d(dist1,dist2,bins=(80,10),norm=colors.LogNorm())
plt.show()




















#【例8-2】根据自行生成一个正态分布的样本数据，绘制直方图以及对应的拟合曲线。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np

rng=np.random.default_rng(19781101)	# 设置随机数生成器的种子
# 示例数据
mu=106  				# 分布的均值
sigma=17  			# 分布的标准差
x=rng.normal(loc=mu,scale=sigma,size=420)

num_bins=42
fig,ax=plt.subplots()			# 创建图形和坐标轴
n,bins,patches=ax.hist(x,num_bins,density=True)	# 绘制数据的直方图

# 添加拟合线
y=((1/(np.sqrt(2*np.pi)*sigma))*
     np.exp(-0.5*(1/sigma*(bins-mu))**2))
ax.plot(bins,y,'--')
ax.set_xlabel('Value')				# 设置 x 轴标签
ax.set_ylabel('Probability density')	# 设置 y 轴标签
ax.set_title('Histogram of normal distribution sample:'	# 设置标题
             fr'$\mu={mu:.0f}$,$\sigma={sigma:.0f}$')

# 调整间距以防止y轴标签被裁剪
fig.tight_layout()
plt.show()






















#【例8-3】绘制一堆叠直方图，用于显示不同车辆类型（class）的发动机排量（displ）分布情况，并为每个车辆类型分配不同的颜色。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19781101)			# 固定随机种子，以便结果可复现
# 生成第1个正态分布
mu_x=200
sigma_x=25
x=np.random.normal(mu_x,sigma_x,size=100)
# 生成第2个正态分布
mu_w=200
sigma_w=10
w=np.random.normal(mu_w,sigma_w,size=100)

fig,axs=plt.subplots(nrows=2,ncols=2)	# 创建 2x2 的子图
# 绘制步骤填充的直方图
axs[0,0].hist(x,20,density=True,histtype='stepfilled',facecolor='g',
               alpha=0.75)
axs[0,0].set_title('stepfilled')

# 绘制步骤的直方图
axs[0,1].hist(x,20,density=True,histtype='step',facecolor='g',alpha=0.75)
axs[0,1].set_title('step')

# 绘制堆叠的直方图
axs[1,0].hist(x,density=True,histtype='barstacked',rwidth=0.8)
axs[1,0].hist(w,density=True,histtype='barstacked',rwidth=0.8)
axs[1,0].set_title('barstacked')

# 创建直方图，并提供不等间距的箱体边界
bins=[100,150,180,195,205,220,250,300]
axs[1,1].hist(x,bins,density=True,histtype='bar',rwidth=0.8)
axs[1,1].set_title('bar,unequal bins')

fig.tight_layout()		# 调整布局以避免重叠
plt.show()
























#【例8-4】使用Matplotlib绘制不同类型的直方图，并对其进行堆叠和图例标注。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19781101)			# 固定随机种子，以便结果可复现
n_bins=10
x=np.random.randn(1000,3)

fig,((ax0,ax1),(ax2,ax3))=plt.subplots(nrows=2,ncols=2)	# 创建2x2的子图
colors=['red','tan','blue']	# 定义颜色列表

# 绘制带有图例的直方图
ax0.hist(x,n_bins,density=True,histtype='bar',color=colors,label=colors)
ax0.legend(prop={'size':10})
ax0.set_title('bars with legend')

# 绘制堆叠的直方图
ax1.hist(x,n_bins,density=True,histtype='bar',stacked=True)
ax1.set_title('stacked bar')

# 绘制堆叠的步骤直方图（未填充）
ax2.hist(x,n_bins,histtype='step',stacked=True,fill=False)
ax2.set_title('stack step (unfilled)')

# 绘制不同样本大小的多个直方图
x_multi=[np.random.randn(n)for n in [10000,5000,2000]]
ax3.hist(x_multi,n_bins,histtype='bar')
ax3.set_title('different sample sizes')

fig.tight_layout()			# 调整布局以避免重叠
plt.show()




























#【例8-5】绘制堆叠直方图，用于显示不同车辆类型（class）的发动机排量（displ）分布情况，并为每个车辆类型分配不同的颜色。输入代码如下：
import pandas as pd            		# 导入pandas库并简写为pd
import numpy as np               		# 导入numpy库并简写为np
import matplotlib.pyplot as plt	# 导入matplotlib.pyplot库并简写为plt

# 导入数据
df=pd.read_csv(r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mpg_ggplot2.csv")# 读取数据并存储在DataFrame对象df中
x_var='displ'     			# 指定x轴变量为发动机排量
groupby_var='class'			# 指定分组变量为车辆类型

# 根据分组变量对数据进行分组并聚合
df_agg=df.loc[:,[x_var,groupby_var]].groupby(groupby_var)

vals=[df[x_var].values.tolist()for i,df in df_agg]	# 提取每个分组的数据值
plt.figure(figsize=(8,4),dpi=250)						# 设置图形大小

# 使用色谱来为每个分组分配颜色
colors=[plt.cm.Spectral(i/float(len(vals)-1))for i in range(len(vals))]
# 绘制堆叠直方图
n,bins,patches=plt.hist(vals,30,stacked=True,density=False,
						     color=colors[:len(vals)])
# 图例
legend_dict={group:col for group,col in zip(np.unique(df[groupby_var]).tolist(),colors[:len(vals)])}
plt.legend(legend_dict)		# 添加图例

# 图形修饰
plt.title(f"${x_var}$ colored by ${groupby_var}$",fontsize=18)# 设置标题
plt.xlabel(x_var)				# 设置x轴标签
plt.ylabel("Frequency")		# 设置y轴标签
plt.ylim(0,25)				# 设置y轴范围
plt.xticks(ticks=bins[::3],
           labels=[round(b,1)for b in bins[::3]])		# 设置x轴刻度和标签
plt.show()








print('5')















#【例8-6】使用matplotlib库绘制堆叠直方图，展示不同车辆类型在发动机排量（displ）上的分布情况，不同颜色代表不同的车辆类型。输入代码如下：
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv(r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mpg_ggplot2.csv")		# 导入数据
# 准备数据
x_var='manufacturer'			# x轴变量为制造商
groupby_var='class'				# 分组变量为车辆类型
# 根据分组变量对数据进行分组并聚合
df_agg=df.loc[:,[x_var,groupby_var]].groupby(groupby_var)
# 提取每个分组的数据值
vals=[df[x_var].values.tolist()for i,df in df_agg]
plt.figure(figsize=(8,5),dpi=250)		# 创建图形
# 使用色谱来为每个分组分配颜色
colors=[plt.cm.Spectral(i/float(len(vals)-1))for i in range(len(vals))]

# 绘制堆叠直方图，其中，n为bin的数量，bins为bin的边界值，patches为patch对象
n,bins,patches=plt.hist(vals,df[x_var].unique().__len__(),
               stacked=True,density=False,color=colors[:len(vals)])

# 创建一个字典，将分组变量与对应的颜色关联起来
legend_dict={group:col for group,col in 
               zip(np.unique(df[groupby_var]).tolist(),colors[:len(vals)])}
plt.legend(legend_dict)

# 图形修饰
plt.title(f"${x_var}$ colored by ${groupby_var}$",fontsize=18)
plt.xlabel(x_var)
plt.ylabel("Frequency")
plt.ylim(0,40)									# 设置y轴的范围
plt.xticks(ticks=np.arange(len(np.unique(df[x_var]))),
           labels=np.unique(df[x_var]).tolist(),
           rotation=30,horizontalalignment='right')	# 设置x轴的刻度和标签
plt.show()





















# 8.2核密度图
#     核密度图（Kernel Density Plot）是一种用于估计连续变量的概率密度函数，并展示数据
# 的分布情况。核密度图的主要特点是通过平滑连续变量的数据分布来估计其概率密度函数。
# 它通过将每个数据点周围的核函数进行叠加，并使用适当的带宽参数来调整平滑程度，从而
# 得到连续的概率密度曲线。
#     核密度图常用于分析数据的分布形态和峰值位置，并与其他分布进行比较。它可以帮助
# 观察数据的集中趋势、峰态（比如是否呈现单峰、多峰或无峰分布）、密度变化等。通过核密
# 度图，我们可以直观地了解数据的分布特征，有助于进行数据探索、比较和推断分析。
#     在Seaborn中，可以使用seaborn.kdeplot()函数来绘制KDE图。可以绘制一维、二维或
# 多维的KDE图，还可以通过调整参数来定制图形的外观，如选择核函数、调整带宽、设置
# 颜色等。KDE图通常与其他图表（如散点图）一起使用，以提供更全面的数据分析。

#【例8-7】使用Seaborn库来绘制核密度估计图，展示鸢尾花数据集中萼片宽度和萼片长度之间的关系。输入代码如下：
import matplotlib.pyplot as plt	# 导入Matplotlib库并简写为plt
import seaborn as sns  				# 导入Seaborn库并简写为sns

df=sns.load_dataset('iris')			# 加载iris数据集
sns.set_style("white")				# 设置Seaborn的样式为"white"

sns.kdeplot(x=df.sepal_width,y=df.sepal_length)		# 绘制核密度估计图
plt.show()

# 绘制填充的核密度估计图，并使用Reds调色板
sns.kdeplot(x=df.sepal_width,y=df.sepal_length,
            cmap="Reds",fill=True)
plt.show()

# 绘制填充的核密度估计图，并使用Blues调色板，调整带宽参数为0.5
sns.kdeplot(x=df.sepal_width,y=df.sepal_length,
            cmap="Blues",fill=True,bw_adjust=0.5)
plt.show()
























#【例8-8】使用Seaborn绘制根据不同气缸数（cylinders）的城市里程（city mileage）的密度图。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mpg_ggplot2.csv")		# 导入数据
plt.figure(figsize=(10,6),dpi=80)								# 创建图形

# 对于每个气缸数，绘制对应的城市里程密度图
sns.kdeplot(df.loc[df['cyl']==4,"cty"],shade=True,color="g",
            label="Cyl=4",alpha=.7)
sns.kdeplot(df.loc[df['cyl']==5,"cty"],shade=True,color="deeppink",
            label="Cyl=5",alpha=.7)
sns.kdeplot(df.loc[df['cyl']==6,"cty"],shade=True,color="dodgerblue",
            label="Cyl=6",alpha=.7)
sns.kdeplot(df.loc[df['cyl']==8,"cty"],shade=True,color="orange",
            label="Cyl=8",alpha=.7)

# 图形修饰
plt.title('City Mileage by n_Cylinders',fontsize=16)
										# 设置标题
plt.legend()						# 显示图例
plt.xlabel('City Mileage')			# 设置x轴标签
plt.ylabel('Density')				# 设置y轴标签
plt.show()























#【例8-9】利用Seaborn库中的distplot函数绘制了不同车辆类型的城市里程密度图和直方图。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mpg_ggplot2.csv")			# 导入数据
plt.figure(figsize=(8,6),dpi=80)							# 创建图形

# 使用sns.distplot绘制密度图和直方图
# 对于每种车辆类型，绘制对应的城市里程密度图和直方图
sns.distplot(df.loc[df['class']=='compact',"cty"],color="dodgerblue",
             label="Compact",hist_kws={'alpha':.7},
             kde_kws={'linewidth':3})
sns.distplot(df.loc[df['class']=='suv',"cty"],color="orange",
             label="SUV",hist_kws={'alpha':.7},
             kde_kws={'linewidth':3})
sns.distplot(df.loc[df['class']=='minivan',"cty"],color="g",
             label="Minivan",hist_kws={'alpha':.7},
             kde_kws={'linewidth':3})
plt.ylim(0,0.35)						# 设置y轴的范围

# 图形修饰
plt.title('City Mileage by Vehicle Type',
          fontsize=16)					# 设置标题
plt.xlabel('City Mileage')				# 设置x轴标签
plt.ylabel('Density')					# 设置y轴标签
plt.legend()							# 显示图例
plt.show()
























#【例8-10】利用matplotlib库绘制密度图。输入代码如下：
from matplotlib import pyplot as plt, cm, colors
import numpy as np

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
side = np.linspace(-2, 2, 15)
X, Y = np.meshgrid(side, side)
Z = np.exp(-((X - 1) ** 2 + Y ** 2))
plt.pcolormesh(X, Y, Z, shading='auto')
plt.show()






















# 8.3 箱线图
#     箱线图（Box Plot），也称为盒须图或盒式图，是一种常用的统计图表，用于显示数值变
# 量的分布情况（包括中位数、四分位数、离散程度等）和异常值的存在，如图8-11所示。
#     箱线图的主要组成部分包括：
#     （1）箱体：箱体由两条水平线和一条垂直线组成。箱体的底边界表示数据的下四分位
# 数（Q1），顶边界表示数据的上四分位数（Q3），箱体的中线表示数据的中位数（或称为第二
# 四分位数）。
#     （2）须线：从箱体延伸出两条线段，分别表示数据的最小值和最大值，也可以是在异
# 常值之外的数据范围。
#     （3）异常值点：在箱体外部的点表示数据中的异常值，即与其他观测值相比显著偏离
# 的值。

#【例8-11】使用Matplotlib生成多个箱线图，并通过设置不同的参数来改变箱线图的外观和显示方式。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19781101)			# 固定随机种子，以便结果可复现
# 生成随机数据
spread=np.random.rand(50)*100
center=np.ones(25)*50
flier_high=np.random.rand(10)*100+100
flier_low=np.random.rand(10)*-100
data=np.concatenate((spread,center,flier_high,flier_low))

fig,axs=plt.subplots(2,3)

# 基本的箱线图
axs[0,0].boxplot(data)
axs[0,0].set_title('basic plot')

# 带缺口的箱线图
axs[0,1].boxplot(data,notch=True)
axs[0,1].set_title('notched plot')

# 更改异常值点的符号
axs[0,2].boxplot(data,sym='gD')
axs[0,2].set_title('change outlier\npoint symbols')

# 不显示异常值点
axs[1,0].boxplot(data,showfliers=False)
axs[1,0].set_title("don't show\noutlier points")

# 水平箱线图
axs[1,1].boxplot(data,vert=False,sym='rs')
axs[1,1].set_title('horizontal boxes')

# 更改箱须长度
axs[1,2].boxplot(data,vert=False,sym='rs',whis=0.75)
axs[1,2].set_title('change whisker length')

fig.subplots_adjust(left=0.08,right=0.98,bottom=0.05,top=0.9,
                    hspace=0.4,wspace=0.3)

# 生成更多的虚拟数据
spread=np.random.rand(50)*100
center=np.ones(25)*40
flier_high=np.random.rand(10)*100+100
flier_low=np.random.rand(10)*-100
d2=np.concatenate((spread,center,flier_high,flier_low))
# 如果所有列的长度相同，则可以将数据组织为2-D数组。如果不同，则使用列表。
# 使用列表更有效率，因为boxplot内部会将2-D数组转换为矢量列表。
data=[data,d2,d2[::2]]

# 在同一个坐标轴上绘制多个箱线图
fig,ax=plt.subplots()
ax.boxplot(data)
plt.show()





print('11')





















#【例8-12】绘制两个箱线图，分别展示不同样本数据的分布情况。其中，一个是矩形箱线图，另一个是有缺口的箱线图。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np

# 随机测试数据
np.random.seed(19781101)			# 固定随机种子，以便结果可复现
all_data=[np.random.normal(0,std,size=100)for std in range(1,4)]
labels=['x1','x2','x3']

# 创建图形和轴对象
fig,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,figsize=(9,4))

# 绘制矩形箱线图
bplot1=ax1.boxplot(all_data,
					vert=True,   			# 垂直箱体对齐
					patch_artist=True,		# 用颜色填充
					labels=labels)			# 将用于标记 x 轴刻度
ax1.set_title('Rectangular box plot')

# 绘制有缺口的箱线图
bplot2=ax2.boxplot(all_data,
					notch=True,  			# 缺口形状
					vert=True,   			# 垂直箱体对齐
					patch_artist=True,		# 用颜色填充
					labels=labels)			# 将用于标记 x 轴刻度
ax2.set_title('Notched box plot')

# 使用颜色填充
colors=['pink','lightblue','lightgreen']
for bplot in (bplot1,bplot2):
    for patch,color in zip(bplot['boxes'],colors):
        patch.set_facecolor(color)

# 添加水平网格线
for ax in [ax1,ax2]:
    ax.yaxis.grid(True)
    ax.set_xlabel('Three separate samples')
    ax.set_ylabel('Observed values')
plt.show()
























#【例8-13】生成并展示不同类型的箱线图和小提琴图。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# 构造数据集
a=pd.DataFrame({'group':np.repeat('A',500),
                  'value':np.random.normal(10,5,500)})
b=pd.DataFrame({'group':np.repeat('B',500),
                  'value':np.random.normal(13,1.2,500)})
c=pd.DataFrame({'group':np.repeat('B',500),
                  'value':np.random.normal(18,1.2,500)})
d=pd.DataFrame({'group':np.repeat('C',20),
                  'value':np.random.normal(25,4,20)})
e=pd.DataFrame({'group':np.repeat('D',100),
                  'value':np.random.uniform(12,size=100)})
df=pd.concat([a,b,c,d,e])	# 合并数据框

sns.boxplot(x='group',y='value',data=df)	# 常规箱线图
plt.show()

# 添加jitter的箱线图
ax=sns.boxplot(x='group',y='value',data=df)
# 添加stripplot
ax=sns.stripplot(x='group',y='value',data=df,color="orange",
                   jitter=0.2,size=2.5)
plt.title("Boxplot with jitter",loc="left")	# 添加标题
plt.show()

# 绘制小提琴图
sns.violinplot(x='group',y='value',data=df)
plt.title("Violin plot",loc="left")	# 添加标题
plt.show()

# 绘制基本的箱线图
sns.boxplot(x="group",y="value",data=df)

# 计算每组的观测数量和中位数以定位标签
medians=df.groupby(['group'])['value'].median().values
nobs=df.groupby("group").size().values
nobs=[str(x)for x in nobs.tolist()]
nobs=["n:"+i for i in nobs]

# 添加到图形中
pos=range(len(nobs))
for tick,label in zip(pos,ax.get_xticklabels()):
    plt.text(pos[tick],medians[tick]+0.4,nobs[tick],
             horizontalalignment='center',size='medium',
             color='w',weight='semibold')

plt.title("Boxplot with number of observation",loc="left")	# 添加标题
plt.show()






















#【例8-14】利用mpg_ggplot2数据集，通过绘制箱线图展示不同车辆类型的高速公路里程分布情况。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mpg_ggplot2.csv")		# 导入数据

# 绘制图表
plt.figure(figsize=(13,10),dpi=80)
sns.boxplot(x='class',y='hwy',data=df,notch=False)

# 在箱线图中添加观测数量（可选）
def add_n_obs(df,group_col,y):
    # 计算每个类别的中位数
    medians_dict={grp[0]:grp[1][y].median()for grp in
					 df.groupby(group_col)}
    # 获取x轴标签
    xticklabels=[x.get_text()for x in plt.gca().get_xticklabels()]
    # 计算每个类别的观测数量
    n_obs=df.groupby(group_col)[y].size().values
    # 遍历每个类别，在箱线图上方添加观测数量信息
    for (x,xticklabel),n_ob in zip(enumerate(xticklabels),n_obs):
        plt.text(x,medians_dict[xticklabel]*1.01,"#obs :"+str(n_ob),
                 horizontalalignment='center',fontdict={'size':10},
                 color='white')

# 调用函数添加观测数量信息
add_n_obs(df,group_col='class',y='hwy')

# 图表修饰
plt.title('Highway Mileage by Vehicle Class',fontsize=22)
plt.ylim(10,40)
plt.show()
























#【例8-15】绘制一个箱线图和散点图的组合图，展示不同车辆类型的高速公路里程分布情况，并根据汽缸数进行了着色区分。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mpg_ggplot2.csv")	# 导入数据
plt.figure(figsize=(10,6),dpi=80)	# 绘制图表

# 绘制箱线图，并根据汽缸数进行着色
sns.boxplot(x='class',y='hwy',data=df,hue='cyl')
# 绘制散点图
sns.stripplot(x='class',y='hwy',data=df,color='black',size=3,jitter=0.6)

# 在图上添加垂直线
for i in range(len(df['class'].unique())-1):
    plt.vlines(i+.5,10,45,linestyles='solid',colors='gray',alpha=0.2)

# 图表修饰
plt.title('Highway Mileage by Vehicle Class',fontsize=16)
plt.legend(title='Cylinders')
plt.show()





















# 8.4小提琴图
#     小提琴图（Violin Plot）用于展示数值变量的分布情况。它结合了箱线图和核密度图的特
# 点，可以同时显示数据的中位数、四分位数、离群值以及数据的密度分布。小提琴图的主要
# 组成部分包括：
#     （1）小提琴身体：由两个镜像的核密度估计曲线组成，展示了数据的密度分布情况。
# 较宽的部分表示密度高，较窄的部分表示密度低。
#     （2）白点/线条：表示数据的中位数和四分位数。
#     （3）边缘：垂直的线条称为边缘，显示了数据的范围。离群值可以通过边缘以外的点
# 来表示。
#     小提琴图常用于比较不同组别之间数值变量的分布情况，可以帮助观察数据的集中趋势、
# 离散程度以及异常值的存在情况。


#【例8-16】通过绘制小提琴图，展示不同车辆类型的高速公路里程分布情况。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mpg_ggplot2.csv")	# 导入数据
# 绘制小提琴图
plt.figure(figsize=(10,6),dpi=80)
sns.violinplot(x='class',y='hwy',data=df,scale='width',inner='quartile')

plt.title('Highway Mileage by Vehicle Class',fontsize=16)	# 图表修饰
plt.show()



























#【例8-17】使用 Seaborn 库绘制小提琴图，展示脑网络之间的相关性。输入代码如下：
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

# 加载示例数据集，这是一个关于脑网络相关性的数据集
df=sns.load_dataset("brain_networks",header=[0,1,2],index_col=0)

# 提取特定子集的网络
used_networks=[1,3,4,5,6,7,8,11,12,13,16,17]
used_columns=(df.columns.get_level_values("network")
						   .astype(int)
						   .isin(used_networks))
df=df.loc[:,used_columns]

# 计算相关矩阵并对网络进行平均
corr_df=df.corr().groupby(level="network").mean()
corr_df.index=corr_df.index.astype(int)
corr_df=corr_df.sort_index().T

# 设置 matplotlib 图形
f,ax=plt.subplots(figsize=(11,6))
# 绘制小提琴图，带有比默认值更窄的带宽
sns.violinplot(data=corr_df,bw_adjust=.5,cut=1,
                   linewidth=1,palette="Set3")
# 完成图形
ax.set(ylim=(-.7,1.05))
sns.despine(left=True,bottom=True)
plt.show()

























#【例8-18】绘制小提琴图，其中一个采用默认样式，另一个采用自定义样式。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np

# 定义函数：计算邻近值
def adjacent_values(vals,q1,q3):
    upper_adjacent_value=q3+(q3-q1)*1.5
    upper_adjacent_value=np.clip(upper_adjacent_value,q3,vals[-1])

    lower_adjacent_value=q1-(q3-q1)*1.5
    lower_adjacent_value=np.clip(lower_adjacent_value,vals[0],q1)
    return lower_adjacent_value,upper_adjacent_value

# 定义函数：设置坐标轴样式
def set_axis_style(ax,labels):
    ax.set_xticks(np.arange(1,len(labels)+1),labels=labels)
    ax.set_xlim(0.25,len(labels)+0.75)
    ax.set_xlabel('Sample name')

# 创建测试数据
np.random.seed(19781101)			# 固定随机种子，以便结果可复现
data=[sorted(np.random.normal(0,std,100))for std in range(1,5)]

# 创建图形和子图
fig,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,figsize=(9,4),sharey=True)

# 绘制默认小提琴图
ax1.set_title('Default violin plot')
ax1.set_ylabel('Observed values')
ax1.violinplot(data)

# 绘制自定义样式的小提琴图
ax2.set_title('Customized violin plot')
parts=ax2.violinplot(
        data,showmeans=False,showmedians=False,
        showextrema=False)

# 设置小提琴图的填充颜色和边缘颜色
for pc in parts['bodies']:
    pc.set_facecolor('#D43F3A')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

# 计算四分位数和范围
quartile1,medians,quartile3=np.percentile(data,[25,50,75],axis=1)
whiskers=np.array([
    adjacent_values(sorted_array,q1,q3)
    for sorted_array,q1,q3 in zip(data,quartile1,quartile3)])
whiskers_min,whiskers_max=whiskers[:,0],whiskers[:,1]

inds=np.arange(1,len(medians)+1)
ax2.scatter(inds,medians,marker='o',color='white',s=30,zorder=3)
ax2.vlines(inds,quartile1,quartile3,color='k',linestyle='-',lw=5)
ax2.vlines(inds,whiskers_min,whiskers_max,color='k',linestyle='-',lw=1)

# 设置坐标轴样式
labels=['A','B','C','D']
for ax in [ax1,ax2]:
    set_axis_style(ax,labels)

plt.subplots_adjust(bottom=0.15,wspace=0.05)
plt.show()

























#【例8-19】利用matplotlib绘制不同样式的小提琴图。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19781101)			# 固定随机种子，以便结果可复现
# 生成随机数据
fs=10  				# 字体大小
pos=[1,2,4,5,7,8]	# 每个小提琴图的位置
data=[np.random.normal(0,std,size=100)for std in pos]	# 正态分布随机数据

# 创建图形和子图
fig,axs=plt.subplots(nrows=2,ncols=5,figsize=(10,6))

# 绘制自定义小提琴图
axs[0,0].violinplot(data,pos,points=20,widths=0.3,# 绘制第1个小提琴图
					showmeans=True,showextrema=True,showmedians=True)
axs[0,0].set_title('Custom violinplot 1',fontsize=fs)	# 设置子图标题

axs[0,1].violinplot(data,pos,points=40,widths=0.5,# 绘制第2个小提琴图
					showmeans=True,showextrema=True,showmedians=True,
					bw_method='silverman')
axs[0,1].set_title('Custom violinplot 2',fontsize=fs)	# 设置子图标题

axs[0,2].violinplot(data,pos,points=60,widths=0.7,showmeans=True,
					showextrema=True,showmedians=True,bw_method=0.5)
axs[0,2].set_title('Custom violinplot 3',fontsize=fs)

axs[0,3].violinplot(data,pos,points=60,widths=0.7,showmeans=True,
					showextrema=True,showmedians=True,bw_method=0.5,
					quantiles=[[0.1],[],[],[0.175,0.954],[0.75],[0.25]])
axs[0,3].set_title('Custom violinplot 4',fontsize=fs)

axs[0,4].violinplot(data[-1:],pos[-1:],points=60,widths=0.7,
					showmeans=True,showextrema=True,showmedians=True,
					quantiles=[0.05,0.1,0.8,0.9],bw_method=0.5)
axs[0,4].set_title('Custom violinplot 5',fontsize=fs)

axs[1,0].violinplot(data,pos,points=80,vert=False,widths=0.7,
					showmeans=True,showextrema=True,showmedians=True)
axs[1,0].set_title('Custom violinplot 6',fontsize=fs)

axs[1,1].violinplot(data,pos,points=100,vert=False,widths=0.9,
					showmeans=True,showextrema=True,showmedians=True,
					bw_method='silverman')
axs[1,1].set_title('Custom violinplot 7',fontsize=fs)

axs[1,2].violinplot(data,pos,points=200,vert=False,widths=1.1,
					showmeans=True,showextrema=True,showmedians=True,
					bw_method=0.5)
axs[1,2].set_title('Custom violinplot 8',fontsize=fs)

axs[1,3].violinplot(data,pos,points=200,vert=False,widths=1.1,
					showmeans=True,showextrema=True,showmedians=True,
					quantiles=[[0.1],[],[],[0.175,0.954],[0.75],[0.25]],
					bw_method=0.5)
axs[1,3].set_title('Custom violinplot 9',fontsize=fs)

axs[1,4].violinplot(data[-1:],pos[-1:],points=200,vert=False,widths=1.1,
					showmeans=True,showextrema=True,showmedians=True,
					quantiles=[0.05,0.1,0.8,0.9],bw_method=0.5)
axs[1,4].set_title('Custom violinplot 10',fontsize=fs)

# 隐藏每个子图的y轴刻度标签
for ax in axs.flat:
    ax.set_yticklabels([])

fig.suptitle("Violin Plotting Examples")	# 设置图形标题
fig.subplots_adjust(hspace=0.4)	# 调整子图之间的垂直间距
plt.show()



























#【例8-20】利用绘制箱线图与小提琴图。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

# Fixing random state for reproducibility
np.random.seed(19680801)


# generate some random test data
all_data = [np.random.normal(0, std, 100) for std in range(6, 10)]

# plot violin plot
axs[0].violinplot(all_data,
                  showmeans=False,
                  showmedians=True)
axs[0].set_title('Violin plot')

# plot box plot
axs[1].boxplot(all_data)
axs[1].set_title('Box plot')

# adding horizontal grid lines
for ax in axs:
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(all_data))],
                  labels=['x1', 'x2', 'x3', 'x4'])
    ax.set_xlabel('Four separate samples')
    ax.set_ylabel('Observed values')

plt.show()





















# 8.5 金字塔图
#     金字塔图通常用于展示层级关系或者数据分层的结构。金字塔图的形状类似于金字塔，
# 底部较宽，顶部较窄，由一系列水平横条组成。金字塔图可以用于人口统计数据、市场份额
# 分析、销售层级、组织结构等。

#【例8-21】通过绘制金字塔图展示不同性别用户在购买阶段的数量分布情况。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df=pd.read_csv(r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\email_campaign_funnel.csv")

# 绘制图表
plt.figure(figsize=(10,8),dpi=80)
group_col='Gender'
order_of_bars=df.Stage.unique()[::-1]
colors=[plt.cm.Spectral(i/float(len(df[group_col].unique())-1))for i in
						        range(len(df[group_col].unique()))]

for c,group in zip(colors,df[group_col].unique()):
    # 绘制金字塔图
    sns.barplot(x='Users',y='Stage',data=df.loc[df[group_col]==group,:],
                  order=order_of_bars,color=c,label=group)

# 图表修饰    
plt.xlabel("$Users$")
plt.ylabel("Stage of Purchase")
plt.yticks(fontsize=12)
plt.title("Marketing Funnel",fontsize=18)
plt.legend()
plt.show()
























# 8.6 脊线图
#     脊线图（Ridge plot）是一种用于可视化多个概率密度函数或频率分布的图表类型。它通
# 过在横向轴上放置多个密度曲线或直方图，并将它们沿纵向轴对齐，形成一系列相互堆叠的
# 曲线或柱状图来展示数据分布的变化情况。脊线图常用于比较不同组、类别或条件下的数据
# 分布，并可用于发现和显示分布之间的差异和相似性。
#     在脊线图中，每个密度曲线或柱状图代表一个组、类别或条件的数据分布。它们沿着纵
# 向轴对齐，并且在水平轴上根据相对密度或频率进行堆叠。通过堆叠的方式，脊线图可以显
# 示整体分布形态以及各组之间的差异。
#     脊线图通常使用透明度来避免不同曲线或柱状图之间的重叠，从而提高可视化的可读性。
# 此外，还可以通过添加标签、颜色编码等方式来进一步增强脊线图的信息展示。

#【例8-23】使用joypy库绘制城市和高速公路里程按照车辆类型进行可视化。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt
import joypy 					# 确保已经安装joypy库

mpg=pd.read_csv(r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mpg_ggplot2.csv")	# 导入数据
plt.figure(figsize=(10,6),dpi=80)	# 创建图形

# 使用joypy.joyplot函数绘制Joy Plot
# column参数指定要绘制的列，by参数指定分类的基准列
# ylim='own'表示y轴范围会根据数据自动调整
fig,axes=joypy.joyplot(mpg,column=['hwy','cty'],by="class",
						        ylim='own',figsize=(10,6))

# 图形修饰
plt.title('City and Highway Mileage by Class',fontsize=18)
					   		# 设置标题
plt.xlabel('Mileage')		# 设置x轴标签
plt.ylabel('Class')		# 设置y轴标签
plt.xticks(fontsize=12)	# 设置x轴标签的字体大小
plt.yticks(fontsize=12)	# 设置y轴标签的字体大小
plt.grid(True,which='both',linestyle='--',linewidth=0.5)	# 添加网格线
plt.tight_layout()		# 调整子图的布局以适应画布
plt.show()























#【例8-24】基于Weather-data数据集，通过Seaborn和Matplotlib可视化某年每月的平均温度分布情况。输入代码如下：
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置Seaborn的样式，包括白色背景和隐藏坐标轴背景
sns.set_theme(style="white",rc={"axes.facecolor":(0,0,0,0)})

# 获取数据
temp=pd.read_csv(r'S:\Desktop_new\caxbook_python\python_plot_202405\PyData\Weather-data.csv')
# 将日期列转换为月份，并存储在一个单独的 'month' 列中
temp['month']=pd.to_datetime(temp['Date']).dt.month 

# 定义一个字典，将月份数字映射到对应的月份名称
month_dict={1:'january',2:'february',3:'march',4:'april',
              5:'may',6:'june',7:'july',8:'august',
              9:'september',10:'october',11:'november',12:'december'}

# 使用字典映射月份数字到月份名称，创建一个新的 'month' 列
temp['month']=temp['month'].map(month_dict)

# 生成一个包含每月平均温度的Series（用于图中的颜色），并创建一个新列
month_mean_serie=temp.groupby('month')['Mean_TemperatureC'].mean()
temp['mean_month']=temp['month'].map(month_mean_serie)

# 生成一个调色板，用于在图中表示不同月份
pal=sns.color_palette(palette='viridis',n_colors=12)

# 创建一个FacetGrid对象，将数据按月份分组，并以月份为行，使用颜色表示平均温度
g=sns.FacetGrid(temp,row='month',hue='mean_month',aspect=15,
                  height=0.75,palette=pal)

# 添加每个月的密度估计KDE图
g.map(sns.kdeplot,'Mean_TemperatureC',bw_adjust=1,clip_on=False,
      fill=True,alpha=1,linewidth=1.5)

# 添加表示每个KDE图轮廓的白色线
g.map(sns.kdeplot,'Mean_TemperatureC',
      bw_adjust=1,clip_on=False,color="w",lw=2)

g.map(plt.axhline,y=0,lw=2,clip_on=False)		# 添加每个子图的水平线

# 在每个子图中添加文本，表示对应的月份，文本颜色与KDE图的颜色相匹配
for i,ax in enumerate(g.axes.flat):
    ax.text(-15,0.02,month_dict[i+1],
            fontweight='bold',fontsize=15,
            color=ax.lines[-1].get_color())
g.fig.subplots_adjust(hspace=-0.3)		# 使用matplotlib调整子图之间的间距

# 移除子图的标题、y轴刻度和脊柱
g.set_titles("")
g.set(yticks=[],ylabel="")					# 不显示y轴刻度和标签
g.despine(bottom=True,left=True)

# 设置x轴标签的字体大小和粗细
plt.setp(ax.get_xticklabels(),fontsize=15,fontweight='bold')
plt.xlabel('Temperature in degree Celsius',fontweight='bold',fontsize=15)

# 设置图的标题
g.fig.suptitle('Daily average temperature in Seattle per month',
               ha='right',fontsize=20,fontweight=20)
plt.show()























#【例8-25】利用Bokeh库创建一组脊线图，展示一系列类别数据的分布情况。输入代码如下：
import colorcet as cc
from numpy import linspace
from scipy.stats import gaussian_kde

from bokeh.models import ColumnDataSource,FixedTicker,PrintfTickFormatter
from bokeh.plotting import figure,show
from bokeh.sampledata.perceptions import probly

# 定义函数用于生成脊线图数据
def ridge(category,data,scale=20):
    return list(zip([category]*len(data),scale*data))

# 反转类别顺序并获取颜色
cats=list(reversed(probly.keys()))
palette=[cc.rainbow[i*15]for i in range(17)]

x=linspace(-20,110,500)	# 生成 x 坐标轴数据
source=ColumnDataSource(data=dict(x=x))	# 创建数据源

# 创建 Bokeh 图表对象
p=figure(y_range=cats,width=900,x_range=(-5,105),toolbar_location=None)

# 遍历每个类别，绘制脊线图
for i,cat in enumerate(reversed(cats)):
    # 使用高斯核密度估计函数拟合数据
    pdf=gaussian_kde(probly[cat])
    # 计算并添加脊线图数据到数据源
    y=ridge(cat,pdf(x))
    source.add(y,cat)
    # 绘制脊线图
    p.patch('x',cat,color=palette[i],alpha=0.6,line_color="black",
                   source=source)

# 设置图表样式
p.outline_line_color=None
p.background_fill_color="#efefef"

# 设置 x 轴刻度和格式
p.xaxis.ticker=FixedTicker(ticks=list(range(0,101,10)))
p.xaxis.formatter=PrintfTickFormatter(format="%d%%")

# 隐藏网格线和轴线
p.ygrid.grid_line_color=None
p.xgrid.grid_line_color="#dddddd"
p.xgrid.ticker=p.xaxis.ticker
p.axis.minor_tick_line_color=None
p.axis.major_tick_line_color=None
p.axis.axis_line_color=None

p.y_range.range_padding=0.12# 设置 y 轴范围填充
show(p)




















# 8.7 累积分布曲线图
#     累积分布函数（Cumulative Distribution Function，CDF）是描述随机变量在某一取值之
# 前累积概率的函数。对于连续型随机变量，累积分布函数在一个特定取值处的值等于该取值
# 以下所有可能取值的概率之和；对于离散型随机变量，CDF在某个取值处的值等于该取值及
# 其以下所有可能取值的概率之和。
#     在统计学和概率论中，累积分布函数是对一个随机变量的全部可能取值的概率分布进行
# 描述的一种方法。通过累积分布函数，我们可以得知随机变量小于或等于某个特定值的概率。
#     使用经验累积分布函数（Empirical CDF，ECDF）可以估计累积分布函数。ECDF 是对
# 数据集的累积分布进行估计的一种非参数方法，它将每个数据点视为累积概率，并将这些点
# 连接起来形成一个阶梯状的曲线。ECDF提供了一种直观的方式来了解数据的累积分布情况，
# 尤其适用于小样本数据。
#     通过绘制累积分布曲线图，我们可以直观地观察到随机变量在不同取值下的累积概率分
# 布情况，从而更好地理解数据的分布特征和变化趋势。

#【例8-26】绘制累积分布和互补累积分布曲线图。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19781101)			# 固定随机种子，以便结果可复现
mu=200
sigma=25
n_bins=25
data=np.random.normal(mu,sigma,size=100)

# 创建图形和子图
fig=plt.figure(figsize=(9,4),constrained_layout=True)
axs=fig.subplots(1,2,sharex=True,sharey=True)

# 累积分布
n,bins,patches=axs[0].hist(data,n_bins,density=True,histtype="step",
         cumulative=True,label="Cumulative histogram")	# 绘制累积分布曲线图
x=np.linspace(data.min(),data.max())
y=((1/(np.sqrt(2*np.pi)*sigma))*
     np.exp(-0.5*(1/sigma*(x-mu))**2))
y=y.cumsum()
y /=y[-1]
axs[0].plot(x,y,"k--",linewidth=1.5,label="Theory")	# 绘制理论曲线

# 互补累积分布
axs[1].hist(data,bins=bins,density=True,histtype="step",cumulative=-1,
            label="Reversed cumulative histogram")		# 绘制反向累积直方图
axs[1].plot(x,1-y,"k--",linewidth=1.5,label="Theory")	# 绘制理论曲线

# 图形标签
fig.suptitle("Cumulative distributions")
for ax in axs:
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("Annual rainfall (mm)")
    ax.set_ylabel("Probability of occurrence")
    ax.label_outer()
plt.show()