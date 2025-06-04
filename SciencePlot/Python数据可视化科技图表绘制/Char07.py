#七、局部整体型数据可视化
#     当涉及局部整体型数据可视化时，Python提供了强大的工具包，使得分析人员能够以直
# 观和有力的方式呈现数据的结构和模式。在本章的学习过程中，将使用公开的数据集和示例
# 代码来演示可视化方法，包括饼图、华夫图、马赛克图等。希望读者在面对局部整体型数据
# 时，能够灵活、准确地选择合适的可视化方法，并从中获得有价值的见解和发现。


# 7.1饼图
#     饼图（Pie Chart）是一种用于展示各类别在整体中的比例关系。它以圆形为基础，将整
# 体分成多个扇形，每个扇形的角度大小表示该类别在总体中的比例或占比。
#     饼图的优点是直观地展示了各类别在整体中的相对比例，常用于表示不同类别的市场份
# 额、调查结果中的频数分布等。在使用饼图时，应选择合适的数据和合适的类别数量，以确
# 保图表的可读性和准确传达数据。在使用饼图时，建议明确标记饼图每个部分的百分比或数
# 字。
#【例7-1】饼图绘制示例1。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt

# 准备数据
df_raw=pd.read_csv("S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mpg_ggplot2.csv")		# 读取数据
# 根据车辆类型（class）对数据进行分组，并计算每个类型的数量
df=df_raw.groupby('class').size()
df.plot(kind='pie',subplots=True,figsize=(5,5))		# 使用pandas绘制饼图

# 设置标题和坐标轴标签
plt.title("Pie Chart of Vehicle Class-Bad")
plt.ylabel("")
plt.show()






























#【例7-2】饼图绘制示例2。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 从 CSV 文件中读取数据
df_raw=pd.read_csv("S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mpg_ggplot2.csv")

# 准备数据
# 根据车辆类型（class）对数据进行分组，并计算每个类型的数量
df=df_raw.groupby('class').size().reset_index(name='counts')

# 绘制图形
# 创建图形和轴对象，设置图形大小和等轴比例
fig,ax=plt.subplots(figsize=(8,6),subplot_kw=dict(aspect="equal"))

# 提取数据和类别
data=df['counts']
categories=df['class']
explode=[0,0,0,0,0,0.1,0]		# 设置饼图的爆炸程度

# 定义一个函数，用于在饼图上显示百分比和绝对值
def func(pct,allvals):
    absolute=int(pct/100.*np.sum(allvals))
    return "{:.1f}% ({:d} )".format(pct,absolute)

# 绘制饼图
wedges,texts,autotexts=ax.pie(data,
			           autopct=lambda pct:func(pct,data),
			           textprops=dict(color="w"),
			           colors=plt.cm.Dark2.colors,
			           startangle=140,
			           explode=explode)
# 图形修饰
ax.legend(wedges,categories,title="Vehicle Class",loc="center left",
          bbox_to_anchor=(1,0,0.5,1))				# 添加图例
plt.setp(autotexts,size=10,weight=700)			# 设置自动文本的大小和字重
ax.set_title("Class of Vehicles:Pie Chart")		# 设置图形标题
plt.show()































#【例7-3】饼图绘制示例3。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

# 创建图形和轴对象
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,5))
fig.subplots_adjust(wspace=0)

# 饼图参数
overall_ratios=[.27,.56,.17]
labels=['Approve','Disapprove','Undecided']
explode=[0.1,0,0]
angle=-180*overall_ratios[0]	# 旋转角度使得第1个楔形图被 x 轴分隔
wedges,*_=ax1.pie(overall_ratios,autopct='%1.1f%%',startangle=angle,
					labels=labels,explode=explode)

# 柱状图参数
age_ratios=[.33,.54,.07,.06]
age_labels=['Under 35','35-49','50-65','Over 65']
bottom=1
width=.2

# 从顶部开始添加，以匹配图例
for j,(height,label)in enumerate(reversed([*zip(age_ratios,age_labels)])):
    bottom -=height
    bc=ax2.bar(0,height,width,bottom=bottom,color='C0',label=label,
                 alpha=0.1+0.25*j)
    ax2.bar_label(bc,labels=[f"{height:.0%}"],label_type='center')

# 设置柱状图的标题、图例和坐标轴
ax2.set_title('Age of approvers')
ax2.legend()
ax2.axis('off')
ax2.set_xlim(- 2.5*width,2.5*width)

# 使用 ConnectionPatch 在两个子图之间绘制连接线
theta1,theta2=wedges[0].theta1,wedges[0].theta2
center,r=wedges[0].center,wedges[0].r
bar_height=sum(age_ratios)

# 绘制顶部连接线
x=r*np.cos(np.pi/180*theta2)+center[0]
y=r*np.sin(np.pi/180*theta2)+center[1]
con=ConnectionPatch(xyA=(-width/2,bar_height),coordsA=ax2.transData,
					 xyB=(x,y),coordsB=ax1.transData)
con.set_color([0,0,0])
con.set_linewidth(2)
ax2.add_artist(con)

# 绘制底部连接线
x=r*np.cos(np.pi/180*theta1)+center[0]
y=r*np.sin(np.pi/180*theta1)+center[1]
con=ConnectionPatch(xyA=(-width/2,0),coordsA=ax2.transData,
					 xyB=(x,y),coordsB=ax1.transData)
con.set_color([0,0,0])
ax2.add_artist(con)
con.set_linewidth(2)
plt.show()
































#【例7-4】绘制环形饼图，并对其进行标记。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np

# 创建一个大小为(6,4)的图，并设置子图属性为“等比例”
fig,ax=plt.subplots(figsize=(6,4),subplot_kw=dict(aspect="equal"))

# 配方和数据
recipe=["225 g flour","90 g sugar","1 egg",
          "60 g butter","100 ml milk","1/2 package of yeast"]
data=[225,90,50,60,100,5]
# 绘制饼图
wedges,texts=ax.pie(data,wedgeprops=dict(width=0.5),startangle=-40)

# 注释框的属性
bbox_props=dict(boxstyle="square,pad=0.3",fc="w",ec="k",lw=0.72)
kw=dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props,zorder=0,va="center")

# 遍历每个扇形并添加注释
for i,p in enumerate(wedges):
    # 计算注释的位置
    ang=(p.theta2-p.theta1)/2.+p.theta1
    y=np.sin(np.deg2rad(ang))
    x=np.cos(np.deg2rad(ang))
    # 水平对齐方式根据 x 坐标的正负确定
    horizontalalignment={-1:"right",1:"left"}[int(np.sign(x))]
    # 设置连接线的样式
    connectionstyle=f"angle,angleA=0,angleB={ang}"
    kw["arrowprops"].update({"connectionstyle":connectionstyle})
    # 添加注释
    ax.annotate(recipe[i],xy=(x,y),xytext=(1.35*np.sign(x),1.4*y),
                horizontalalignment=horizontalalignment,**kw)
ax.set_title("Matplotlib bakery:A donut")	# 设置标题
plt.show()































# 7.2 嵌套饼图
#     嵌套饼图（Nested Pie Charts）是一种用于可视化多层次数据的图表类型。它通过将多个
# 圆形饼图嵌套在一起，以一种分层的方式来显示数据。每个圆环表示数据的一个层次，而每
# 个扇形表示该层次下的数据占比。嵌套饼图可实现以下功能。
#     （1）显示多层次数据关系：嵌套饼图能够清晰地展示数据的多层次结构，使观察者能
# 够理解各层次之间的关系。
#     （2）比较不同层次的占比：通过不同大小的圆环和扇形，可以直观地比较不同层次的
# 数据占比。
#     （3）可视化层次结构：每个圆环代表一个层次，内部的扇形代表该层次下的子类别或
# 数据分组，从而可视化层次结构。
#     在Python中可以使用Matplotlib、Seaborn等等绘制出嵌套饼图。在绘制时，可以通过调
# 整参数（如标签、颜色、大小等）来定制图表的外观，使其更具可读性和吸引力。


#【例7-5】绘制嵌套饼图。输入代码如下：
import matplotlib.pyplot as plt
import numpy as np

fig,ax=plt.subplots()							# 创建图形和轴对象
# 设置环形饼图的参数
size=0.3
vals=np.array([[60.,32.],[37.,40.],[29.,10.]])	# 数据

# 使用 colormap 获取颜色
cmap=plt.colormaps["tab20c"]
outer_colors=cmap(np.arange(3)*4)
inner_colors=cmap([1,2,5,6,9,10])

# 绘制外层环
ax.pie(vals.sum(axis=1),radius=1,colors=outer_colors,
       wedgeprops=dict(width=size,edgecolor='w'))
# 绘制内层环
ax.pie(vals.flatten(),radius=1-size,colors=inner_colors,
       wedgeprops=dict(width=size,edgecolor='w'))

ax.set(aspect="equal",title='Pie plot with ax.pie')	# 设置图形属性
plt.show()


























# 7.3华夫图
#     华夫图（Waffle Chart）是一种用于展示部分与整体之间比例关系的可视化图表。它以方
# 格或正方形为基础，通过填充或着色的方式，将整体分成若干个小区块，每个小区块表示一
# 个部分，并以它在整体中所占比例来决定区块的大小。
#     华夫图的优点是简单直观，能够快速展示部分与整体之间的比例关系。它常用于表示市
# 场份额、人口组成、调查结果中的比例等。在使用华夫图时，应谨慎选择适合的数据和合适
# 的部分与整体比例，以确保图表的可读性和准确传达数据。
#     在Python中，使用pywaffle包可以创建华夫图，并用于显示更大群体中组的组成。


#【例7-6】华夫图绘制示例。输入代码如下：
import matplotlib.pyplot as plt	# 导入Matplotlib库
from pywaffle import Waffle	# 导入pywaffle库

# 创建一个Figure对象，绘制Waffle图
fig=plt.figure(FigureClass=Waffle,# 使用Waffle类作为FigureClass参数
                 rows=5,					# 设置行数为5
                 columns=10,				# 设置列数为10
                 values=[48,46,6],		# 设置值
                 figsize=(5,3))			# 设置图表尺寸
plt.show()

# 在图中添加标签
data={'Cat1':40,'Cat2':16,'Cat3':28}
fig=plt.figure( FigureClass=Waffle,rows=5,values=data,
    legend={'loc':'upper left','bbox_to_anchor':(1.05,1)},)
plt.show()

# 样式设置，包括图例、标题、颜色、方向、排列样式等
data={'Car':58,'Pickup':21,'Truck':11,'Motorcycle':7}
fig=plt.figure( FigureClass=Waffle,rows=5,values=data,
    colors=["#C1D82F","#00A4E4","#FBB034",'#6A737B'],
    title={'label':'Vehicle Sales by Vehicle Type','loc':'left'},
    labels=[f"{k} ({v}%)" for k,v in data.items()],
    legend={'loc':'lower left','bbox_to_anchor':(0,-0.4),
            'ncol':len(data),'framealpha':0},
    starting_location='NW',vertical=True,
    block_arranging_style='snake')
fig.set_facecolor('#EEEEEE')
plt.show()

# 利用图标绘图（象形图）
data={'Car':58,'Pickup':21,'Truck':11,'Motorcycle':7}
fig=plt.figure( FigureClass=Waffle,rows=5,values=data,
    colors=["#c1d82f","#00a4e4","#fbb034",'#6a737b'],
    legend={'loc':'upper left','bbox_to_anchor':(1,1)},
    icons=['car-side','truck-pickup','truck','motorcycle'],
    font_size=12,icon_legend=True)
plt.show()

# 在现有图形和轴上绘图
fig=plt.figure()
ax=fig.add_subplot(111)
# 修改现有轴
ax.set_title("Axis Title")
ax.set_aspect(aspect="equal")

# 在轴上绘制Waffle图
Waffle.make_waffle(ax=ax,# 将轴传递给make_waffle函数
    rows=5,columns=10,values=[40,26,8],
    title={"label":"Waffle Title","loc":"left"} )
plt.show()































#【例7-7】组合华夫图绘制示例。输入代码如下：
import matplotlib.pyplot as plt	# 导入Matplotlib库
from pywaffle import Waffle	# 导入pywaffle库
import pandas as pd

data=pd.DataFrame(
    {
        'labels':['Car','Truck','Motorcycle'],# 创建包含车辆类型的标签
        'Factory A':[32384,13354,5245],		# 工厂A的车辆生产量
        'Factory B':[22147,6678,2156],		# 工厂B的车辆生产量
        'Factory C':[8932,3879,82896],		# 工厂C的车辆生产量
    },
).set_index('labels')	# 将标签设置为DataFrame的索引

# A glance of the data:
#             Factory A  Factory B  Factory C
# labels
# Car             27384      22147       8932
# Truck            7354       6678       3879
# Motorcycle       3245       2156       1196

fig=plt.figure(	# 创建一个新的Figure对象
    FigureClass=Waffle,# 使用Waffle类创建图形
    plots={	# 在子图中绘制Waffle图
        311:{	# 子图1
            'values':data['Factory A']/1000,# 将实际数量转换为合理的块数
            'labels':[f"{k} ({v})" for k,v in
	                      data['Factory A'].items()],# 添加标签和值
            'legend':{'loc':'upper left','bbox_to_anchor':(1.05,1),
	                       'fontsize':8},# 图例设置
            'title':{'label':'Vehicle Production of Factory A','loc':'left',
	                      'fontsize':12}	# 子图标题设置
        },
        312:{	# 子图2
            'values':data['Factory B']/1000,# 将实际数量转换为合理的块数
            'labels':[f"{k} ({v})" for k,v in
	                       data['Factory B'].items()],# 添加标签和值
            'legend':{'loc':'upper left','bbox_to_anchor':(1.2,1),
	                      'fontsize':8},# 图例设置
            'title':{'label':'Vehicle Production of Factory B','loc':'left',
	                      'fontsize':12}	# 子图标题设置
        },
        313:{	# 子图3
            'values':data['Factory C']/1000,		# 将实际数量转换为合理的块数
            'labels':[f"{k} ({v})" for k,v in
	                       data['Factory C'].items()],		# 添加标签和值
            'legend':{'loc':'upper left','bbox_to_anchor':(1.3,1),
	                      'fontsize':8},		# 图例设置
            'title':{'label':'Vehicle Production of Factory C','loc':'left',
	                      'fontsize':12}		# 子图标题设置
        },
    },
    rows=5,					# 设置外部参数应用于所有子图
    cmap_name="Accent",		# 使用cmap更改颜色
    rounding_rule='ceil',	# 更改舍入规则，以便值小于1000的仍至少有1个块
    figsize=(6,5))			# 设置图形大小

fig.suptitle('Vehicle Production by Vehicle Type',fontsize=14,
             fontweight='bold')		# 设置图形的总标题
fig.supxlabel('1 block=1000 vehicles',fontsize=8,x=0.14)	# 设置x轴标签
fig.set_facecolor('#EEEDE7')		# 设置图形的背景颜色
plt.show()



































# 7.4 马赛克图
#     马赛克图（Mosaic Plot）是一种用于可视化多个分类变量之间关系的图表类型。它以矩
# 形区域为基础，将整体分割成多个小矩形，每个小矩形的面积大小表示对应分类变量组合的
# 频数或占比。
#     马赛克图的优点是能够同时显示多个分类变量之间的关系，并直观地展示各个组合的频
# 数或占比。它适用于探索多个分类变量之间的相关性和模式，并且可以帮助发现不同组合之
# 间的差异。在使用马赛克图时，应谨慎选择合适的数据和合适的分类变量组合，以确保图表
# 的可读性和准确传达数据。


#【例7-8】利用Altair和Vega数据集绘制了一个堆叠矩形图，展示了汽车数据集中不同产地和汽缸数的汽车数量。输入代码如下：
import altair as alt
from vega_datasets import data

source=data.cars()				# 载入汽车数据集
# 创建基础图表
base=( alt.Chart(source)
    # 聚合数据，计算每个组合的数量
    .transform_aggregate(count_="count()",groupby=["Origin","Cylinders"])
    # 堆叠数据，为堆叠创建必要的字段
    .transform_stack(
        stack="count_",									# 堆叠的字段
        as_=["stack_count_Origin1","stack_count_Origin2"],	# 创建的堆叠字段
        offset="normalize",								# 堆叠的方式
        sort=[alt.SortField("Origin","ascending")],		# 排序方式
        groupby=[],# 不进行分组
    )
    # 窗口转换，计算堆叠后的范围
    .transform_window(
        x="min(stack_count_Origin1)",				# x 轴的起始值
        x2="max(stack_count_Origin2)",				# x 轴的结束值
        rank_Cylinders="dense_rank()",				# Cylinders 排名
        distinct_Cylinders="distinct(Cylinders)",	# 不同Cylinders的数量
        groupby=["Origin"],							# 按Origin分组
        frame=[None,None],							# 窗口的范围
        sort=[alt.SortField("Cylinders","ascending")],	# 排序方式
    )
    # 窗口转换，计算Origin排名
    .transform_window(
        rank_Origin="dense_rank()",					# Origin 排名
        frame=[None,None],# 窗口的范围
        sort=[alt.SortField("Origin","ascending")],	# 排序方式
    )
    # 堆叠数据，为堆叠创建必要的字段
    .transform_stack(
        stack="count_",					# 堆叠的字段
        groupby=["Origin"],				# 按 Origin 分组
        as_=["y","y2"],					# 创建的堆叠字段
        offset="normalize",				# 堆叠的方式
        sort=[alt.SortField("Cylinders","ascending")],	# 排序方式
    )
    # 计算坐标轴的位置
    .transform_calculate(
        ny="datum.y+(datum.rank_Cylinders-1)*  \
            datum.distinct_Cylinders*0.01 / 3",			# y轴的起始位置
        ny2="datum.y2+(datum.rank_Cylinders-1)*\
            datum.distinct_Cylinders*0.01 / 3",			# y轴的结束位置
        nx="datum.x+(datum.rank_Origin-1)*0.01",		# x轴的起始位置
        nx2="datum.x2+(datum.rank_Origin-1)*0.01",		# x轴的结束位置
        xc="(datum.nx+datum.nx2)/2",						# x轴中心位置
        yc="(datum.ny+datum.ny2)/2",						# y轴中心位置
    )
)
# 绘制矩形图层
rect=base.mark_rect().encode(
    x=alt.X("nx:Q",axis=None),							# x轴坐标
    x2="nx2",											# x轴的结束坐标
    y="ny:Q",											# y轴坐标
    y2="ny2",											# y轴的结束坐标
    color=alt.Color("Origin:N",legend=None),			# 颜色编码
    opacity=alt.Opacity("Cylinders:Q",legend=None),	# 透明度编码
    tooltip=["Origin:N","Cylinders:Q"],					# 提示信息
)

# 绘制文本图层
text=base.mark_text(baseline="middle").encode(
    x=alt.X("xc:Q",axis=None),						# x轴坐标
    y=alt.Y("yc:Q",title="Cylinders"),				# y轴坐标
    text="Cylinders:N",								# 显示的文本
)
mosaic=rect+text# 组合图层
# 添加 Origin 标签
origin_labels=base.mark_text(baseline="middle",align="center").encode(
    x=alt.X(
        "min(xc):Q",
        axis=alt.Axis(title="Origin",orient="top"),		# 设置Origin 标题位置
    ),
    color=alt.Color("Origin",legend=None),				# 颜色编码
    text="Origin",# 显示的文本
)
# 配置图表外观和布局
(
    (origin_labels & mosaic)					# 图层组合
    .resolve_scale(x="shared")				# x 轴范围共享
    .configure_view(stroke="")				# 设置视图样式
    .configure_concat(spacing=10)				# 设置图层间距
    .configure_axis(domain=False,ticks=False,
                    labels=False,grid=False)		# 配置坐标轴
)
