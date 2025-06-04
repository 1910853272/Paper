# 六、层次关系数据可视化
#     层次关系数据是一种常见的数据类型，涉及到数据之间的多层次结构和关联。在许多领
# 域中，如生物学、社交网络分析、组织结构等，层次关系数据都扮演着重要的角色。在学习
# 过程中，将使用公开的数据集和示例代码来演示不同的可视化方法。希望读者在面对层次关
# 系数据时，能够灵活、准确地选择合适的可视化方法，并从中获得有价值的见解和发现。


# 6.1 旭日图
#     旭日图（Sunburst chart）是一种环形图，用于显示层次结构数据的分布和比例关系。它
# 以太阳系的形象为灵感，将数据分层显示为环形的扇形区域，每个扇形表示一个类别或子类
# 别，并显示其在整体中的比例。旭日图的主要特点包括：
#     （1）环形结构：旭日图呈现为一个环形，内部是根节点或整体，外部是子节点或类别。
# 每个层级都由一条环形扇区表示。
#     （2）扇形区域：每个扇形区域的大小表示该类别在整体中的比例。较大的扇形表示具
# 有更高比例的类别，而较小的扇形表示具有较低比例的类别。
#     （3）颜色编码：通常，每个扇形区域使用不同的颜色来区分类别或子类别。颜色可以
# 帮助观察者快速识别不同的数据部分。
#     （4）层级结构：旭日图可以显示多个层级，每个层级可以细分为更小的子类别。这种
# 层级结构使观察者能够了解不同类别之间的分布和比例关系。
#     下面介绍交互式可视化包plotly绘制旭日图。该包根据需要可以显示所有级，也可以仅
# 显示父级及其子级，方便用户探索各级的比例以及父级的子级占该父级的比例。

#【例6-1】利用Plotly库绘制旭日图示例1，采用不同的参数设置。输入代码如下：
import plotly.express as px		# 导入Plotly库中的Express模块，用于快速绘制图表

df=px.data.tips()	# 使用Plotly Express提供的示例数据tips()
# 创建旭日图，路径为'day','time','sex'，数值列为'total_bill'
fig1=px.sunburst(df,path=['day','time','sex'],values='total_bill')
fig1.show()

# 创建旭日图，并设置路径、数值列，根据'day'进行颜色着色
fig2=px.sunburst(df,path=['sex','day','time'],
						  values='total_bill',color='day')
fig2.show()

# 创建旭日图，并设置路径、数值列，根据'time'进行颜色着色
fig3=px.sunburst(df,path=['sex','day','time'],
					 values='total_bill',color='time')
fig3.show()

# 创建旭日图，并设置路径、数值列，根据'time'进行颜色着色，
# 并使用离散颜色映射为不同的时间段设置不同的颜色
fig4=px.sunburst(df,path=['sex','day','time'],
					values='total_bill',color='time',
                    color_discrete_map={'(?)':'black','Lunch':'gold',
										  'Dinner':'darkblue'})
fig4.show()





























#【例6-2】利用plotly绘制旭日图示例2。输入代码如下：
import plotly.express as px		# 导入Plotly库中的Express模块，用于快速绘制图表

# 创建一个包含角色、父母和数值的字典数据
data=dict(
    character=["Eve","Cain","Seth","Enos","Noam",
               "Abel","Awan","Enoch","Azura"],
    parent=["","Eve","Eve","Seth","Seth","Eve",
            "Eve","Awan","Eve"],
    value=[10,14,12,10,2,6,6,4,4])

# 使用Plotly Express的sunburst函数创建旭日图，传入数据和对应的列名
fig=px.sunburst(data,names='character',parents='parent',values='value')
fig.show()



























#【例6-3】利用plotly绘制旭日图示例3。输入代码如下：
import plotly.express as px
import numpy as np

# 使用Plotly Express提供的示例数据gapminder()，并筛选出年份为2007的数据
df=px.data.gapminder().query("year==2007")

# 使用Plotly Express的sunburst函数创建旭日图，指定路径以及数值列'pop'
# 并设置颜色映射为'lifeExp'，悬停数据为'iso_alpha'列
# 设置颜色映射的连续色板为'Red-Blue'，以及颜色映射的中点为'lifeExp'列值的加权平均值
fig=px.sunburst(df,path=['continent','country'],values='pop',
              color='lifeExp',hover_data=['iso_alpha'],
              color_continuous_scale='RdBu',
              color_continuous_midpoint=np.average(df['lifeExp'],
               weights=df['pop']))
fig.show()

































#【例6-4】创建一个包含供应商、行业、地区和销售额的数据框，并使用Plotly Express绘制旭日图。输入代码如下：
import plotly.express as px		# 导入Plotly库中的Express模块，用于快速绘制图表
import pandas as pd  			# 导入Pandas库，用于数据处理

# 创建包含供应商、行业、地区和销售额的数据
vendors=["A","B","C","D",None,"E","F","G","H",None]
sectors=["Tech","Tech","Finance","Finance","Other",
           "Tech","Tech","Finance","Finance","Other"]
regions=["North","North","North","North","North",
           "South","South","South","South","South"]
sales=[1,3,2,4,1,2,2,1,4,1]
df=pd.DataFrame(
    dict(vendors=vendors,sectors=sectors,regions=regions,sales=sales))
# print(df)

# 使用Plotly Express的sunburst函数创建旭日图，设置路径与数值列
fig=px.sunburst(df,path=['regions','sectors','vendors'],values='sales')
fig.show()





























#【例6-5】利用Plotly的Graph Objects模块创建旭日图。输入代码如下：
import plotly.graph_objects as go

# 创建旭日图
fig=go.Figure(go.Sunburst(
 ids=[
    "North America","Europe","Australia","North America-Football",
    "Soccer","North America-Rugby","Europe-Football","Rugby",
    "Europe-American Football","Australia-Football","Association",
    "Australian Rules","Autstralia-American Football","Australia-Rugby",
    "Rugby League","Rugby Union"],
  labels=[
    "North<br>America","Europe","Australia","Football","Soccer","Rugby",
    "Football","Rugby","American<br>Football","Football","Association",
   "Australian<br>Rules","American<br>Football","Rugby","Rugby<br>League",
    "Rugby<br>Union"],
  parents=[
    "","","","North America","North America","North America","Europe",
    "Europe","Europe","Australia","Australia-Football",
    "Australia-Football","Australia-Football","Australia-Football",
    "Australia-Rugby","Australia-Rugby"],
))
fig.update_layout(margin=dict(t=0,l=0,r=0,b=0))		# 更新布局，设置边距
fig.show()


































#【例6-6】利用plotly绘制旭日图。输入代码如下：
import plotly.graph_objects as go
import pandas as pd

# 从CSV文件读取数据
df1=pd.read_csv(r'S:\Desktop_new\caxbook_python\python_plot_202405\PyData\coffee-flavors-complete.csv')
df2=pd.read_csv(r'S:\Desktop_new\caxbook_python\python_plot_202405\PyData\coffee-flavors.csv')

fig=go.Figure()	# 创建一个空的图形对象

# 向图形对象添加第1个旭日图
fig.add_trace(go.Sunburst(ids=df1.ids,labels=df1.labels,
           parents=df1.parents,domain=dict(column=0)))

# 向图形对象添加第2个旭日图
fig.add_trace(go.Sunburst( ids=df2.ids,labels=df2.labels,
    parents=df2.parents,domain=dict(column=1),maxdepth=2
))

# 更新布局，设置网格和边距
fig.update_layout(grid=dict(columns=2,rows=1),	# 设置网格布局，2列1行
					margin=dict(t=0,l=0,r=0,b=0))	# 设置边距
fig.show()






























#【例6-7】利用plotly绘制旭日图示例1。输入代码如下：
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

df=pd.read_csv(r'S:\Desktop_new\caxbook_python\python_plot_202405\PyData\sales_success.csv')
print(df.head())

levels=['salesperson','county','region']	# 层级用于构建层次结构图表
color_columns=['sales','calls']	# 颜色列
value_column='calls'	# 数值列

def build_hierarchical_dataframe(df,levels,
						             value_column,color_columns=None):
    df_all_trees=[]
    for i,level in enumerate(levels):
        df_tree=pd.DataFrame(columns=['id','parent','value','color'])
        dfg=df.groupby(levels[i:]).sum()
        dfg=dfg.reset_index()
        df_tree['id']=dfg[level].copy()
        if i<len(levels)-1:
            df_tree['parent']=dfg[levels[i+1]].copy()
        else:
            df_tree['parent']='total'
        df_tree['value']=dfg[value_column]
        df_tree['color']=dfg[color_columns[0]]/dfg[color_columns[1]]
        df_all_trees.append(df_tree)
    total=pd.Series(dict(id='total',parent='',
             value=df[value_column].sum(),
             color=df[color_columns[0]].sum()/df[color_columns[1]].sum()))
    df_all_trees.append(pd.DataFrame(total).T)
    return pd.concat(df_all_trees,ignore_index=True)

# 构建层次结构的数据框
df_all_trees=build_hierarchical_dataframe(df,levels,
						   value_column,color_columns)
average_score=df['sales'].sum()/df['calls'].sum()

# 创建子图，1行2列
fig=make_subplots(1,2,specs=[[{"type":"domain"},{"type":"domain"}]],)

# 向第1个子图添加旭日图
fig.add_trace(go.Sunburst(labels=df_all_trees['id'],
     parents=df_all_trees['parent'],
     values=df_all_trees['value'],branchvalues='total',
     marker=dict( colors=df_all_trees['color'],
        colorscale='RdBu',cmid=average_score),
    hovertemplate='<b>%{label} </b> <br> Sales:%{value}<br>  \
         Success rate:%{color:.2f}',
    name='' ),1,1)

# 向第2个子图添加旭日图，设置最大深度为2
fig.add_trace(go.Sunburst( labels=df_all_trees['id'],
    parents=df_all_trees['parent'],values=df_all_trees['value'],
    branchvalues='total',marker=dict(colors=df_all_trees['color'],
					  colorscale='RdBu',cmid=average_score),
    hovertemplate='<b>%{label} </b> <br> Sales:%{value} <br>  \
          Success rate:%{color:.2f}',maxdepth=2 ),1,2)

# 更新布局，设置边距
fig.update_layout(margin=dict(t=10,b=10,r=10,l=10))
fig.show()




























# 6.2 树状图
#     树状图（Dendrogram）有时也称为谱系图，是一种用于可视化层级结构和分支关系的图
# 表类型。它以树的形式展示数据的层级关系，其中每个节点代表一个数据点或一个层级，而
# 分支表示节点之间的连接关系或从属关系。树状图基于给定的距离度量将相似的点组合在一
# 起，并基于点的相似性将它们组织在树状链接中。
#     树状图的优点是能够清晰地展示数据的层级结构和分支关系。它可以帮助观察数据的组
# 织结构、从属关系和分支发展，并帮助用户理解数据的分层逻辑。


#【例6-8】树状图绘制示例1。输入代码如下：
import plotly.figure_factory as ff	# 导入P figure_factory模块，用于创建图表
import numpy as np	# 导入NumPy库，用于生成随机数据

# 生成随机数据，创建一个18x10的随机数组，表示18个样本，每个样本有10个维度
X=np.random.rand(18,10)
# 创建谱系图，使用随机数据X，设置颜色阈值为1.5
fig=ff.create_dendrogram(X,color_threshold=1.5)
# 更新图表布局，设置宽度为800像素，高度为500像素
fig.update_layout(width=800,height=500)
fig.show()






























#【例6-9】树状图绘制示例2。输入代码如下：
import scipy.cluster.hierarchy as shc
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r'S:\Desktop_new\caxbook_python\python_plot_202405\PyData\USArrests.csv')		# 导入数据
# 绘制谱系图
plt.figure(figsize=(12,6),dpi=80)						# 设置图表大小
plt.title("USArrests Dendograms",fontsize=22)		# 设置图表标题
# 使用ward方法计算层次聚类，并绘制谱系图
dend=shc.dendrogram(
    shc.linkage(df[['Murder','Assault','UrbanPop','Rape']],
                method='ward'),
    labels=df.State.values,		# 设置州名为标签
    color_threshold=100)		# 设置颜色阈值，超过此值的线将以相同颜色显示
plt.xticks(fontsize=12)		# 设置x轴刻度标签的字体大小
plt.show()	# 显示谱系图





























#【例6-10】使用plotly库绘制在热图上添加树状图。输入代码如下：
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from scipy.spatial.distance import pdist,squareform

# 从文件中获取数据
data=np.genfromtxt(r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\ExpRawData_E_TABM.tab",
					     names=True,usecols=tuple(range(1,30)),
					     dtype=float,delimiter="\t")

# 转换数据格式
data_array=data.view((float,len(data.dtype.names)))
data_array=data_array.transpose()
labels=data.dtype.names					# 获取数据的列标签

# 创建一个上方的树状图
fig=ff.create_dendrogram(data_array,orientation='bottom',labels=labels)
# 将树状图中的所有数据的y轴设置为第2个y轴（即右侧的y轴）
for i in range(len(fig['data'])):
    fig['data'][i]['yaxis']='y2'

# 创建一个右侧的树状图
dendro_side=ff.create_dendrogram(data_array,orientation='right')
# 将右侧树状图中的所有数据的x轴设置为第2个x轴（即上方的x轴）
for i in range(len(dendro_side['data'])):
    dendro_side['data'][i]['xaxis']='x2'

# 将右侧树状图的数据添加到主图中
for data in dendro_side['data']:
    fig.add_trace(data)

# 获取右侧树状图的叶子节点顺序
dendro_leaves=dendro_side['layout']['yaxis']['ticktext']
dendro_leaves=list(map(int,dendro_leaves))

data_dist=pdist(data_array)			# 计算数据的距离矩阵
heat_data=squareform(data_dist)		# 将距离矩阵转换为方阵
# 根据树状图的顺序重新排列距离矩阵
heat_data=heat_data[dendro_leaves,:]
heat_data=heat_data[:,dendro_leaves]

# 创建热图
heatmap=[go.Heatmap(x=dendro_leaves,y=dendro_leaves,
					z=heat_data,colorscale='Blues')]

# 将热图的x和y轴与树状图对应的轴匹配
heatmap[0]['x']=fig['layout']['xaxis']['tickvals']
heatmap[0]['y']=dendro_side['layout']['yaxis']['tickvals']

for data in heatmap:fig.add_trace(data)	# 将热图添加到主图中

# 编辑图的布局
fig.update_layout({'width':800,'height':800,'showlegend':False,
             'hovermode':'closest'})
# 编辑x轴的样式
fig.update_layout(xaxis={'domain':[.15,1],'mirror':False,
	             'showgrid':False,'showline':False,
	             'zeroline':False,'ticks':""})
# 编辑x轴2的样式
fig.update_layout(xaxis2={'domain':[0,.15],'mirror':False,
	             'showgrid':False,'showline':False,
	             'zeroline':False,'ticks':""})
# 编辑y轴的样式
fig.update_layout(yaxis={'domain':[0,.85],'mirror':False,
	             'showgrid':False,'showline':False,
	             'zeroline':False,'ticks':""})
# 编辑y轴2的样式
fig.update_layout(yaxis2={'domain':[.825,.975],'mirror':False,
	             'showgrid':False,'showline':False,
	             'zeroline':False,'ticks':""})
fig.show()


























# 6.3 桑基图
#     桑基图（Sankey Diagram）是一种用于可视化流量、流程或能量转移的图表类型。它使
# 用有向图的方式表示数据的流动，通过不同宽度的箭头连接表示不同的流量量级，并显示出
# 流量的起点和终点。
#     桑基图的优点是能够直观地显示数据的流动和转移过程，帮助观察数据的来源、目的和
# 量级。它可用于可视化各种流程，如物质流量、能源转移、人员流动等。

#【例6-11】绘制简易桑基图，展示节点之间的流动关系和数量关系。输入代码如下：
import plotly.graph_objects as go

# 创建桑基图
fig=go.Figure(go.Sankey(
    arrangement="snap",							# 设置节点位置的排列方式
    node={"label":["A","B","C","D","E","F"],	# 节点标签
            "x":[0.2,0.1,0.5,0.7,0.3,0.5],		# 节点的x坐标
            "y":[0.7,0.5,0.2,0.4,0.2,0.3],		# 节点的y坐标
            'pad':10},							# 节点的间距
    link={"source":[0,0,1,2,5,4,3,5],			# 每条链接的源节点索引
           "target":[5,3,4,3,0,2,2,3],			# 每条链接的目标节点索引
           "value":[1,2,1,1,1,1,1,2]} 			# 每条链接的值，表示流动的数量
))
fig.show()
































#【例6-12】通过绘制桑基图展示2050年能源预测数据。输入代码如下：
import plotly.graph_objects as go
import json

# 从本地文件获取数据
file_path='S:\Desktop_new\caxbook_python\python_plot_202405\PyData\sankey_energy.json'
with open(file_path,'r',encoding='gb18030',errors = 'ignore') as file:data=json.load(file)

# 重写灰色链接的颜色为对应源节点的颜色，并添加透明度
opacity=0.4
data['data'][0]['node']['color']=['rgba(255,0,255,0.8)' 
	             if color=="magenta" else color 
	             for color in data['data'][0]['node']['color']]
data['data'][0]['link']['color']=[data['data'][0]['node']['color']
	              [src].replace("0.8",str(opacity))
	             for src in data['data'][0]['link']['source']]

# 创建 Sankey 图
fig=go.Figure(data=[go.Sankey(
    valueformat=".0f",
    valuesuffix="TWh",
    # 定义节点
    node=dict( pad=15,thickness=15,
             line=dict(color="black",width=0.5),
             label=data['data'][0]['node']['label'],
             color=data['data'][0]['node']['color']),
    # 添加链接
    link=dict( source=data['data'][0]['link']['source'],
             target=data['data'][0]['link']['target'],
             value=data['data'][0]['link']['value'],
             label=data['data'][0]['link']['label'],
             color=data['data'][0]['link']['color']))])

# 设置图表标题和字体大小
fig.update_layout(title_text="Energy forecast for 2050",font_size=10)
fig.show()






























# 矩形树状图
#     矩形树状图（Rectangular Tree Diagram）是一种用于可视化层级结构和分支关系的图表
# 类型。它以矩形的形式展示数据的层级关系，其中每个矩形代表一个数据点或一个层级，而
# 矩形之间的相对位置和大小表示节点之间的连接关系或从属关系。
#     矩形树状图的优点是能够清晰地展示数据的层级结构和分支关系，并通过矩形的相对位
# 置和大小来传达从属关系。它可以帮助观察数据的组织结构、层级关系和分支发展，并帮助
# 用户理解数据的分层逻辑。

#【例6-13】矩形树状图创建示例。输入代码如下：
import plotly.express as px
import numpy as np

# 从plotly中导入gapminder数据集，并选择2007年的数据
df=px.data.gapminder().query("year==2007")

# 使用treemap图表绘制
fig=px.treemap(df,
                 path=[px.Constant("world"),'continent','country'],
                 values='pop',color='lifeExp',
                 hover_data=['iso_alpha'],
                 color_continuous_scale='RdBu',
                 color_continuous_midpoint=np.average(df['lifeExp'],
						         weights=df['pop']))

fig.update_layout(margin=dict(t=50,l=25,r=25,b=25))	# 更新图表布局
fig.show()

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

df=pd.read_csv('S:\Desktop_new\caxbook_python\python_plot_202405\PyData\sales_success.csv')# 读取数据集
print(df.head())			# 打印数据集的前几行，输出略


# 设置层级关系
levels=['salesperson','county','region']
color_columns=['sales','calls']
value_column='calls'

def build_hierarchical_dataframe(df,levels,
						value_column,color_columns=None):
    """
    构建用于Sunburst或Treemap图的层次结构。
    从底层到顶层给出层次关系，即最后一级对应根。
    """
    df_all_trees=pd.DataFrame(columns=['id','parent','value','color'])
    for i,level in enumerate(levels):
        df_tree=pd.DataFrame(columns=['id','parent','value','color'])
        dfg=df.groupby(levels[i:]).sum()
        dfg=dfg.reset_index()
        df_tree['id']=dfg[level].copy()
        if i<len(levels)-1:
            df_tree['parent']=dfg[levels[i+1]].copy()
        else:
            df_tree['parent']='total'
        df_tree['value']=dfg[value_column]
        df_tree['color']=dfg[color_columns[0]]/dfg[color_columns[1]]
        df_all_trees=pd.concat([df_all_trees,df_tree],ignore_index=True)
    total=pd.Series(dict(id='total',parent='',
              value=df[value_column].sum(),
             color=df[color_columns[0]].sum()/df[color_columns[1]].sum()))
    df_all_trees=pd.concat([df_all_trees,pd.DataFrame([total],
          columns=total.index)],ignore_index=True)
    return df_all_trees

# 构建层次结构数据
df_all_trees=build_hierarchical_dataframe(df,levels,
	                value_column,color_columns)
average_score=df['sales'].sum()/df['calls'].sum()

# 创建子图
fig=make_subplots(1,2,specs=[[{"type":"domain"},{"type":"domain"}]],)

# 添加Treemap图表
fig.add_trace(go.Treemap(
    labels=df_all_trees['id'],
    parents=df_all_trees['parent'],
    values=df_all_trees['value'],
    branchvalues='total',
    marker=dict( colors=df_all_trees['color'],colorscale='RdBu',
                cmid=average_score),
    hovertemplate='<b>%{label} </b> <br> Sales:%{value}<br>   \
             Success rate:%{color:.2f}',
    name='' ),1,1)

# 添加Treemap图表，设置最大深度为2
fig.add_trace(go.Treemap(
    labels=df_all_trees['id'],
    parents=df_all_trees['parent'],
    values=df_all_trees['value'],
    branchvalues='total',
    marker=dict(
        colors=df_all_trees['color'],
        colorscale='RdBu',
        cmid=average_score),
    hovertemplate='<b>%{label} </b> <br> Sales:%{value}<br>    \
        Success rate:%{color:.2f}',
    maxdepth=2
    ),1,2)

# 更新图表布局
fig.update_layout(margin=dict(t=50,l=25,r=25,b=25))
fig.show()





























# 6.5 圆堆积图
#     圆堆积图（Circle Packing）是树形图的变体，使用圆形（而非矩形）一层又一层地代表
# 整个层次结构：树木的每个分支由一个圆圈表示，而其子分支则以圆圈内的圆圈来表示。每
# 个圆形的面积也可用来表示额外任意数值，如数量或文件大小。也可用颜色将数据进行分类，
# 或通过不同色调表示另一个变量。

#【例6-14】利用Matplotlib与Plotly库绘制圆堆积图。输入代码如下：
import circlify
import matplotlib.pyplot as plt  # 导入 matplotlib 库用于绘图
import plotly.graph_objects as go  # 导入 plotly 库用于交互式绘图

magnitudes=[2,10,12,23,65,87]# 定义数据列表

# 计算圆的位置和大小
circles=circlify.circlify(magnitudes,show_enclosure=False,
    target_enclosure=circlify.Circle(x=0,y=0,r=1))

# 根据父圆位置和大小创建子圆
child_circle_groups=[]
for i in range(len(magnitudes)):
    child_circle_groups.append(circlify.circlify(
        magnitudes,show_enclosure=False,
        target_enclosure=circlify.Circle(x=circles[i].x,
                                y=circles[i].y,r=circles[i].r)))

# Matplotlib绘图
fig,ax=plt.subplots(figsize=(10,10))		# 创建一个图形和一个子图

# 设置图形属性
ax.axis('off')								# 关闭坐标轴
lim=max(max(abs(circle.x)+ circle.r,
              abs(circle.y)+ circle.r,)
    for circle in circles)
plt.xlim(-lim,lim)							# 设置 x 轴范围
plt.ylim(-lim,lim)							# 设置 y 轴范围

# 添加父圆
for circle in circles:
    x,y,r=circle
    ax.add_patch(plt.Circle((x,y),r,alpha=0.2,linewidth=2,fill=False))

# 添加子圆
for child_circles in child_circle_groups:
    for child_circle in child_circles:
        x,y,r=child_circle
        ax.add_patch(plt.Circle((x,y),r,alpha=0.2,linewidth=2,fill=False))
plt.show()

# Plotly绘图
fig=go.Figure()								# 创建一个新的图形
# 设置坐标轴属性
fig.update_xaxes(range=[-1.05,1.05],		# 设置 x 轴范围
    showticklabels=False,					# 不显示刻度标签
    showgrid=False,							# 不显示网格线
    zeroline=False)							# 不显示零线
fig.update_yaxes(range=[-1.05,1.05],		# 设置 y 轴范围
    showticklabels=False,					# 不显示刻度标签
    showgrid=False,							# 不显示网格线
    zeroline=False,)							# 不显示零线
# 添加父圆
for circle in circles:
    x,y,r=circle
    fig.add_shape(type="circle",xref="x",yref="y",
                  x0=x-r,y0=y-r,x1=x+r,y1=y+r,
                  line_color="LightSeaGreen",			# 设置圆边框颜色
                  line_width=2)						# 设置圆边框宽度
# 添加子圆
for child_circles in child_circle_groups:
    for child_circle in child_circles:
        x,y,r=child_circle
        fig.add_shape(type="circle",xref="x",yref="y",
                      x0=x-r,y0=y-r,x1=x+r,y1=y+r,
                      line_color="LightSeaGreen",		# 设置圆边框颜色
                      line_width=2)					# 设置圆边框宽度
# 设置图形大小
fig.update_layout(width=800,height=800,plot_bgcolor="white")
fig.show()


