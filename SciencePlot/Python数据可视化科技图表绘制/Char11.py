# 十一、网络关系数据可视化
#     网络关系数据描述了个体或实体之间的相互联系，这些联系可以是社交网络中的友谊关
# 系、互联网上的网页链接、生物学中的蛋白质相互作用，或者是任何其他形式的关系。本章
# 探讨如何使用Python来实现网络关系数据的可视化，希望读者在面对网络关系数据时，能够
# 灵活、准确地选择合适的可视化方法，并从中获得有价值的见解和发现。

# 11.1 节点链接图
#     节点连接图（Node-Link Diagram），也称为网络图或关系图，是一种用于可视化节点之
# 间关系的图表类型。它通过使用节点和连接线表示数据中的实体和它们之间的关系，帮助观
# 察和分析复杂的网络结构。
#     节点链接图的优点是能够直观地显示节点之间的连接关系和结构。它可以帮助观察数据
# 的网络、集群或关联模式，并揭示数据中隐藏的关系和趋势。

#【例11-1】利用igraph包绘制节点链接图示例。输入代码如下：
import plotly.graph_objects as go
import networkx as nx

# 创建一个具有200个节点和连接概率为0.125的随机几何图形图
G = nx.random_geometric_graph(200, 0.8)
# 初始化边的坐标列表
edge_x = []
edge_y = []
# 遍历图中的每条边，提取节点坐标信息
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    # 添加边的起点和终点的坐标到对应的列表中，并使用None分隔每条边
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

# 创建边的散点图
edge_trace = go.Scatter(x=edge_x, y=edge_y,
                        line=dict(width=0.5, color='#888'),  # 设置边的样式
                        hoverinfo='none', mode='lines')

# 初始化节点的坐标列表
node_x = []
node_y = []
# 遍历图中的每个节点，提取节点坐标信息
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    # 添加节点的坐标到对应的列表中
    node_x.append(x)
    node_y.append(y)

# 创建节点的散点图
node_trace = go.Scatter(x=node_x, y=node_y,
                        mode='markers', hoverinfo='text',
                        # 设置节点的颜色和大小
                        marker=dict(showscale=True,
                                    colorscale='YlGnBu',  # 颜色映射
                                    reversescale=True,
                                    color=[],  # 存储节点的连接数
                                    size=10,
                                    colorbar=dict(thickness=15, title='Node Connections',
                                                  xanchor='left', titleside='right'),
                                    line_width=2))

# 存储每个节点的邻接节点数和文本信息
node_adjacencies = []
node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append('# of connections:' + str(len(adjacencies[1])))

# 将节点的邻接节点数赋给节点的颜色属性，将文本信息赋给节点的悬停文本属性
node_trace.marker.color = node_adjacencies
node_trace.text = node_text

# 创建图形对象
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='<br>Network graph made with Python',  # 设置图的标题
                    titlefont_size=16, showlegend=False, hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    # 添加注释
                    annotations=[dict(text="Python code", showarrow=False,
                                      xref="paper", yref="paper", x=0.005, y=-0.002)],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
fig.show()






























#【例11-2】使用Bokeh绘制一个图形交互演示，展示了一个社交网络图。输入代码如下：
import networkx as nx
from bokeh.models import MultiLine, Scatter
from bokeh.plotting import figure, from_networkx, show

G = nx.karate_club_graph()  # 创建一个 Karate Club 图

# 定义同一俱乐部和不同俱乐部的边颜色
SAME_CLUB_COLOR, DIFFERENT_CLUB_COLOR = "darkgrey", "red"

edge_attrs = {}
# 遍历每条边，设置边的颜色属性
for start_node, end_node, _ in G.edges(data=True):
    edge_color = SAME_CLUB_COLOR if G.nodes[start_node]["club"] \
                                    == G.nodes[end_node]["club"] else DIFFERENT_CLUB_COLOR
    edge_attrs[(start_node, end_node)] = edge_color

# 设置边的颜色属性
nx.set_edge_attributes(G, edge_attrs, "edge_color")

# 创建 Bokeh 图表对象，并将背景设置为白色
plot = figure(width=400, height=400, x_range=(-1.2, 1.2), y_range=(-1.2, 1.2),
              x_axis_location=None, y_axis_location=None,
              toolbar_location=None,
              title="Graph Interaction Demo", background_fill_color="white",
              tooltips="index: @index,club: @club")
plot.grid.grid_line_color = None

# 从 NetworkX 图创建图渲染器
graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0, 0))
# 设置节点渲染器
graph_renderer.node_renderer.glyph = Scatter(size=15,
                                             fill_color="lightblue")
# 设置边渲染器
graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color",
                                               line_alpha=1, line_width=2)
# 将图渲染器添加到图表中
plot.renderers.append(graph_renderer)
show(plot)



























# 11.2 弧线图
#     弧线图（Arc Diagram）是一种用于可视化关系和连接的图表类型。它通过使用弧线来表
# 示数据中的连接关系，帮助展示复杂网络的结构和模式。
#     弧线图的优点是能够简明地显示节点之间的关系和连接，帮助观察数据中的模式和趋势。
# 它常用于显示关系网络、时间序列数据、进化树等。


#【例11-3】简单弧线图的绘制示例。输入代码如下：
import numpy as np
import matplotlib.pyplot as plt

plt.figure('example')
plt.axis([-10, 140, 90, -10])

plt.axis('off')
plt.grid(False)

#显示辅助坐标系
plt.arrow(0,0, 20,0, head_length = 4, head_width = 3, color = 'k')
plt.arrow(0,0, 0,20, head_length = 4, head_width = 3, color = 'k')
plt.text(15, -3, 'x')
plt.text(-5, 15, 'y')

#
xc = 20
yc = 20
r = 40
#画圆心
plt.scatter(xc, yc, color = 'b', s = 5)
#画一段圆弧
phi1 = 20*np.pi/180.0
phi2 = 70*np.pi/180.0
dphi = (phi2 - phi1)/20.0
for phi in np.arange(phi1, phi2, dphi):
    x = xc + r*np.cos(phi)
    y = xc + r*np.sin(phi)
    plt.scatter(x, y, s = 2, color = 'g')

#连接圆心与端点（x1,y1)
plt.plot([xc, xc + r*np.cos(phi1)], [yc, yc + r*np.sin(phi1)], color = 'k')

#p1指示线段
x1 = xc + (r + 3)*np.cos(phi1)
x2 = xc + (r + 10)*np.cos(phi1)
y1 = yc + (r + 3)*np.sin(phi1)
y2 = yc + (r + 10)*np.sin(phi1)
plt.plot([x1, x2], [y1, y2], color = 'k')

#p2指示线段
x1 = xc + (r + 3)*np.cos(phi2)
x2 = xc + (r + 30)*np.cos(phi2)
y1 = yc + (r + 3)*np.sin(phi2)
y2 = yc + (r + 30)*np.sin(phi2)
plt.plot([x1, x2], [y1, y2], color = 'k')

#连接圆心与端点（x2, y2)
plt.plot([xc, xc + r*np.cos(phi2)], [yc, yc + r*np.sin(phi2)], color = 'k')

#中间位置显示角度变化量
phihalf = (phi1 + phi2)*0.5
phi3 = phihalf - dphi/2
phi4 = phihalf + dphi/2

plt.plot([xc, xc + r*np.cos(phi3)], [yc, yc + r*np.sin(phi3)], color = 'k')
plt.plot([xc, xc + r*np.cos(phi4)], [yc, yc + r*np.sin(phi4)], color = 'k')

#dp1指示线段
x1 = xc + (r + 3)*np.cos(phi3)
x2 = xc + (r + 15)*np.cos(phi3)
y1 = yc + (r + 3)*np.sin(phi3)
y2 = yc + (r + 15)*np.sin(phi3)
plt.plot([x1, x2], [y1, y2], color = 'k')

#dp2指示线段
x1 = xc + (r + 3)*np.cos(phi4)
x2 = xc + (r + 15)*np.cos(phi4)
y1 = yc + (r + 3)*np.sin(phi4)
y2 = yc + (r + 15)*np.sin(phi4)
plt.plot([x1, x2], [y1, y2], color = 'k')

#p1圆弧
dphi = (phi3)/100
for phi in np.arange(0, phi1/2 - 3.2*np.pi/180, dphi):
    x = xc + (r + 5)*np.cos(phi)
    y = yc + (r + 5)*np.sin(phi)
    plt.scatter(x, y, s = 0.1, color = 'k')

for phi in np.arange(phi1/2 + 3.3*np.pi/180, phi1, dphi):
    x = xc + (r + 5)*np.cos(phi)
    y = yc + (r + 5)*np.sin(phi)
    plt.scatter(x, y, s = 0.1, color = 'k')

#p2圆弧
dphi = (phi3)/100
for phi in np.arange(0, phi2/2 - 3.2*np.pi/180, dphi):
    x = xc + (r + 25)*np.cos(phi)
    y = yc + (r + 25)*np.sin(phi)
    plt.scatter(x, y, s = 0.1, color = 'k')

for phi in np.arange(phi2/2 + 3.2*np.pi/180, phi2, dphi):
    x = xc + (r + 25)*np.cos(phi)
    y = yc + (r + 25)*np.sin(phi)
    plt.scatter(x, y, s = 0.1, color = 'k')
#p圆弧
dphi = (phi3)/100
for phi in np.arange(0, phi3/2 - 0.5*np.pi/180, dphi):
    x = xc + (r + 13)*np.cos(phi)
    y = yc + (r + 13)*np.sin(phi)
    plt.scatter(x, y, s = 0.1, color = 'k')

for phi in np.arange(phi3/2 + 9.0*np.pi/180, phi3, dphi):
    x = xc + (r + 13)*np.cos(phi)
    y = yc + (r + 13)*np.sin(phi)
    plt.scatter(x, y, s = 0.1, color = 'k')
#dp圆弧
dphi = (phi3)/100
for phi in np.arange(phi3 + 5*dphi, phi3 + 25*dphi, dphi):
    x = xc + (r + 13)*np.cos(phi)
    y = yc + (r + 13)*np.sin(phi)
    plt.scatter(x, y, s = 0.1, color = 'k')
#画直角坐标线
plt.plot([xc, 100], [yc, yc], 'k')
plt.plot([xc, xc], [yc, 80], 'k')

#显示标签
plt.text(71, 58, 'p2', size = 'small')
plt.text(66, 44, 'p', size = 'small')
plt.text(63, 29, 'p1', size = 'small')
plt.text(45, 66, 'dp', size = 'small')
plt.text(41, 26, 'r', size = 'small')
plt.text(3, 17, '(xc, yc)', size = 'small')

#显示R*COS
plt.plot([xc + r*np.cos(phi3), xc + r*np.cos(phi3)], [yc - 8, yc + r*np.sin(phi3)], 'k:')
plt.plot([xc, xc], [yc-2, yc - 8], 'k:')

plt.text(25, 17, 'R*cos(p)', size = 'small')

#显示R*SIN
plt.plot([xc - 8, xc + r*np.cos(phi3)], [yc + r*np.sin(phi3), yc + r*np.sin(phi3)], 'k:')
plt.plot([xc - 2, xc - 8], [yc, yc], 'k:')
plt.text(13, 37, 'R*sin(p)', size = 'small', rotation = 90)

#
plt.text(49, 30, '(x1, y1)', size = 'small')
plt.text(20, 62, '(x2, y2)', size = 'small')
plt.text(51, 49, '(xp, yp)', size = 'small')

#显示箭头
#p2
plt.arrow(47, 79, -2, 1, head_length = 3, head_width = 2, color = 'k')
#p
plt.arrow(62, 53, -2, 2, head_length = 3, head_width = 2, color = 'k')
#p1
plt.arrow(64, 31, -0.9, 2, head_length = 3, head_width = 2, color = 'k')
#dp
plt.arrow(52, 63, 3, -3, head_length = 3, head_width = 2, color = 'k')

plt.show()























# 11.3 蜂巢图
#     蜂巢图（Hive Plot）是一种用于可视化多变量之间关系的图表类型。它通过使用多个轴
# 线和连接线在一个蜂巢形状的布局中显示数据变量和它们之间的关系，帮助观察和分析多变
# 量之间的模式和趋势。
#     蜂巢图的优点是能够同时显示多个变量之间的关系，并将它们组织在一个紧凑且可视化
# 明确的布局中，常用于可视化复杂的关系网络、多维数据分析等。

#【例11-4】利用hiveplotlib库绘制蜂巢图。输入代码如下：
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from hiveplotlib import hive_plot_n_axes  # 用于创建蜂窝图
from hiveplotlib.converters import networkx_to_nodes_edges
# 用于将 NetworkX 图转换为节点和边
from hiveplotlib.node import split_nodes_on_variable  # 用于根据变量拆分节点
from hiveplotlib.viz import hive_plot_viz  # 用于可视化蜂窝图

# 生成具有指定块大小和概率的随机块模型图
G = nx.stochastic_block_model(sizes=[10, 10, 10],
                              p=[[0.1, 0.5, 0.5], [0.05, 0.1, 0.2], [0.05, 0.2, 0.1]],
                              directed=True, seed=0)

# 将图转换为节点和边
nodes, edges = networkx_to_nodes_edges(G)

# 根据变量将节点拆分为块
blocks_dict = split_nodes_on_variable(nodes, variable_name="block")
splits = list(blocks_dict.values())

# 不关心轴上的模式，将节点随机放置在轴上
rng = np.random.default_rng(0)
for node in nodes:
    node.add_data(data={"val": rng.uniform()})

# 创建蜂窝图，并指定节点、边、轴的分配情况以及排序变量
hp = hive_plot_n_axes(node_list=nodes, edges=edges, axes_assignments=splits,
                      sorting_variables=["val"] * 3)
fig, ax = hive_plot_viz(hp)
ax.set_title("Stochastic Block Model,Base Hive Plot Visualization",
             y=1.05, size=20)
plt.show()

# 在可视化函数中更改节点和轴的参数
fig, ax = hive_plot_viz(hp, node_kwargs={"color": "C1", "s": 80},
                        axes_kwargs={"color": "none"}, color="C0", ls="dotted")
ax.set_title("Stochastic Block Model,Changing Kwargs in Viz Function",
             y=1.05, size=20)
plt.show()

# 将所有3个组的重复轴打开
hp = hive_plot_n_axes(node_list=nodes, edges=edges,
                      axes_assignments=splits,
                      sorting_variables=["val"] * 3,
                      repeat_axes=[True, True, True],
                      all_edge_kwargs={"color": "darkgrey"},
                      repeat_edge_kwargs={"color": "C0"},
                      ccw_edge_kwargs={"color": "C1"}, )

# 在图中添加文本说明
fig, ax = hive_plot_viz(hp)
fig.text(0.12, 0.95, "Less", ha="left", va="bottom",
         fontsize=20, color="black")
fig.text(0.19, 0.95, "intra-group activity", weight="heavy",
         ha="left", va="bottom", fontsize=20, color="C0", )
fig.text(0.5, 0.95, "relative to", ha="left", va="bottom",
         fontsize=20, color="black")
fig.text(0.65, 0.95, "inter-group activity", weight="heavy",
         ha="left", va="bottom", fontsize=20, color="darkgrey", )
plt.show()


































# 11.4 和弦图
#     和弦图（Chord Diagram）是一种用于可视化关系和流量的图表类型。它通过使用弦和节
# 点来表示数据中的实体和它们之间的关系，帮助展示复杂网络中的交互和连接模式。
#     弦的宽度和颜色可以用来表示关系的强度或流量的大小，较宽或较深的弦可能表示较强
# 的关系或较大的流量。
#     弦图的优点是能够直观地显示实体之间的关系和流量，并帮助观察数据中的交互模式。
# 它常用于显示社交网络、流量分析、组织结构等。

#【例11-5】使用pycirclize库创建和弦图1。输入代码如下：
from pycirclize import Circos  # 导入pycirclize库中的Circos类
import pandas as pd  # 导入pandas库

# 创建矩阵数据框（3行6列）
row_names = ["S1", "S2", "S3"]  # 行名称
col_names = ["E1", "E2", "E3", "E4", "E5", "E6"]  # 列名称
matrix_data = [  # 矩阵数据
    [4, 14, 13, 17, 5, 2],
    [7, 1, 6, 8, 12, 15],
    [9, 10, 3, 16, 11, 18],
]
matrix_df = pd.DataFrame(matrix_data, index=row_names,
                         columns=col_names)  # 使用pandas创建数据框

# 从矩阵初始化和弦图（也可以直接加载 tsv 格式的矩阵文件）
circos = Circos.initialize_from_matrix(
    matrix_df,
    start=-265,  # 起始角度
    end=95,  # 结束角度
    space=5,  # 每个扇形之间的间隔
    r_lim=(93, 100),  # 设置半径范围
    cmap="tab10",  # 颜色映射
    label_kws=dict(r=94, size=12, color="white"),  # 标签参数
    link_kws=dict(ec="black", lw=0.5),  # 连接线参数
)

print(matrix_df)  # 打印矩阵数据框
fig = circos.plotfig()  # 绘制 和弦图















print('5')

















#【例11-6】使用pycirclize库创建和弦图2。输入代码如下：
from pycirclize import Circos  # 导入pycirclize库中的Circos类
import pandas as pd  # 导入pandas库

# 创建矩阵数据（10行10列）
row_names = list("ABCDEFGHIJ")  # 行名称
col_names = row_names  # 列名称与行名称相同
matrix_data = [  # 矩阵数据
    [51, 115, 60, 17, 120, 126, 115, 179, 127, 114],
    [108, 138, 165, 170, 85, 221, 75, 107, 203, 79],
    [108, 54, 72, 123, 84, 117, 106, 114, 50, 27],
    [62, 134, 28, 185, 199, 179, 74, 94, 116, 108],
    [211, 114, 49, 55, 202, 97, 10, 52, 99, 111],
    [87, 6, 101, 117, 124, 171, 110, 14, 175, 164],
    [167, 99, 109, 143, 98, 42, 95, 163, 134, 78],
    [88, 83, 136, 71, 122, 20, 38, 264, 225, 115],
    [145, 82, 87, 123, 121, 55, 80, 32, 50, 12],
    [122, 109, 84, 94, 133, 75, 71, 115, 60, 210],
]
matrix_df = pd.DataFrame(matrix_data, index=row_names,
                         columns=col_names)  # 使用pandas创建数据框

# 从矩阵初始化和弦图（也可以直接加载tsv格式的矩阵文件）
circos = Circos.initialize_from_matrix(
    matrix_df,
    space=3,  # 每个扇形之间的间隔
    r_lim=(93, 100),  # 设置半径范围
    cmap="tab10",  # 颜色映射
    ticks_interval=500,  # 刻度间隔
    label_kws=dict(r=94, size=12, color="white"),  # 标签参数
)

print(matrix_df)  # 打印矩阵数据框
fig = circos.plotfig()  # 绘制和弦图












print('6')




















#【例11-7】使用pycircliz
from pycirclize import Circos  # 导入pycirclize库中的Circos类
from pycirclize.parser import Matrix  # 导入pycirclize库中的Matrix解析器
import pandas as pd  # 导入pandas库

# 创建from-to表格数据框并转换为矩阵
fromto_table_df = pd.DataFrame([["A", "B", 10], ["A", "C", 5],
                                ["A", "D", 15], ["A", "E", 20], ["A", "F", 3],
                                ["B", "A", 3], ["B", "G", 15], ["F", "D", 13],
                                ["F", "E", 2], ["E", "A", 20], ["E", "D", 6], ],
                               columns=["from", "to", "value"], )  # 列名（可选）

matrix = Matrix.parse_fromto_table(fromto_table_df)
# 从from-to表格数据框解析矩阵

# 从矩阵初始化和弦图
circos = Circos.initialize_from_matrix(matrix,
                                       space=3,  # 每个扇形之间的间隔
                                       cmap="viridis",  # 颜色映射
                                       ticks_interval=5,  # 刻度间隔
                                       label_kws=dict(size=12, r=110),  # 标签参数
                                       link_kws=dict(direction=1, ec="black", lw=0.5), )  # 连接线参数

print(fromto_table_df.to_string(index=False))  # 打印from-to表格数据框
fig = circos.plotfig()  # 绘制和弦图
























print('7')




# 11.5 切尔克斯图
#     切尔科斯图（Circos）是一种用于可视化循环数据的强大工具，它的特点是将数据以圆
# 环的形式展示出来，便于观察和理解数据之间的关系。通常情况下，切尔科斯图用于展示染
# 色体的结构、基因组之间的相互作用、基因表达数据等循环性质的数据。切尔科斯图的组成
# 包括：
#     （1）圆环结构：切尔科斯图以一个圆环的形式展示数据，圆环被分为若干个扇形区域，
# 每个区域代表数据中的一个部分或者数据集。
#     （2）数据编码：切尔科斯图通过将数据映射到圆环的不同区域来展示数据，这些区域
# 可以代表不同的实体、样本或者特征。数据可以通过扇形区域的长度、颜色、弧度等来编码，
# 以表达不同的属性或者数值。
#     （3）连接线：切尔科斯图还可以显示数据之间的连接关系，这些连接关系通过连接扇
# 形区域的方式展示。连接线的粗细、颜色等属性可以反映连接的强度、类型或者其他信息。
#     （4）标签和刻度：切尔科斯图通常会在圆环的外部添加标签和刻度，用于标识不同的
# 区域或者提供数据的附加信息。

#【例11-8】使用pycirclize库创建一个切尔科斯图。输入代码如下：
from pycirclize import Circos  # 导入pycirclize库中的Circos类
import numpy as np  # 导入NumPy库

np.random.seed(0)  # 设置随机种子，确保每次运行结果一致

# 初始化切尔科斯图的扇形区域
sectors = {"A": 10, "B": 15, "C": 12, "D": 20, "E": 15}
circos = Circos(sectors, space=5)  # 创建Circos对象，指定扇形区域和间隔

for sector in circos.sectors:  # 循环遍历每个扇形区域
    # 绘制扇形区域的名称
    sector.text(f"Sector: {sector.name}", r=110, size=15)
    # 创建x坐标位置和随机的y值
    x = np.arange(sector.start, sector.end) + 0.5
    y = np.random.randint(0, 100, len(x))

    # 绘制线条
    track1 = sector.add_track((80, 100), r_pad_ratio=0.1)  # 在扇形区域内添加轨道
    track1.xticks_by_interval(interval=1)  # 设置刻度间隔
    track1.axis()  # 添加轨道的坐标轴
    track1.line(x, y)  # 绘制线条

    # 绘制散点
    track2 = sector.add_track((55, 75), r_pad_ratio=0.1)  # 添加轨道
    track2.axis()  # 添加坐标轴
    track2.scatter(x, y)  # 绘制散点图

    # 绘制条形图
    track3 = sector.add_track((30, 50), r_pad_ratio=0.1)  # 添加轨道
    track3.axis()  # 添加坐标轴
    track3.bar(x, y)  # 绘制条形图

# 绘制连接线
circos.link(("A", 0, 3), ("B", 15, 12))  # 连接扇形区域A和B
circos.link(("B", 0, 3), ("C", 7, 11), color="skyblue")
# 连接扇形区域B和C，指定颜色为天蓝色
circos.link(("C", 2, 5), ("E", 15, 12), color="chocolate", direction=1)
# 连接扇形区域C和E，指定颜色为巧克力色，设置方向为逆时针
circos.link(("D", 3, 5), ("D", 18, 15), color="lime",
            ec="black", lw=0.5, hatch="//", direction=2)
# 连接扇形区域D和D，指定颜色为酸橙色，边缘颜色为黑色，填充样式为斜线
circos.link(("D", 8, 10), ("E", 2, 8), color="violet",
            ec="red", lw=1.0, ls="dashed")
# 连接扇形区域D和E，指定颜色为紫罗兰色，边缘颜色为红色，线型为虚线

circos.savefig("example01.png")  # 将绘制好的切尔科斯图保存为图片文件
fig = circos.plotfig()  # 绘制切尔科斯图










print('8')



















#【例11-9】利用pycirclize库创建一个切尔科斯图图，展示大肠杆菌质粒（NC_002483）的基因组结构信息，包括基因的分布和定位。输入代码如下：
from pycirclize import Circos
from pycirclize.utils import fetch_genbank_by_accid  # 用于下载GenBank数据
from pycirclize.parser import Genbank  # 用于解析GenBank数据
import matplotlib.pyplot as plt

# 下载大肠杆菌质粒（NC_002483）的GenBank数据
gbk_fetch_data = fetch_genbank_by_accid("NC_002483")
gbk = Genbank(gbk_fetch_data)  # 使用GenBank数据初始化Genbank解析器

# 使用基因组大小初始化Circos实例
circos = Circos(sectors={gbk.name: gbk.range_size})
circos.text(f"Escherichia coli K-12 plasmid F\n\n{gbk.name}",
            size=14)  # 在图中添加文本信息
circos.rect(r_lim=(90, 100), fc="lightgrey", ec="none",
            alpha=0.5)  # 绘制一个灰色矩形
sector = circos.sectors[0]  # 获取第一个扇形区域

# 绘制正向链CDS
f_cds_track = sector.add_track((95, 100))
f_cds_feats = gbk.extract_features("CDS", target_strand=1)
f_cds_track.genomic_features(f_cds_feats, plotstyle="arrow",
                             fc="salmon", lw=0.5)

# 绘制反向链CDS
r_cds_track = sector.add_track((90, 95))
r_cds_feats = gbk.extract_features("CDS", target_strand=-1)
r_cds_track.genomic_features(r_cds_feats, plotstyle="arrow",
                             fc="skyblue", lw=0.5)

# 绘制'gene' qualifier标签（如果存在）
labels, label_pos_list = [], []
for feat in gbk.extract_features("CDS"):
    start = int(feat.location.start)
    end = int(feat.location.end)
    label_pos = (start + end) / 2
    gene_name = feat.qualifiers.get("gene", [None])[0]
    if gene_name is not None:
        labels.append(gene_name)
        label_pos_list.append(label_pos)
f_cds_track.xticks(label_pos_list, labels, label_size=6,
                   label_orientation="vertical")

# 绘制x轴刻度（间隔为10Kb）
r_cds_track.xticks_by_interval(10000, outer=False,
                               label_formatter=lambda v: f"{v / 1000:.1f} Kb")
fig = circos.plotfig()  # 绘制Circos图
plt.show()