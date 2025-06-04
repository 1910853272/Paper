# 九、时间序列数据可视化
#     时间序列数据是在各个领域中广泛使用的一种数据类型，它记录了随着时间推移而收集
# 的观测值或测量结果。时间序列数据可用于分析趋势、季节性、周期性和异常事件等，这些
# 信息对于数据分析、预测和决策制定都至关重要。Python提供了丰富的可视化包和函数，用
# 于处理和可视化时间序列数据。

# 9.1 折线图
#     折线图（Line Chart）用于显示随时间、顺序或其他连续变量变化的趋势和模式。它通过
# 连接数据点来展示数据的变化，并利用直线段来表示数据的趋势。
#     折线图的优点是能够清晰地展示变量随时间或顺序的变化趋势，可以帮助观察者发现趋
# 势、周期性、增长或下降趋势等。它常用于分析时间序列数据、比较不同组的趋势、展示实
# 验结果的变化等。

#【例9-1】通过绘制的折线图查看1949年至1969年间航空客运量的变化情况。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('S:\Desktop_new\caxbook_python\python_plot_202405\PyData\AirPassengers.csv')	# 导入数据
plt.figure(figsize=(10,6),dpi=300)		# 绘制图表
plt.plot('date','value',data=df,color='tab:red')		# 绘制折线图

# 图表修饰
plt.ylim(50,750)						# 设置y轴范围
# 设置x轴刻度位置和标签
xtick_location=df.index.tolist()[::12]
xtick_labels=[x[-4:]for x in df.date.tolist()[::12]]
plt.xticks(ticks=xtick_location,labels=xtick_labels,rotation=0,
           fontsize=12,horizontalalignment='center',alpha=.7)
plt.yticks(fontsize=12,alpha=.7)		# 设置y轴刻度标签的字体大小和透明度
plt.title("Air Passengers Traffic (1949-1969)",fontsize=18)	# 设置标题
plt.grid(axis='both',alpha=.3)			# 添加网格线，设置透明度

# 移除边框
plt.gca().spines["top"].set_alpha(0.0)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)
plt.gca().spines["left"].set_alpha(0.3)
plt.show()






























#【例9-2】绘制带波峰波谷标记的折线图，并注释了所选特殊事件的发生。输入代码如下：
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

df=pd.read_csv('S:\Desktop_new\caxbook_python\python_plot_202405\PyData\AirPassengers.csv')	# 导入数据
data=df['value'].values									# 获取峰值和谷值的位置

# 计算一阶差分
doublediff=np.diff(np.sign(np.diff(data)))
peak_locations=np.where(doublediff==-2)[0]+1

# 计算负数序列的一阶差分
doublediff2=np.diff(np.sign(np.diff(-1*data)))
trough_locations=np.where(doublediff2==-2)[0]+1

plt.figure(figsize=(10,6),dpi=300)			# 绘制图表
# 绘制折线图
plt.plot('date','value',data=df,color='tab:blue',label='Air Traffic')
# 绘制峰值和谷值的散点图
plt.scatter(df.date[peak_locations],df.value[peak_locations],
            marker=mpl.markers.CARETUPBASE,color='tab:green',
            s=100,label='Peaks')
plt.scatter(df.date[trough_locations],df.value[trough_locations],
            marker=mpl.markers.CARETDOWNBASE,color='tab:red',s=100,
            label='Troughs')

# 添加标注
for t,p in zip(trough_locations[1::5],peak_locations[::3]):
    plt.text(df.date[p],df.value[p]+15,df.date[p],
             horizontalalignment='center',color='darkgreen')
    plt.text(df.date[t],df.value[t]-35,df.date[t],
             horizontalalignment='center',color='darkred')

# 图表修饰
plt.ylim(50,750)
xtick_location=df.index.tolist()[::6]
xtick_labels=df.date.tolist()[::6]
plt.xticks(ticks=xtick_location,labels=xtick_labels,rotation=90,
           fontsize=12,alpha=.7)
plt.title("Peak and Troughs of Air Passengers Traffic (1949-1969)",
          fontsize=18)
plt.yticks(fontsize=12,alpha=.7)

# 美化边框
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)

# 添加图例、网格和显示图表
plt.legend(loc='upper left')
plt.grid(axis='y',alpha=.3)
plt.show()






























#【例9-3】绘制折线图，并将时间序列分解为趋势、季节和残差分量。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse

# 导入数据
df=pd.read_csv('S:\Desktop_new\caxbook_python\python_plot_202405\PyData\AirPassengers.csv')
dates=pd.DatetimeIndex([parse(d).strftime('%Y-%m-01')for d in df['date']])
df.set_index(dates,inplace=True)

# 分解时间序列
result=seasonal_decompose(df['value'],model='multiplicative')

# 绘图
plt.rcParams.update({'figure.figsize':(10,8)})
result.plot().suptitle('Time Series Decomposition of Air Passengers')
plt.show()
































#【例9-4】在同一图表上绘制多条折线图，表征多个时间序列。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('S:\Desktop_new\caxbook_python\python_plot_202405\PyData\mortality.csv')	# 导入数据

# 定义Y轴的上限、下限、间隔和颜色
y_LL=100  									# Y轴的下限
y_UL=int(df.iloc[:,1:].max().max()*1.1)	# Y轴的上限，取数据中最大值的1.1倍
y_interval=400  							# Y轴刻度的间隔
mycolors=['tab:red','tab:blue','tab:green','tab:orange']	# 折线颜色

fig,ax=plt.subplots(1,1,figsize=(10,6),dpi=80)			# 创建图表

# 遍历每列数据，绘制折线图并添加标签
columns=df.columns[1:]
for i,column in enumerate(columns):   
    plt.plot(df.date.values,df[column].values,lw=1.5,
             color=mycolors[i])					# 绘制折线图  

# 绘制刻度线  
for y in range(y_LL,y_UL,y_interval):   
    plt.hlines(y,xmin=0,xmax=71,colors='black',alpha=0.3,
               linestyles="--",lw=0.5)		# 绘制水平线
# 图表修饰    
plt.tick_params(axis="both",which="both",bottom=False,top=False,
                labelbottom=True,left=False,
                right=False,labelleft=True)	# 设置刻度线参数 
# 美化边框
plt.gca().spines["top"].set_alpha(.3)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.3)
plt.gca().spines["left"].set_alpha(.3)

plt.title('Number of Deaths from Lung Diseases in the UK (1974-1979)',
          fontsize=16)						# 添加标题
plt.yticks(range(y_LL,y_UL,y_interval),
           [str(y)for y in range(y_LL,y_UL,y_interval)],
           fontsize=12)						# 设置Y轴刻度及标签
plt.xticks(range(0,df.shape[0],12),df.date.values[::12],
           horizontalalignment='left',fontsize=12)	# 设置X轴刻度及标签
plt.ylim(y_LL,y_UL)							# 设置Y轴范围
plt.xlim(-2,80)								# 设置X轴范围
plt.show()






























#【例9-5】数据集中每个时间点（日期/时间戳）有多个观测值，请计算95%置信区间，并试构建带有误差带的折线图（时间序列）。输入代码如下：
from scipy.stats import sem					# 导入sem函数，用于计算标准误差
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据
df_raw=pd.read_csv('S:\Desktop_new\caxbook_python\python_plot_202405\PyData\orders_45d.csv',
parse_dates=['purchase_time','purchase_date'])

# 准备数据：每日订单数量的平均值和标准误差带
df_mean=df_raw.groupby('purchase_date').quantity.mean()# 计算每日订单数量的均值
df_se=df_raw.groupby('purchase_date').quantity.apply(sem).mul(1.96)# 计算每日订单数量的标准误差，并乘以1.96得到95%置信区间

# 绘图
plt.figure(figsize=(12,6),dpi=300)
plt.ylabel("# Daily Orders",fontsize=16)
x=[d.date().strftime('%Y-%m-%d')for d in df_mean.index]# 提取每日日期并转换为字符串格式
plt.plot(x,df_mean,color="white",lw=2)		# 绘制每日订单数量的折线图
plt.fill_between(x,df_mean-df_se,df_mean+df_se,
            		color="#3F5D7D")			# 填充95%置信区间

# 图表修饰
# 美化边框
plt.gca().spines["top"].set_alpha(0)
plt.gca().spines["bottom"].set_alpha(1)
plt.gca().spines["right"].set_alpha(0)
plt.gca().spines["left"].set_alpha(1)
plt.xticks(x[::6],[str(d)for d in x[::6]],
          fontsize=12)			# 设置X轴刻度及标签
plt.title("Daily Order Quantity with Error Bands (95% confidence)"
          ,fontsize=16)

# 坐标轴限制
s,e=plt.gca().get_xlim()		# 获取X轴的起始值和结束值
plt.xlim(s,e-2,)				# 设置X轴范围
plt.ylim(4,10)				# 设置Y轴范围

# 绘制水平刻度线  
for y in range(5,10,1):   
    plt.hlines(y,xmin=s,xmax=e,colors='black',
                alpha=0.5,linestyles="--",lw=0.5)		# 绘制水平虚线
plt.show()
































#【例9-6】创建一个包含三个子图的布局，用于可视化随机信号数据的不同视图。输入代码如下：
import time
import matplotlib.pyplot as plt
import numpy as np

# 创建一个 3x1 的子图布局
fig,axes=plt.subplots(nrows=3,figsize=(6,8),layout='constrained')

np.random.seed(19781101)			# 固定随机种子，以便结果可复现
# 生成一些数据; 一维随机游走+微小部分正弦波
num_series=1000
num_points=100
SNR=0.10						# 信噪比
x=np.linspace(0,4*np.pi,num_points)
# 生成无偏高斯随机游走
Y=np.cumsum(np.random.randn(num_series,num_points),axis=-1)
# 生成正弦信号
num_signal=round(SNR*num_series)
phi=(np.pi/8)*np.random.randn(num_signal,1)		# 小的随机偏移
Y[-num_signal:]=(np.sqrt(np.arange(num_points))	# 随机游走的 RMS 缩放因子
                 *(np.sin(x-phi)
      +0.05*np.random.randn(num_signal,num_points))	# 小的随机噪声
)

# 使用`plot`绘制系列，并使用小值的`alpha`
# 因为有太多重叠的系列在该视图中很难观察到正弦行为
tic=time.time()
axes[0].plot(x,Y.T,color="C0",alpha=0.1)
toc=time.time()
axes[0].set_title("Line plot with alpha")
print(f"{toc-tic:.3f} sec. elapsed")


# 将多个时间序列转换为直方图。不仅隐藏的信号更容易看到，而且这是一个更快的过程。
tic=time.time()
# 在每个时间序列中的点之间进行线性插值
num_fine=800
x_fine=np.linspace(x.min(),x.max(),num_fine)
y_fine=np.concatenate([np.interp(x_fine,x,y_row)for y_row in Y])
x_fine=np.broadcast_to(x_fine,(num_series,num_fine)).ravel()


# 使用对数颜色标度在2D直方图中绘制(x,y)点，可以看出，噪声下存在某种结构
# 调整 vmax 使信号更可见
cmap=plt.colormaps["plasma"]
cmap=cmap.with_extremes(bad=cmap(0))
h,xedges,yedges=np.histogram2d(x_fine,y_fine,bins=[400,100])
pcm=axes[1].pcolormesh(xedges,yedges,h.T,cmap=cmap,
						  norm="log",vmax=1.5e2,rasterized=True)
fig.colorbar(pcm,ax=axes[1],label="# points",pad=0)
axes[1].set_title("2d histogram and log color scale")

# 线性颜色标度下的相同数据
pcm=axes[2].pcolormesh(xedges,yedges,h.T,cmap=cmap,
						  vmax=1.5e2,rasterized=True)
fig.colorbar(pcm,ax=axes[2],label="# points",pad=0)
axes[2].set_title("2d histogram and linear color scale")

toc=time.time()
print(f"{toc-tic:.3f} sec. elapsed")
plt.show()




























#【例9-7】当在同一时间点测量两个不同数量的两个时间序列时，可以在右侧的辅助Y轴上再绘制第2个系列，即绘制多Y轴图。
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("S:\Desktop_new\caxbook_python\python_plot_202405\PyData\economics.csv")# 导入数据

x=df['date']
y1=df['psavert']
y2=df['unemploy']

# 绘制线条1（左Y轴）
fig,ax1=plt.subplots(1,1,figsize=(14,6),dpi=300)
ax1.plot(x,y1,color='tab:red')

# 绘制线条2（右Y轴）
ax2=ax1.twinx()	# 实例化一个共享相同x轴的第2个坐标轴
ax2.plot(x,y2,color='tab:blue')

# 图表修饰
# ax1（左Y轴）
ax1.set_xlabel('Year',fontsize=20)
ax1.tick_params(axis='x',rotation=0,labelsize=12)
ax1.set_ylabel('Personal Savings Rate',color='tab:red',fontsize=20)
ax1.tick_params(axis='y',rotation=0,labelcolor='tab:red' )
ax1.grid(alpha=.4)

# ax2（右Y轴）
ax2.set_ylabel("# Unemployed (1000's)",color='tab:blue',fontsize=20)
ax2.tick_params(axis='y',labelcolor='tab:blue')
ax2.set_xticks(np.arange(0,len(x),60))
ax2.set_xticklabels(x[::60],rotation=90,fontdict={'fontsize':10})
ax2.set_title("Personal Savings Rate vs Unemployed",
              fontsize=22)
fig.tight_layout()
plt.show()




























# 9.2 K线图
#     K线图是一种用于展示金融市场价格走势的图表，主要用于股票、期货、外汇等金融市
# 场。它由一系列矩形盒子（称为“K线”）组成，每个矩形盒子代表一段时间内的价格变动情
# 况，通常包括开盘价、收盘价、最高价和最低价。K线图的构成部分有：
#     （1）实体：表示开盘价和收盘价之间的价格区间。如果收盘价高于开盘价，通常使用
# 填充实体或者颜色填充表示上涨，反之表示下跌。
#     （2）上影线（上影线）：表示最高价和实体上端之间的价格区间。
#     （3）下影线（下影线）：表示实体下端和最低价之间的价格区间。
#     K线图能够提供关于价格走势、市场情绪和交易活动的重要信息，包括支撑阻力位、趋
# 势方向、买卖信号等。在金融分析中，K线图是一种常见的技术分析工具，被广泛应用于制
# 定交易策略和预测价格走势。
#     在Python 中，使用Plotly 可以绘制K线图，此时需要创建一个go.Candlestick对象，并
# 将其添加到图表中。


#【例9-8】K线图绘制示例1。输入代码如下：
import plotly.graph_objects as go
import pandas as pd

# 使用Pandas的read_csv函数从CSV文件中读取数据，并存储在DataFrame对象df中
df=pd.read_csv(r'S:\Desktop_new\caxbook_python\python_plot_202405\PyData\finance-charts-apple.csv')

# 使用Plotly的Candlestick对象go.Candlestick创建K线图
fig=go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['AAPL.Open'],high=df['AAPL.High'],
                low=df['AAPL.Low'],close=df['AAPL.Close'])])
fig.show()

# 隐藏x轴上的滚动窗口
fig.update_layout(xaxis_rangeslider_visible=False)
fig.show()

# 更新布局，添加标题、y轴标题、形状和注释
fig.update_layout(title='The Great Recession',yaxis_title='AAPL Stock',
    shapes=[dict(
        x0='2016-12-09',x1='2016-12-09',y0=0,y1=1,
        xref='x',yref='paper',line_width=2)],
    annotations=[dict(		# 添加注释，说明某个时间点
        x='2016-12-09',y=0.05,xref='x',yref='paper',
        showarrow=False,xanchor='left',text='Increase Period Begins')])
fig.show()






print('8')

























# 9.3 子弹图
#     子弹图（Bullet Chart）是一种用于显示单个指标在与目标值、良好和不良范围的比较中
# 的表现的图表类型。它可以用于直观地展示一个度量值在一个或多个维度上的表现，通常用
# 于业务绩效指标的可视化分析。子弹图通常包含以下几个元素：
#     （1）目标值线：表示目标值的水平线，用于显示期望的性能水平。
#     （2）实际值条：表示实际的度量值，通常以矩形条的形式显示。
#     （3）良好范围区域：表示良好性能的区域，通常以较浅的颜色或阴影标识。
#     （4）不良范围区域：表示不良性能的区域，通常以较深的颜色或阴影标识。
#     子弹图的优点在于能够清晰地显示实际值与目标值之间的差距，以及实际值在良好和不
# 良范围内的位置，从而帮助用户快速判断表现情况并进行比较分析。在Python中，可以使用
# Plotly或其他数据可视化库来绘制子弹图。


#【例9-9】创建三个不同风格的子弹图，分别用于展示利润指标，每个子弹图通过不同的设置来强调特定的信息和视觉效果。输入代码如下：
import plotly.graph_objects as go

# 子弹图1：简单的子弹图，只显示了实际值和参考值之间的比较。
fig=go.Figure(go.Indicator(
    mode="number+gauge+delta",			# 指示器模式，包括数字、仪表盘和增减值
    gauge={'shape':"bullet"},			# 设置子弹图的形状为bullet
    value=220,							# 实际值
    delta={'reference':300},			# 增减值的参考值
    domain={'x':[0,1],'y':[0,1]},		# 子弹图所占的区域
    title={'text':"Profit"}))			# 子弹图的标题

fig.update_layout(height=250)			# 更新布局，设置图表的高度
fig.show()

# 子弹图2：增加阈值和颜色阶梯，显示更详细的信息，包括颜色的变化表示不同的区间范围
fig=go.Figure(go.Indicator(
    mode="number+gauge+delta",value=220,	# 实际值
    domain={'x':[0.1,1],'y':[0,1]},			# 子弹图所占的区域
    title={'text':"<b>Profit</b>"},			# 子弹图的标题
    delta={'reference':200},				# 增减值的参考值
    gauge={'shape':"bullet",				# 设置子弹图的形状为bullet
         'axis':{'range':[None,300]},		# 指示器轴的范围
         'threshold':{'line':{'color':"red",'width':2},		# 阈值线的样式
					  'thickness':0.75,'value':280},			# 阈值的样式和值
         'steps':[{'range':[0,150],'color':"lightgray"},	# 不同范围的颜色
                  {'range':[150,250],'color':"gray"}]}))
fig.update_layout(height=250)				# 更新布局，设置图表的高度
fig.show()





























# 9.4 仪表图
#     仪表图（Gauge chart）是一种用于展示单一指标或数值的图表类型，通常用于显示目标
# 值与实际值之间的比较。它们类似于汽车仪表板上的速度计或油量计，因此也常被称为仪表
# 板图。仪表图通常由一个圆形或半圆形的指示器和刻度盘组成，指示器的位置代表指标的值，
# 而刻度盘上的刻度则表示了该指标的范围。
#     在数据可视化中，仪表图通常用来展示某个指标的当前值，以及该值与理想目标值或预
# 期范围之间的关系。它们在监控关键性能指标、比较实际和目标数值、评估进度等方面非常
# 有用。
#     仪表图的设计旨在引人注目并直观地传达信息，因此在创建时需要考虑美学和易读性。
# 虽然仪表图在一些情况下可能会过于炫目或不够准确，但在适当的情况下，它们可以是非常
# 有用的工具，帮助人们迅速了解关键指标的状态和趋势。

#【例9-11】使用Plotly库创建仪表图。输入代码如下：
import plotly.graph_objects as go

# 示例 1:创建一个简单的仪表图
fig=go.Figure(go.Indicator(
    mode="gauge+number",				# 模式设置为仪表盘模式并显示数值
    value=270,							# 设定指示器的数值为270
    domain={'x':[0,1],'y':[0,1]},		# 指示器的位置占据整个图表空间
    title={'text':"Speed"}))			# 指示器的标题为"Speed"
fig.show()

# 示例 2:创建一个带有增量和阈值的仪表图
fig=go.Figure(go.Indicator(
    domain={'x':[0,1],'y':[0,1]},		# 指示器的位置占据整个图表空间
    value=450,							# 设定指示器的数值为450
    mode="gauge+number+delta",			# 模式设置为仪表盘模式、显示数值和增量
    title={'text':"Speed"},			# 指示器的标题为"Speed"
    delta={'reference':380},			# 增量设置为380
    gauge={
        'axis':{'range':[None,500]},		# 指示器轴范围设定为0到500
        'steps' :[			# 阶梯设置，将范围划分为两段，分别设定为灰色和深灰色
            {'range':[0,250],'color':"lightgray"},
            {'range':[250,400],'color':"gray"}],
        'threshold' :{'line':{'color':"red",'width':4},
             'thickness':0.75,'value':490}  	# 设定阈值，超过阈值时显示红色
    }))
fig.show()

# 示例 3:创建一个带有增量和阈值的仪表图，样式定制更多
fig=go.Figure(go.Indicator(
    mode="gauge+number+delta",			# 模式设置为仪表盘模式、显示数值和增量
    value=420,							# 设定指示器的数值为420
    domain={'x':[0,1],'y':[0,1]},		# 指示器的位置占据整个图表空间
    title={'text':"Speed",'font':{'size':24}},		# 指示器标题，字体大小
    delta={'reference':400,'increasing':{'color':"RebeccaPurple"}},
     										# 增量设置为400，且增大时显示紫色
    gauge={
        'axis':{'range':[None,500],'tickwidth':1,
                'tickcolor':"darkblue"},		# 指示器轴范围设定，设置刻度宽度和颜色
        'bar':{'color':"darkblue"},			# 指示器条颜色设定为深蓝色
        'bgcolor':"white",					# 背景色设定为白色
        'borderwidth':2,					# 边框宽度设定为2
        'bordercolor':"gray",				# 边框颜色设定为灰色
        'steps':[    			# 阶梯设置，将范围划分为两段，分别设定为青色和皇家蓝
            {'range':[0,250],'color':'cyan'},
            {'range':[250,400],'color':'royalblue'}],
        'threshold':{		# 设定阈值为490，超过阈值时指示器显示红色
            'line':{'color':"red",'width':4},
            'thickness':0.75,'value':490 }}))

# 更新图表布局，设置背景色和字体颜色
fig.update_layout(paper_bgcolor="lavender",
                 font={'color':"darkblue",'family':"Arial"})
fig.show()

















print('11')











# 9.5 面积图
#     面积图（Area Chart）类似于折线图，也用于显示随时间、顺序或其他连续变量变化的趋
# 势和模式。与折线图不同，面积图通过填充折线下的区域来强调数据的相对大小和累积值。
#     面积图的优点是能够清晰地展示变量随时间或顺序的变化趋势，并突出显示数据的相对
# 大小和累积值。它常用于比较不同组的趋势、展示时间序列数据的变化情况以及观察数据的
# 累积效果。
#     针对时间序列数据，通过对轴和线之间的区域进行着色，面积图不仅强调峰和谷，而且
# 还强调高点和低点的持续时间。高点持续时间越长，线下面积越大。

#【例9-12】绘制面积图。输入代码如下：
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据，并解析'date'列为日期类型，然后选取前100行
df=pd.read_csv(r"S:\Desktop_new\caxbook_python\python_plot_202405\PyData\economics.csv",
                parse_dates=['date']).head(100)
x=np.arange(df.shape[0])					# 生成x轴的数据，从0到df的行数-1

# 计算月度储蓄率的变化率（回报率）
y_returns=(df.psavert.diff().fillna(0)/
             df.psavert.shift(1)).fillna(0)*100

plt.figure(figsize=(10,6),dpi=300)			# 绘图

# 使用fill_between函数填充正值区域为绿色，负值区域为红色
plt.fill_between(x[1:],y_returns[1:],0,where=y_returns[1:]>=0,
                facecolor='green',interpolate=True,alpha=0.7)
plt.fill_between(x[1:],y_returns[1:],0,where=y_returns[1:]<=0,
                facecolor='red',interpolate=True,alpha=0.7)

# 添加注释
plt.annotate('Peak \n1975',xy=(94.0,21.0),xytext=(88.0,28),
             bbox=dict(boxstyle='square',fc='firebrick'),
             arrowprops=dict(facecolor='steelblue',shrink=0.05),
             fontsize=15,color='white')

# 图形修饰
# 设置x轴刻度值为日期的月份和年份的简写
xtickvals=[str(m)[:3].upper()+"-"+str(y)for y,
             m in zip(df.date.dt.year,df.date.dt.month_name())]
plt.gca().set_xticks(x[::6])
plt.gca().set_xticklabels(xtickvals[::6],rotation=90,
						   fontdict={'horizontalalignment':'center',
						             'verticalalignment':'center_baseline'})
plt.ylim(-35,35)									# 设置y轴范围
plt.xlim(1,100)									# 设置x轴范围
plt.title("Month Economics Return %",fontsize=22)	# 设置标题
plt.ylabel('Monthly returns %')					# 设置y轴标签
plt.grid(alpha=0.5)									# 添加网格线
plt.show()


















print('12')














#【例9-13】绘制堆叠面积图，展示澳大利亚各地区夜间游客数量随时间的变化。堆叠面积图可以直观地显示多个时间序列的贡献程度，因此很容易相互比较。输入代码如下：
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据
df=pd.read_csv(r'S:\Desktop_new\caxbook_python\python_plot_202405\PyData\nightvisitors.csv')

# 决定颜色
mycolors=['tab:red','tab:blue','tab:green','tab:orange','tab:brown',
            'tab:grey','tab:pink','tab:olive']

# 绘制图表并添加标注
fig,ax=plt.subplots(1,1,figsize=(10,6),dpi=80)
columns=df.columns[1:]
labs=columns.values.tolist()

# 准备数据
x=df['yearmon'].values.tolist()
y0=df[columns[0]].values.tolist()
y1=df[columns[1]].values.tolist()
y2=df[columns[2]].values.tolist()
y3=df[columns[3]].values.tolist()
y4=df[columns[4]].values.tolist()
y5=df[columns[5]].values.tolist()
y6=df[columns[6]].values.tolist()
y7=df[columns[7]].values.tolist()
y=np.vstack([y0,y2,y4,y6,y7,y5,y1,y3])

# 绘制每一列的堆叠区域图
labs=columns.values.tolist()
ax=plt.gca()
ax.stackplot(x,y,labels=labs,colors=mycolors,alpha=0.8)

# 修饰图表
ax.set_title('Night Visitors in Australian Regions',fontsize=18)
ax.set(ylim=[0,100000])
ax.legend(fontsize=10,ncol=4)
plt.xticks(x[::5],fontsize=10,horizontalalignment='center')
plt.yticks(np.arange(10000,100000,20000),fontsize=10)
plt.xlim(x[0],x[-1])

# 柔化边界
plt.gca().spines["top"].set_alpha(0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(0)
plt.gca().spines["left"].set_alpha(.3)
plt.show()






























#【例9-14】绘制未堆叠面积图，可视化两个或更多个系列相对于彼此的进度（起伏）。输入代码如下：
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("S:\Desktop_new\caxbook_python\python_plot_202405\PyData\economics.csv")	# 导入数据
# 准备数据
x=df['date'].values.tolist()
y1=df['psavert'].values.tolist()
y2=df['uempmed'].values.tolist()
mycolors=['tab:red','tab:blue','tab:green','tab:orange',
            'tab:brown','tab:grey','tab:pink','tab:olive']
columns=['psavert','uempmed']

# 绘制图表
fig,ax=plt.subplots(1,1,figsize=(10,6),dpi=300)
ax.fill_between(x,y1=y1,y2=0,label=columns[1],alpha=0.5,
                color=mycolors[1],linewidth=2)
ax.fill_between(x,y1=y2,y2=0,label=columns[0],alpha=0.5,
                color=mycolors[0],linewidth=2)

# 修饰图表
ax.set_title('Personal Savings Rate vs Median Duration of Unemployment',
             fontsize=16)
ax.set(ylim=[0,30])
ax.legend(loc='best',fontsize=12)
plt.xticks(x[::50],fontsize=10,horizontalalignment='center')
plt.yticks(np.arange(2.5,30.0,2.5),fontsize=10)
plt.xlim(-10,x[-1])

# 绘制刻度线
for y in np.arange(2.5,30.0,2.5):   
    plt.hlines(y,xmin=0,xmax=len(x),colors='black',alpha=0.3,
               linestyles="--",lw=0.5)

# 柔化边界
plt.gca().spines["top"].set_alpha(0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(0)
plt.gca().spines["left"].set_alpha(.3)
plt.show()

































#【例9-15】利用altair库绘制分面面积图，展示不同股票的价格变化情况。输入代码如下：
import altair as alt
from vega_datasets import data

source=data.stocks()
alt.Chart(source).transform_filter(
    alt.datum.symbol != "GOOG",
).mark_area().encode(
    x="date:T",
    y="price:Q",
    color="symbol:N",
    row=alt.Row("symbol:N",sort=["MSFT","AAPL","IBM","AMZN"]),
).properties(height=50,width=400)





























# 9.6 日历图
#     日历图（Calendar Chart）是一种用于显示时间数据在一年中的分布和趋势。它以日历的
# 形式呈现数据，将每个日期表示为一个方格或单元格，并通过单元格的颜色或填充来表示该
# 日期的特定指标或数值。
#     日历图的优点是能够以直观的方式展示时间数据的分布和趋势，尤其适用于数据的季节
# 性或周期性变化的观察。它常用于表示每天的销售额、气温、疾病发病率等与日期相关的数
# 据。


#【例9-16】使用calmap库创建一个日历热图，展示时间序列数据的变化趋势。输入代码如下：
import pandas as pd
import matplotlib.pyplot as plt
import calmap

# 读取包含日期和值的CSV文件，解析日期列
df=pd.read_csv('S:\Desktop_new\caxbook_python\python_plot_202405\PyData\Calendar.csv',parse_dates=['date'])
df.set_index('date',inplace=True)

# 创建日历热图
# fillcolor: 填充颜色；linecolor: 边框颜色；linewidth: 边框线宽；cmap: 颜色映射
# yearlabel_kws: 年份标签的样式；fig_kws:图形参数，包括figsize和dpi
fig,ax=calmap.calendarplot(df['value'],fillcolor='grey',
                              linecolor='w',linewidth=0.1,cmap='RdYlGn',
                              yearlabel_kws={'color': 'black','fontsize': 12},
                              fig_kws=dict(figsize=(14,8),dpi=300))
# 添加颜色条
fig.colorbar(ax[0].get_children()[1],ax=ax.ravel().tolist())
plt.show()