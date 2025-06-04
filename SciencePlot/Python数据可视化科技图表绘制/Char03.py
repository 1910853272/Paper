#【例3-1】利用axes()函数可以在一幅图中生成多个坐标图形（axes）
import matplotlib.pyplot as plt

plt.figure()
plt.axes([0.0,0.0,1,1])
plt.axes([0.1,0.1,.5,.5],facecolor='blue')
plt.axes([0.2,0.2,.5,.5],facecolor='pink')
plt.axes([0.3,0.3,.5,.5],facecolor='green')
plt.axes([0.4,0.4,.5,.5],facecolor='skyblue')
plt.show()

























#【例3-2】图3-1可以通过一下代码创建，通过这段代码可以了解Python的作图流程。
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke
from matplotlib.ticker import AutoMinorLocator,MultipleLocator

royal_blue=[0,20/256,82/256]		# 自定义的颜色
# 创建图形
np.random.seed(19781101)			# 固定随机种子，以便结果可复现

# 生成数据
X=np.linspace(0.5,3.5,100)			# 生成等间隔的X值
Y1=3+np.cos(X)						# 第一组数据，基于余弦函数
Y2=1+np.cos(1+X/0.75)/2				# 第二组数据，变化的余弦函数
Y3=np.random.uniform(Y1,Y2,len(X))	# 第三组数据，Y1与Y2之间的随机数

# 创建并配置图形和轴
fig=plt.figure(figsize=(7.5,7.5))	# 创建图形，指定大小
ax=fig.add_axes([0.2,0.17,0.68,0.7],aspect=1)			# 添加轴，设置宽高比

# 设置主要和次要刻度定位器
ax.xaxis.set_major_locator(MultipleLocator(1.000))		# X轴的主要刻度间隔
ax.xaxis.set_minor_locator(AutoMinorLocator(4))		# X轴的次要刻度间隔
ax.yaxis.set_major_locator(MultipleLocator(1.000))		# Y轴的主要刻度间隔
ax.yaxis.set_minor_locator(AutoMinorLocator(4))		# Y轴的次要刻度间隔
ax.xaxis.set_minor_formatter("{x:.2f}")				# 设置次要刻度的格式

# 设置坐标轴的显示范围
ax.set_xlim(0,4)
ax.set_ylim(0,4)

# 配置刻度标签的样式
ax.tick_params(which='major',width=1.0,length=10,labelsize=14)	# 主刻度
ax.tick_params(which='minor',width=1.0,length=5,
	                 labelsize=10,labelcolor='0.25')						# 次刻度

# 添加网格
ax.grid(linestyle="--",linewidth=0.5,
          color='.25',zorder=-10)			# 设置网格样式和图层顺序

# 绘制数据
ax.plot(X,Y1,c='C0',lw=2.5,label="Blue signal",
					  zorder=10)				# 绘制第一组数据，设置图层顺序
ax.plot(X,Y2,c='C1',lw=2.5,label="Orange signal")		# 绘制第二组数据
# 绘制第三组数据作为散点图
ax.plot(X[::3],Y3[::3],linewidth=0,markersize=9,
        marker='s',markerfacecolor='none',markeredgecolor='C4',
        markeredgewidth=2.5)

# 设置标题和轴标签
ax.set_title("Anatomy of a figure",fontsize=20,verticalalignment='bottom')
ax.set_xlabel("x Axis label",fontsize=14)
ax.set_ylabel("y Axis label",fontsize=14)
ax.legend(loc="upper right",fontsize=14)	# 添加图例

# 标注图形
def annotate(x,y,text,code):
# 添加圆形标记
    c=Circle((x,y),radius=0.15,clip_on=False,zorder=10,linewidth=2.5,
               edgecolor=royal_blue+[0.6],facecolor='none',
               path_effects=[withStroke(linewidth=7,foreground='white')])
# 使用路径效果突出标记
    ax.add_artist(c)

# 使用路径效果为文本添加背景
# 分别绘制路径效果和彩色文本，以避免路径效果裁剪其他文本
    for path_effects in [[withStroke(linewidth=7,foreground='white')],[]]:
        color='white' if path_effects else royal_blue
        ax.text(x,y-0.2,text,zorder=100,
                ha='center',va='top',weight='bold',color=color,
                style='italic',fontfamily='monospace',
                path_effects=path_effects)

        color='white' if path_effects else 'black'
        ax.text(x,y-0.33,code,zorder=100,
                ha='center',va='top',weight='normal',color=color,
                fontfamily='monospace',fontsize='medium',
                path_effects=path_effects)


# 通过调用自定义的annotate函数来添加多个图形标注
# 具体标注调用代码，每次调用都是标注图形的一个特定部分和相关的Matplotlib命令
annotate(3.5,-0.13,"Minor tick label","ax.xaxis.set_minor_formatter")
annotate(-0.03,1.0,"Major tick","ax.yaxis.set_major_locator")
annotate(0.00,3.75,"Minor tick","ax.yaxis.set_minor_locator")
annotate(-0.15,3.00,"Major tick label","ax.yaxis.set_major_formatter")
annotate(1.68,-0.39,"xlabel","ax.set_xlabel")
annotate(-0.38,1.67,"ylabel","ax.set_ylabel")
annotate(1.52,4.15,"Title","ax.set_title")
annotate(1.75,2.80,"Line","ax.plot")
annotate(2.25,1.54,"Markers","ax.scatter")
annotate(3.00,3.00,"Grid","ax.grid")
annotate(3.60,3.58,"Legend","ax.legend")
annotate(2.5,0.55,"Axes","fig.subplots")
annotate(4,4.5,"Figure","plt.figure")
annotate(0.65,0.01,"x Axis","ax.xaxis")
annotate(0,0.36,"y Axis","ax.yaxis")
annotate(4.0,0.7,"Spine","ax.spines")
# 给图形周围添加边框
fig.patch.set(linewidth=4,edgecolor='0.5')
plt.show()						# 显示图形



































#【例3-3】Matplotlib创建图形示例。
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 加载 iris 数据集
iris=load_iris()
data=iris.data
target=iris.target

# 提取数据
sepal_length=data[:,0]
petal_length=data[:,2]

# 创建图形和子图
fig,axs=plt.subplots(1,2,figsize=(10,5))	# 创建包含两个子图的图形
fig.suptitle('Sepal Length vs Petal Length',fontsize=16)		# 设置图形标题

# 第1个子图：线图
axs[0].plot(sepal_length,label='Sepal Length',color='blue',
             linestyle='-')				# 绘制线图
axs[0].plot(petal_length,label='Petal Length',color='green',
             linestyle='--')				# 绘制另一个线图
axs[0].set_xlabel('Sample')				# 设置x轴标签
axs[0].set_ylabel('Length')				# 设置y轴标签
axs[0].legend()							# 添加图例
axs[0].grid(True)						# 添加网格线

# 第2个子图：散点图
scatter=axs[1].scatter(sepal_length,petal_length,c=target,
						       cmap='viridis',label='Data Points')	# 绘制散点图
axs[1].set_xlabel('Sepal Length')		# 设置x轴标签
axs[1].set_ylabel('Petal Length')		# 设置y轴标签
axs[1].legend()							# 添加图例
axs[1].grid(True)						# 添加网格线
fig.colorbar(scatter,ax=axs[1],label='Species')				# 添加颜色条

plt.tight_layout()						# 自动调整子图布局
plt.show()		# 显示图形






































#【例3-4】使用iris数据集作为示例，演示使用.plt.subplot()函数创建子图。每个子图展示iris数据集中一个特征的直方图。
# 安装和导入必要的库：
import matplotlib.pyplot as plt
import seaborn as sns				# seaborn 库内置了iris数据集

# 加载iris数据集并查看其结构
iris=sns.load_dataset('iris')
iris.head()							# 输出略
plt.figure(figsize=(10,6))			# 设置画布大小

# 第1个子图
plt.subplot(2,2,1)					# 2行2列的第1个
plt.hist(iris['sepal_length'],color='blue')
plt.title('Sepal Length')
# 第2个子图
plt.subplot(2,2,2)				# 2行2列的第2个
plt.hist(iris['sepal_width'],color='orange')
plt.title('Sepal Width')
# 第3个子图
plt.subplot(2,2,3)				# 2行2列的第3个
plt.hist(iris['petal_length'],color='green')
plt.title('Petal Length')
# 第4个子图
plt.subplot(2,2,4)				# 2行2列的第4个
plt.hist(iris['petal_width'],color='red')
plt.title('Petal Width')

plt.tight_layout()				# 自动调整子图间距
plt.show()







































#【例3-5】使用iris数据集作为示例，演示使用.plt.subplots()函数创建子图。
import matplotlib.pyplot as plt
import seaborn as sns

data=sns.load_dataset("iris")				# 加载内置的iris数据集

# 使用plt.subplots()创建一个2行3列的子图布局
fig,axs=plt.subplots(2,3,figsize=(15,8))

# 第1个子图：绘制sepal_length和sepal_width的散点图
axs[0,0].scatter(data['sepal_length'],data['sepal_width'])
axs[0,0].set_title('Sepal Length vs Sepal Width')

# 第2个子图：绘制petal_length和petal_width的散点图
axs[0,1].scatter(data['petal_length'],data['petal_width'])
axs[0,1].set_title('Petal Length vs Petal Width')

# 第3个子图：绘制sepal_length的直方图
axs[0,2].hist(data['sepal_length'],bins=20)
axs[0,2].set_title('Sepal Length Distribution')

# 4个子图：绘制petal_length的直方图
axs[1,0].hist(data['petal_length'],bins=20)
axs[1,0].set_title('Petal Length Distribution')

# 第5和第6位置合并为一个大图，展示species的计数条形图
# 为了合并第二行的中间和最右侧位置，使用subplot2grid功能
plt.subplot2grid((2,3),(1,1),colspan=2)
sns.countplot(x='species',data=data)
plt.title('Species Count')

plt.tight_layout()			# 调整子图之间的间距
plt.show()







































#【例3-6】演示使用figure.add_subplot()函数创建子图。
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,4))		# 创建一个图形实例

# 添加第1个子图：1行2列的第1个位置
ax1=fig.add_subplot(1,2,1)
ax1.plot([1,2,3,4],[1,4,2,3])		# 绘制一条简单的折线图
ax1.set_title('First Subplot')

# 添加第2个子图：1行2列的第2个位置
ax2=fig.add_subplot(1,2,2)
ax2.bar([1,2,3,4],[10,20,15,25])	# 绘制一个条形图
ax2.set_title('Second Subplot')

# 显示图形
plt.tight_layout()					# 自动调整子图参数，使之填充整个图形区域
plt.show()







































#【例3-7】使用matplotlib自带的iris数据集作为示例，演示使用subplot2grid()函数创建子图。
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 载入鸢尾花数据集
iris=load_iris()
data=iris.data
target=iris.target
feature_names=iris.feature_names
target_names=iris.target_names

grid_size=(3,3)			# 定义网格大小为3x3

# 第1个子图占据位置 (0,0)
ax1=plt.subplot2grid(grid_size,(0,0),facecolor='orange')
ax1.scatter(data[:,0],data[:,1],c=target,cmap='viridis')
ax1.set_xlabel(feature_names[0])
ax1.set_ylabel(feature_names[1])

# 第2个子图占据位置(0,1)，并跨越2列
ax2=plt.subplot2grid(grid_size,(0,1),colspan=2,facecolor='pink')
ax2.scatter(data[:,1],data[:,2],c=target,cmap='viridis')
ax2.set_xlabel(feature_names[1])
ax2.set_ylabel(feature_names[2])

# 第3个子图占据位置(1,0)，并跨越2行
ax3=plt.subplot2grid(grid_size,(1,0),rowspan=2,facecolor='grey')
ax3.scatter(data[:,0],data[:,2],c=target,cmap='viridis')
ax3.set_xlabel(feature_names[0])
ax3.set_ylabel(feature_names[2])

# 第4个子图占据位置 (1,1)，并跨越到最后
ax4=plt.subplot2grid(grid_size,(1,1),colspan=2,
						     rowspan=2,facecolor='skyblue')
ax4.scatter(data[:,2],data[:,3],c=target,cmap='viridis')
ax4.set_xlabel(feature_names[2])
ax4.set_ylabel(feature_names[3])

plt.tight_layout()
plt.show()






































#【例3-8】使用iris数据集作为示例，演示使用gridspec.GridSpec()函数创建子图。
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_iris

# 载入Iris数据集
iris=load_iris()
data=iris.data
target=iris.target
feature_names=iris.feature_names
target_names=iris.target_names

# 创建一个2x2的子图网格
fig=plt.figure(figsize=(10,6))
gs=gridspec.GridSpec(2,2,height_ratios=[1,1],width_ratios=[1,1])

# 在网格中创建子图
ax1=plt.subplot(gs[0,0])
ax1.scatter(data[:,0],data[:,1],c=target,cmap='viridis')
ax1.set_xlabel(feature_names[0])
ax1.set_ylabel(feature_names[1])
ax1.set_title('Sepal Length vs Sepal Width')

ax2=plt.subplot(gs[0,1])
ax2.scatter(data[:,1],data[:,2],c=target,cmap='viridis')
ax2.set_xlabel(feature_names[1])
ax2.set_ylabel(feature_names[2])
ax2.set_title('Sepal Width vs Petal Length')

ax3=plt.subplot(gs[1,:])
ax3.scatter(data[:,2],data[:,3],c=target,cmap='viridis')
ax3.set_xlabel(feature_names[2])
ax3.set_ylabel(feature_names[3])
ax3.set_title('Petal Length vs Petal Width')

plt.tight_layout()				# 调整布局
plt.show()



































#【例3-9】使用iris数据集作为示例，演示图表元素的添加。
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

# 载入Iris数据集
iris=load_iris()
data=iris.data
target=iris.target
feature_names=iris.feature_names
target_names=iris.target_names

fig,ax=plt.subplots(figsize=(6,4))		# 创建图形和子图
# 绘制散点图
for i in range(len(target_names)):
    ax.scatter(data[target==i,0],data[target==i,1],label=target_names[i])
ax.set_title('Sepal Length vs Sepal Width',fontsize=16)	# 添加标题
ax.legend(fontsize=12)						# 添加图例
ax.grid(True,linestyle='--',alpha=0.5)	# 添加网格线

# 自定义坐标轴标签
ax.set_xlabel(feature_names[0],fontsize=14)
ax.set_ylabel(feature_names[1],fontsize=14)
# 设置坐标轴刻度标签大小
ax.tick_params(axis='both',which='major',labelsize=12)

plt.tight_layout()							# 调整图形边界
plt.show()







































#【例3-10】极坐标示例。
import matplotlib.pyplot as plt
import numpy as np

# 创建一些示例数据
theta=np.linspace(0,2*np.pi,100)
r=np.abs(np.sin(theta))

plt.figure(figsize=(6,6))
ax=plt.subplot(111,projection='polar')			# 创建极坐标系图形
ax.plot(theta,r,color='blue',linewidth=2)		# 绘制极坐标系图形
ax.set_title('Polar Plot',fontsize=16)			# 添加标题
plt.show()										# 显示图形





































#【例3-11】图表风格示例。
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

np.random.seed(19781101)			# 固定随机种子，以便结果可复现

def plot_scatter(ax,prng,nb_samples=100):
    """绘制散点图"""
    for mu,sigma,marker in [(-.5,0.75,'o'),(0.75,1.,'s')]:
        x,y=prng.normal(loc=mu,scale=sigma,size=(2,nb_samples))
        ax.plot(x,y,ls='none',marker=marker)
    ax.set_xlabel('X-label')
    ax.set_title('Axes title')
    return ax

def plot_colored_lines(ax):
    """绘制颜色循环线条"""
    t=np.linspace(-10,10,100)
    def sigmoid(t,t0):
        return 1/(1+np.exp(-(t-t0)))
    nb_colors=len(plt.rcParams['axes.prop_cycle'])
    shifts=np.linspace(-5,5,nb_colors)
    amplitudes=np.linspace(1,1.5,nb_colors)
    for t0,a in zip(shifts,amplitudes):
        ax.plot(t,a*sigmoid(t,t0),'-')
    ax.set_xlim(-10,10)
    return ax

def plot_bar_graphs(ax,prng,min_value=5,max_value=25,nb_samples=5):
    """绘制两个并排的柱状图。"""
    x=np.arange(nb_samples)
    ya,yb=prng.randint(min_value,max_value,size=(2,nb_samples))
    width=0.25
    ax.bar(x,ya,width)
    ax.bar(x+width,yb,width,color='C2')
    ax.set_xticks(x+width,labels=['a','b','c','d','e'])
    return ax

def plot_colored_circles(ax,prng,nb_samples=15):
    """绘制彩色圆形。"""
    for sty_dict,j in zip(plt.rcParams['axes.prop_cycle'](),
						    range(nb_samples)):
        ax.add_patch(plt.Circle(prng.normal(scale=3,size=2),
						         radius=1.0,color=sty_dict['color']))
    ax.grid(visible=True)
  	# 添加标题以启用网格
    plt.title('ax.grid(True)',family='monospace',fontsize='small')
    ax.set_xlim([-4,8])
    ax.set_ylim([-5,6])
    ax.set_aspect('equal',adjustable='box')			# 绘制圆形
    return ax

def plot_image_and_patch(ax,prng,size=(20,20)):
    """绘制图像和圆形补丁。"""
    values=prng.random_sample(size=size)
    ax.imshow(values,interpolation='none')
    c=plt.Circle((5,5),radius=5,label='patch')
    ax.add_patch(c)
    # 移除刻度
    ax.set_xticks([])
    ax.set_yticks([])

def plot_histograms(ax,prng,nb_samples=10000):
    """绘制四个直方图和一个文本注释。"""
    params=((10,10),(4,12),(50,12),(6,55))
    for a,b in params:
        values=prng.beta(a,b,size=nb_samples)
        ax.hist(values,histtype="stepfilled",bins=30,
                alpha=0.8,density=True)
    # 添加小注释。
    ax.annotate('Annotation',xy=(0.25,4.25),
           xytext=(0.9,0.9),textcoords=ax.transAxes,
           va="top",ha="right",
           bbox=dict(boxstyle="round",alpha=0.2),
           arrowprops=dict(arrowstyle="->",
	               connectionstyle="angle,angleA=-95,angleB=35,rad=10"),)
    return ax

def plot_figure(style_label=""):
    """设置并绘制具有给定样式的演示图。"""
    # 在不同的图之间使用专用的RandomState实例绘制相同的“随机”值
    prng=np.random.RandomState(96917002)
    # 创建具有特定样式的图和子图
    fig,axs=plt.subplots(ncols=6,nrows=1,num=style_label,
                 figsize=(14.8,2.8),layout='constrained')

    # 添加统一的标题，标题颜色与背景颜色相匹配
    background_color=mcolors.rgb_to_hsv(
        mcolors.to_rgb(plt.rcParams['figure.facecolor']))[2]
    if background_color<0.5:
        title_color=[0.8,0.8,1]
    else:
        title_color=np.array([19,6,84])/256
    fig.suptitle(style_label,x=0.01,ha='left',color=title_color,
                 fontsize=14,fontfamily='DejaVu Sans',fontweight='normal')
    plot_scatter(axs[0],prng)
    plot_image_and_patch(axs[1],prng)
    plot_bar_graphs(axs[2],prng)
    plot_colored_lines(axs[3])
    plot_histograms(axs[4],prng)
    plot_colored_circles(axs[5],prng)
  	# 添加分隔线
    rec=Rectangle((1+0.025,-2),0.05,16,
                    clip_on=False,color='gray')
    axs[4].add_artist(rec)

if __name__=="__main__":
  	# 获取所有可用的样式列表，按字母顺序排列
    style_list=['default','classic']+sorted(
        style for style in plt.style.available
        if style !='classic' and not style.startswith('_'))

    # 绘制每种样式的演示图
    for style_label in style_list:
        with plt.rc_context({"figure.max_open_warning":len(style_list)}):
            with plt.style.context(style_label):
                plot_figure(style_label=style_label)
    plt.show()




































#【例3-12】使用自带数据集利用Seaborn函数绘图。运行后输出结果如图所示。
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据集
iris=sns.load_dataset("iris",data_home='seaborn-data',cache=True)
tips=sns.load_dataset("tips",data_home='seaborn-data',cache=True)
car_crashes=sns.load_dataset("car_crashes",data_home='seaborn-data',cache=True)
penguins=sns.load_dataset("penguins",data_home='seaborn-data',cache=True)
diamonds=sns.load_dataset("diamonds",data_home='seaborn-data',cache=True)

plt.figure(figsize=(15,8))				# 设置画布
# 第1幅图：iris数据集的散点图
plt.subplot(2,3,1)
sns.scatterplot(x="sepal_length",y="sepal_width",hue="species",
                   data=iris)
plt.title("Iris scatterplot")

# 第2幅图：tips 数据集的箱线图
plt.subplot(2,3,2)
tips=sns.load_dataset("tips",data_home='seaborn-data',cache=True)
sns.boxplot(x="day",y="total_bill",hue="smoker",data=tips)
plt.title("Tips boxplot")

# 第3幅图：tips 数据集的小提琴图
plt.subplot(2,3,3)
sns.violinplot(x="day",y="total_bill",hue="smoker",data=tips)
plt.title("Tips violinplot")

# 第4幅图：car_crashes 数据集的直方图
plt.subplot(2,3,4)
sns.histplot(car_crashes['total'],bins=20)
plt.title("Car Crashes histplot")

# 第5幅图：penguins 数据集的点图
plt.subplot(2,3,5)
sns.pointplot(x="island",y="bill_length_mm",hue="species",data=penguins)
plt.title("Penguins pointplot")

# 第6幅图：diamonds 数据集的计数图
plt.subplot(2,3,6)
sns.countplot(x="cut",data=diamonds)
plt.title("Diamonds countplot")

plt.tight_layout()
plt.show()







































#【例3-13】通过Seaborn自带数据集演示图表风格的设置示例。
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")			# 设置图表风格为 darkgrid
iris=sns.load_dataset("iris")		# 加载 iris 数据集

# 绘制花瓣长度与宽度的散点图
sns.scatterplot(x="petal_length",y="petal_width",
                   hue="species",data=iris)
plt.title("Scatter Plot of Petal Length vs Petal Width")
plt.show()


































#【例3-14】通过Seaborn自带数据集演示颜色主题的设置。
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_palette("deep")				# 设置颜色主题为deep
tips=sns.load_dataset("tips")		# 加载 tips 数据集

# 绘制小费金额的小提琴图，按照就餐日期和吸烟者区分颜色
sns.violinplot(x="day",y="total_bill",hue="smoker",data=tips)
plt.title("Tips violinplot")		# 设置图表标题
plt.show()							# 显示图表



































#【例3-15】演示查看当前默认的颜色主题和一些常见颜色主题的颜色列表。
import seaborn as sns

# 查看当前默认的颜色主题
current_palette=sns.color_palette()
print("Current default color palette:",current_palette)

# 查看常见的颜色主题
print("\nSome common color palettes:")
print("Accent:",sns.color_palette("Accent"))
print("Blues:",sns.color_palette("Blues"))
print("Greens:",sns.color_palette("Greens"))
print("Reds:",sns.color_palette("Reds"))
print("Purples:",sns.color_palette("Purples"))
print("Oranges:",sns.color_palette("Oranges"))
print("YlGnBu:",sns.color_palette("YlGnBu"))












































#【例3-16】演示了使用图表分面绘制多个散点图。
import seaborn as sns
import matplotlib.pyplot as plt

iris=sns.load_dataset("iris")			# 加载iris数据集

# 创建 FacetGrid 对象，按照种类（'species'）进行分面
g=sns.FacetGrid(iris,col="species",margin_titles=True)
# 在每个子图中绘制花萼长度与花萼宽度的散点图
g.map(sns.scatterplot,"sepal_length","sepal_width")
g.set_axis_labels("Sepal Length","Sepal Width")		# 设置子图标题

plt.show()









































#【例3-17】使用penguins数据集演示利用plotnine绘制散点图。
from plotnine import *
import seaborn as sns

penguins=sns.load_dataset("penguins",data_home='seaborn-data',cache=True)	# 加载数据集

p=ggplot(penguins,aes(x='bill_length_mm',y='bill_depth_mm',
            color='species'))				# 创建画布和导入数据

p=p+geom_point(size=3,alpha=0.7)		# 添加几何对象图层-散点图，并进行美化

# 设置标度
p=p+scale_x_continuous(name='Length(mm)')
p=p+scale_y_continuous(name='Depth(mm)')

p=p+theme(legend_position='top',figure_size=(6,4))	# 设置主题和其他参数

# 显示图形并保存绘图
print(p)
ggsave(p,filename="penguins_plot.png",dpi=300)







































#【例3-18】使用自带数据集利用plotnine函数绘图。运行后输出结果如图所示。
from plotnine import *
from plotnine.data import *
import pandas as pd

# 散点图-mpg 数据集
p1=(ggplot(mpg)+
     aes(x='displ',y='hwy')+
     geom_point(color='blue')+
     labs(title='Displacement vs Highway MPG')+
     theme(plot_title=element_text(size=14,face='bold')))

# 箱线图-diamonds 数据集
p2=(ggplot(diamonds.sample(1000))+
     aes(x='cut',y='price',fill='cut')+
     geom_boxplot()+
     labs(title='Diamond Price by Cut')+
     scale_fill_brewer(type='qual',palette='Pastel1')+
     theme(plot_title=element_text(size=14,face='bold')))

# 直方图-msleep 数据集
p3=(ggplot(msleep)+
     aes(x='sleep_total')+
     geom_histogram(bins=20,fill='green',color='black')+
     labs(title='Total Sleep in Mammals')+
     theme(plot_title=element_text(size=14,face='bold')))

# 线图-economics 数据集
p4=(ggplot(economics)+
     aes(x='date',y='unemploy')+
     geom_line(color='red')+
     labs(title='Unemployment over Time')+
     theme(plot_title=element_text(size=14,face='bold')))

# 条形图-presidential 数据集
presidential['duration']=(presidential['end']-presidential['start']).dt.days
p5=(ggplot(presidential)+
     aes(x='name',y='duration',fill='name')+
     geom_bar(stat='identity')+
     labs(title='Presidential Terms Duration')+
     scale_fill_hue(s=0.90,l=0.65)+
     theme(axis_text_x=element_text(rotation=90,hjust=1),
           plot_title=element_text(size=14,face='bold')))

# 折线图-midwest 数据集
p6=(ggplot(midwest)+
     aes(x='area',y='popdensity')+
     geom_line(color='purple')+
     labs(title='Population Density vs Area')+
     theme(plot_title=element_text(size=14,face='bold')))

# 直接显示所有图形
plots=[p1,p2,p3,p4,p5,p6]
for plot in plots:
    print(plot)






































#【例3-19】使用自带数据集演示主题的定义。运行后输出结果如图所示。
from plotnine import *
from plotnine.data import mpg

# 创建散点图
p=(ggplot(mpg,aes(x='displ',y='hwy',color='displ'))+
     geom_point()+	# 添加点图层
     scale_color_gradient(low='blue',high='red')+		# 设置颜色渐变
     labs(title='Engine Displacement vs. Highway MPG',	# 设置图表标题
          x='Engine Displacement (L)',					# 设置x轴标题
          y='Miles per Gallon (Highway)')+				# 设置y轴标题
     theme_minimal()+									# 使用最小主题
     theme(axis_text_x=element_text(angle=45,hjust=1),	# 自定义x轴文字样式
           axis_text_y=element_text(color='darkgrey'),	# 自定义y轴文字样式
           plot_background=element_rect(fill='whitesmoke'),	# 自定义图表背景色
           panel_background=element_rect(fill='white',
                   color='black',size=0.5),		# 自定义面板背景和边框
           panel_grid_major=element_line(color='lightgrey'),
											  			# 自定义主要网格线颜色
           panel_grid_minor=element_line(color='lightgrey',linestyle='--'),
														# 自定义次要网格线样式
           legend_position='right',					# 设置图例位置
           figure_size=(8,6)))						# 设置图形大小

print(p)	# 显示图表











































#【例3-20】按照mpg数据集中的class变量进行分面，每种车型的数据都会显示在不同的面板中。
from plotnine import *
from plotnine.data import mpg

# 创建散点图并按照`class`变量进行分面，添加颜色渐变
p=(ggplot(mpg,aes(x='displ',y='hwy',color='displ'))+
     geom_point()+
     scale_color_gradient(low='blue',high='orange')+		# 添加颜色渐变
     facet_wrap('~class')+	# 按照汽车类型分面
     labs(title='Engine Displacement vs. Highway MPG by Vehicle Class',
          x='Engine Displacement (L)',
          y='Miles per Gallon (Highway)'))
print(p)










































#【例3-21】按照驱动类型分为不同的行，而每行内部将根据车辆类型进一步划分为不同的列。
from plotnine import *
from plotnine.data import mpg

# 创建散点图并按照class变量进行分面，根据drv变量映射颜色
p=(ggplot(mpg,aes(x='displ',y='hwy',color='drv'))+
     geom_point()+						# 添加点图层
     scale_color_brewer(type='qual',palette='Set1')+	# 使用定性的颜色方案
     facet_grid('drv ~ class')+			# 行是驱动类型，列是汽车类型
     labs(title='Engine Displacement vs. Highway MPG by Vehicle Class',
          x='Engine Displacement (L)',
          y='Miles per Gallon (Highway)')+
     theme_light()+ 						# 使用亮色主题
     theme(figure_size=(10,6),			# 调整图形大小
           strip_text_x=element_text(size=10,color='black',angle=0),
						            			# 自定义分面标签的样式
           legend_title=element_text(color='blue',size=10),
						            			# 自定义图例标题的样式
           legend_text=element_text(size=8),		# 自定义图例文本的样式
           legend_position='right'))	# 调整图例位置
print(p)