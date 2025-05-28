# 一种仿生的材料内模拟光电储池计算用于人体动作处理 南京大学-施毅

![iShot_2025-05-13_14.15.35](https://raw.githubusercontent.com/1910853272/image/master/img/202505131419773.png)

- **光电分级神经元**
   基于 MoS₂ 光电晶体管的“分级神经元”可在像素点直接感知运动速度，通过调节栅电压实现动态调节。
- **范德华异质结构阵列**
   利用层状材料在栅极上可调光电特性，实现对运动方向与速率的识别。
- **光子忆阻储池计算**
   将光学刺激与忆阻器结合构建“光子储池”，可完成运动模式识别与预测。
- **全忆阻运动检测器**
   采用纯忆阻器实现车道变换预测，兼顾高精度与低计算开销。

上述工作大多停留在简单运动模式，且缺乏在真实数据集上的验证，主要受限于：1.数据预处理复杂；2.器件物理特性难以匹配相应算法需求

## 仿生材料内模拟光电储池计算系统

**储池器件**：采用 InGaZnO（IGZO）光电突触晶体管，实现可调记忆衰减与非线性映射。

**读出层**：基于 TaOx 忆阻器构建的 1T1R 阵列，直接映射线性回归权重并完成矩阵乘加。

## 端到端动作识别

- **输入**：3D 骨骼关节点序列，无需任何手工特征提取。
- **编码**：多组高斯接收域（GRF）神经元将骨骼位移归一化后编码为脉冲序列，直接驱动 IGZO 储池。
- **分类**：线性回归读出层经“赢者通吃”决策，实现动作类型判别。

在 UTD-MHAD、MSR Action3D、MSR Action Pairs、Florence 3D 四大公开数据集上均取得 > 90% 的识别率；全系统处理一次完整动作仅消耗 ~ 45.78 μJ，比 CPU、FPGA、GPU 等数字处理方法低两个数量级以上。

# 仿生材料内模拟光电储池计算系统

![iShot_2025-05-13_14.37.49](https://raw.githubusercontent.com/1910853272/image/master/img/202505131438826.png)

a.光信号由视网膜感受器接收，经过双极细胞传递触发动作电位，下丘脑感受野发放脉冲，进入视觉皮层提取特征。

b.3D 骨骼坐标序列输入高斯感受野GRF，将连续输入直接映射为多路脉冲序列，再进行储池计算。

c.光电突触晶体管阵列

d.不同栅压对应不同 GRF 中心值

e、f.32 × 32 1T1R 阵列实现的物理储池

## GRF神经元工作机制

每个GRF神经元会产生一个高斯分布输出 $G_i(A)$，其中输入强度为 A，高斯中心为 $\mu_i$，标准差为 $\sigma_i$。只有当某个神经元的输出为所有GRF中最大的时，该神经元才会产生脉冲。

$G_i(A)=\frac{1}{\sqrt{2\pi\sigma_i^2}}\exp\left(-\frac{\left(A-\mu_i\right)^2}{2\sigma_i^2}\right)$

神经元数量设为 m，每个GRF的中心由公式设置：

$\mu_i=A_{\min}+\frac{A_{\max}-A_{\min}}{m+1}\times i$

输入归一化到[-1, 1]区间，此时中心变为：

$\mu_i=\frac{2}{m+1}\times i-1$

# 基于高斯感受野的编码与光电储池计算

![iShot_2025-05-13_14.43.16](https://raw.githubusercontent.com/1910853272/image/master/img/202505131443134.png)

a.N 帧骨骼的 K个关节点三维坐标 $(x_i,y_i,z_i)$，每帧展平为长度为 3K的向量 ，整个动作构成一个 $3K\times N$的矩阵

$Act=[J_{11}J_{12}\ldots J_{1k};\ldots\ldots;J_{n1}J_{n2}\ldots J_{nk};\ldots\ldots;J_{N1}J_{N2}\ldots J_{Nk}]$。

以第一帧 $J_1$ 为参考，计算后续每帧相对于第一帧的坐标变化得到增量矩阵

$\Delta Act=[00\ldots0;\ldots\ldots;\Delta J_{n1}\Delta J_{n2}\ldots\Delta J_{nk};\ldots\ldots;\Delta J_{N1}\Delta J_{N2}\ldots\Delta J_{Nk}]$。

随机生成一个尺寸为$3K\times M$的掩码矩阵，元素为$\pm1$，其中 M为“掩码长度”。

增量向量$\Delta J_n$左乘掩码矩阵，得到长度为 M的虚拟节点向量$V_n$。

N 帧的虚拟节点按时间拼接，构成维度为 $1\times(M\cdot N)$的预输入矩阵$PreIn=[V_1,V_2,…,V_N]$。

b.增量矩阵

c.预输入矩阵PreIn

d.将 PreIn 的每个分量 A输入到 m个高斯接收域神经元中，输出满足高斯分布，形成稀疏的 spike 序列。

脉冲序列照射到三台不同栅压的 IGZO 光电突触晶体管上，产生的电流即构成该时间步的储池状态

e.连续4个光脉冲的电流响应

f.不同高斯感受野的记忆容量

# 短时记忆和非线性

利用具有光致电导持续性PPC和电解质耦合效应EDL的 IGZO 光电晶体管，在器件层面模拟出类脑“短时记忆”和“非线性”特性。

## 弛豫行为的数学建模

$\Delta I_{DS}=\kappa\cdot\exp\left[-\left(\frac{t}{\tau}\right)^\beta\right]$

$\kappa$：光刺激后引起的电流变化幅度；$\tau$：松弛时间常数；$\beta$：拉伸指数，决定曲线形状

## 时间序列下的响应叠加

当光脉冲在时刻 $t_1$ 施加，在时刻 $t_n$的响应为：

$\Delta I_{DS}=\begin{cases}0,&t_n<t_1\\\kappa\cdot\exp\left[-\left(\frac{(t_n-t_1)\cdot T}{\tau}\right)^\beta\right],&t_n\geq t_1&&\end{cases}$

若之后另一个相同光脉冲在 $t_2 > t_1$时施加，带来的新增响应为

$\Delta I_{DS}=\kappa\cdot\exp\left[-\left(\frac{(t_n-t_2)\cdot T}{\tau}\right)^\beta\right]\quad(t_n\geq t_2)$

对于两个脉冲的总响应叠加为

$\Delta I_{DS}=\begin{cases}0,&t_n<t_1\\\kappa\cdot\exp\left[-\left(\frac{(t_n-t_1)\cdot T}{\tau}\right)^\beta\right],&t_1\leq t_n<t_2\\\kappa\cdot\exp\left[-\left(\frac{(t_n-t_1)\cdot T}{\tau}\right)^\beta\right]+\kappa\cdot\exp\left[-\left(\frac{(t_n-t_2)\cdot T}{\tau}\right)^\beta\right],&t_n\geq t_2&&\end{cases}$

每个时间步的总输出电流为：

$I_{DS}(t_n) = \Delta I_{DS}(t_n) + I_0$

其中 $I_0$是没有光脉冲时的基线电流。

响应电流不仅取决于当前的光刺激，还与过去所有脉冲的残余弛豫电流有关，模拟了神经元的短期记忆能力。

![iShot_2025-05-27_15.17.53](https://raw.githubusercontent.com/1910853272/image/master/img/202505271549186.png)

# 基于仿生储池计算的标准数据集测试识别结果

![iShot_2025-05-13_14.44.36](https://raw.githubusercontent.com/1910853272/image/master/img/202505131446616.png)

a.自制动作高抛和UTD-MHAD 标准数据集中的投篮动作

b.光电脉冲驱动 IGZO 突触晶体管后采集的储池状态

c. UTD-MHAD 数据集上基于仿生储池计算的分类识别结果

d.不同GRF神经元数量的识别精度

e.其他验证数据集的识别精度

# 基于仿生储池计算系统的跌倒行为识别

![iShot_2025-05-13_14.49.01](https://raw.githubusercontent.com/1910853272/image/master/img/202505131449450.png)

a.计算架构示意。骨骼序列输入、GRF 编码 → 光电脉冲驱动 IGZO 储池 → 1T1R 忆阻器读出层 → 分类/预测输出

b.两类正常动作和三类跌倒动作

c.五种动作的 3D 骨骼帧示例

d、e.将训练得出的数值权重映射到忆阻器电导区间，并通过精确写入 1T1R 阵列

f.五分类识别精度

g.相比于 CPU、FPGA、GPU 等基于 CMOS 的数字处理器能耗对比



