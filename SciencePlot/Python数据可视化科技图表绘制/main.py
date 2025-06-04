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
