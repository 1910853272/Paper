# 面向互联网视频的通用机器人学习：一项综述

本文综述了从互联网视频中提取知识的主流技术，包括：

- **视觉表征学习**：通过自监督学习（如对比学习、掩码建模）从视频中提取通用视觉特征。
- **行为克隆增强**：利用视频数据预训练策略模型，再通过机器人数据微调（如RT-2的VLM微调）。
- **动力学模型预训练**：从视频中学习物理规律（如物体运动预测），提升机器人环境建模能力。
- **跨模态对齐**：将视频中的动作描述（如“打开抽屉”）映射到机器人控制指令。

![iShot_2025-04-02_19.58.04](https://raw.githubusercontent.com/1910853272/image/master/img/202504021958045.png)