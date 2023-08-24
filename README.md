# 11482_cardio_guidance_Taraus

2023年嵌入式大赛11482队的Taraus部分代码

> 其余部分代码见下面的仓库链接

- aac_file 语音文件
- ai_sample Taraus端代码
- models wk模型文件

## 团队名称

狼牙带不动步兵

## 作品选题

有氧吧–让每个人都能享受到专业的有氧运动

## 作品介绍

“有氧吧”借助yolox-pose人体关键点检测技术，识别用户姿势，实时提供专业的健身指导、计数和姿势纠正，同时极具便携性，可以随时随地开展锻炼。“有氧吧”主体包括三部分，分别为Taurus、Pegasus和微信小程序，每一部分又包括多个模块。其中微信小程序是与用户进行交互的模块，通过WIFI将用户的控制指令传送到主控板Pegasus上，将用户指令设置对应的串口信息通过UART传输给Taurus，Taurus根据指令选择不同的模式，根据模式对识别到的关键点进行动作的判定，同时给出语音提示，并且返回动作计数或者运动得分给Pegasus，Pegasus将数据联通传感器模块的数据传给微信小程序，并在小程序端进行相应的数据展示。

## 本项目在海思官方仓库的代码

> 该仓库中的Taraus部分仅包含修改过的文件
> 该仓库中包括Pegasus代码

https://gitee.com/HiSpark/2023_hisilicon_embedded_competition/tree/master/AIOT/11482_cardio_guidance

## 本项目的微信小程序代码仓库

https://github.com/XianMengxi/Aerobic_mini_program

## 本项目神经网络训练简要介绍

https://blog.csdn.net/jingyu_1/article/details/132408986

## 本项目的视频演示（B站）

https://www.bilibili.com/video/BV1H94y1q7sa/#reply361176344
