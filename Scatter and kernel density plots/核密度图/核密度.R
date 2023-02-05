# 加载R包，没有安装请先安装  install.packages("包名") 
library(ggplot2)
library(reshape2)

# 读取核密度图数据文件
df = read.table("D:/研究生打工区/桌面文件/DoubleSG-DTA/R语言代码/散点图/kiba_predict.txt",header = T,sep = '\t')
# 把数据转换成ggplot常用的类型（长数据）
df = melt(df)                    # melt出自reshape2包
head(df)                         # 查看转换完成的数据的前几行
# 绘图
ggplot(df,aes(x=value,   
              fill=variable,     # fill填充颜色，根据变量名赋值
              colour=variable))+ # colour图形边界颜色，根据变量名赋值
  geom_density(alpha=0.2,        # 填充颜色透明度
               size=1.5,           # 线条粗细
               linetype = 1      # 线条类型1是实线，2是虚线
  )+
  
  labs(x ="Affinity Score", y = "Density")+scale_x_continuous(breaks=seq(0, 15, 1))+
  theme_bw()+theme(panel.grid.major.x = element_blank(),
                   panel.grid.minor.x = element_blank(),
                   panel.grid.major.y = element_blank(),
                   panel.grid.minor.y = element_blank())                   # 白色主题

# 补充知识：
# fill   一般是指填充颜色
# color  一般是指线和点的颜色
# colour 一般是指图形边界颜色