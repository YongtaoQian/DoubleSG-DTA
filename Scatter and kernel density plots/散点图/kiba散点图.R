setwd("D:/研究生打工区/桌面文件/DoubleSG-DTA/R语言代码/散点图")
library(ggplot2)
kiba=read.table('kiba_predict.txt',sep = '\t')
p <- ggplot(kiba,aes(V2,V1))+
  
  #使用geom_point()函数绘制散点图，其中alpha为设置透明度（在数据量大，重叠点多的情况下设置alpha）
  
  geom_point(size=2.5,color="#1f77b4",alpha=1)+
  
  #使用geom_smooth()函数绘制拟合线，其中lm为线性拟合，se设置为FALSE取消置信区间
  
  geom_smooth(color="#FF0000",method="lm",se=FALSE)+
  
  #设置x，y标签和散点图标题
  
  labs(y="Prediction Affinities",x="Ground Truth")+scale_x_continuous(breaks=seq(1, 15, 2))+scale_y_continuous(breaks=seq(1, 15, 2))+
  
  #样式大小调整
  
  theme_bw()+theme(plot.title=element_text(hjust=0.5,size=20),axis.title=element_text(size=18,color ='black'),axis.text.x = element_text(size = 15,color ='black'),axis.text.y = element_text(size = 15,color ='black'),panel.grid=element_blank())
p + theme(panel.border = element_rect(fill=NA,color="black", size=4, linetype="solid"))
ggExtra::ggMarginal(p, type ="histogram",fill="#1f77b4",color="#1f77b4")
dev.off()
