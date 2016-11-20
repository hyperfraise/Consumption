library(forecast)

#On charge les csv, la consommation, les vecteurs bianires de dates
#on lit le dataset dans un endroit un peu cache sur l'instance
raw_record <- read.csv("/tmp/nnetar/data10minutes.csv")
consumption = raw_record[,1]
size=max(consumption)

train <- ts(consumption[1:length(consumption)],frequency=144)

#on cree 5 reseaux dont on moyenne la prediction
m=nnetar(train/size,size=10,P=14,p=5,h=144,repeats=5,MaxNWts=30000,trace=F)

f=forecast(m,h=144)$mean*size
#on corrige les points negatifs
for (i in 1:length(f)){
  f[i]=max(0,f[i])
}

print("writing predictions")
write(f, "/tmp/nnetar/output_nnetar.txt")
