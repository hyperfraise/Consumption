library(tsDyn)

#On charge les csv, la consommation, les vecteurs bianires de dates
raw_record <- read.csv("/tmp/nnetts/data10minutes.csv")
consumption = raw_record[,1]
beginning_training_set <- 1
end_training_set <- length(consumption)
n_forecast_points <- 144


train = ts(consumption[beginning_training_set:end_training_set],frequency=144)

m = nnetTs(train/150,m=24,d=48,steps=n_forecast_points,size=6,control=list(trace=F,decay=0.01,MaxNWts=10000,maxit=10000))


f = 150*predict(m,n.ahead=n_forecast_points)
plot(f)
print("writing predictions")
write(f,"/tmp/nnetts/output_nnetts.txt")
