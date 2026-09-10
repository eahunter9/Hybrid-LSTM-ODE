library(MLmetrics)

PCIP <- function(data, realdata,n){
  count = 0
  for(i in 1:n){
    if((realdata$measurement[i] <= data$upper_bound[i]) & (realdata$measurement[i] >= data$lower_bound[i])){
      count  = count + 1
    }
    
  }
  pcip <- count/n
  return (pcip)
}


PCIP(sim_data[301:314,],real_data[301:314,],14)
PCIP(sim_data[301:328,],real_data[301:328,],28)
PCIP(sim_data[301:342,],real_data[301:342,],42)

     
MAPE(sim_data$measurement[301:314],real_data$measurement[301:314])
MAPE(sim_data$measurement[301:328],real_data$measurement[301:328])
MAPE(sim_data$measurement[301:342],real_data$measurement[301:342])
