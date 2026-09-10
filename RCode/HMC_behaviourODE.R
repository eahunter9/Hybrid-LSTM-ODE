  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(lubridate)
  library(here)
  library(cmdstanr)
  library(posterior)
  library(tidybayes)
  library(tidyr)
  library(readsdr)
  library(bayesplot)
  library(extraDistr)
  library(GGally)
  library(ggpubr)
  library(ggridges)
  library(gridExtra)
  library(kableExtra)
  library(Metrics)
  library(patchwork)
  library(purrr)
  library(readr)
  library(scales)
  library(stringr)
  library(viridisLite)
  library(tibble)
  

  source("stan_data.R")

  seir_mdl_sde <- list("Deaths ~ poisson(net_flow(Total_Cases))")
  
  
  DT = 1/10
  
  
  syn <- as.tibble(read.csv("coviddeaths_1styear.csv",header = TRUE))
  
  syn$measurement <- as.numeric(round(syn$Newdeaths))
  syn <- syn[1:320,]
  
  ggplot(syn,aes(x=Time,y=measurement))+geom_point()+geom_line()+
    theme_classic()
  
  stan_filepath <- file.path( "behaviour ODE.stan")
  
  
  stan_d <- list(n_obs      = (nrow(syn)),
                 n_weeks = nrow(syn),
                 Deaths  = syn$measurement,
                 t0         = 0,

                 n_params = 12,
                 n_difeq  = 8,
                 ts         = 1:(nrow(syn))#,
    
                 )
  
  
  
  mod_sde         <- cmdstan_model( stan_filepath)
  
  
  fit <- mod_sde$sample(data              = stan_d,
                    chains            =4,
                    parallel_chains   = 4,
                    iter_warmup       = 1000,
                    iter_sampling     = 1000,
                    refresh           = 100,
                    save_warmup       = FALSE,  
                    max_treedepth  = 20
    
                   ) 
  
  
  fit$diagnostic_summary()
  
  
  posterior_SEIR <-  as_draws_df(fit$draws()) %>%
    as_tibble()
  