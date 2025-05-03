rm(list = ls())

library(zoo)
library(dplyr)
library(ranger)

setwd("/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/rf_thesis")

# load data
stocks <- read.csv("stockdata_2024-11-11.csv")
fred <- read.csv("freddata_2024-11-11.csv", na = ".") %>%
  filter(DATE >= "2007-12-04") %>% 
  transmute(date = DATE, VXVCLS, VXNCLS, VXDCLS)

# helper functions
lgr <- function(x){
  c(NA, 100*(diff(log(x))))
}
corsi_lags <- function(x){
  stopifnot(length(x) == 22)
  z <- abs(x) # take absolute values
  c(z[22], mean(z[18:22]), mean(z))
}

# get list of ticker symbols
tickers <- stocks %>%  
  pull(ticker) %>% unique %>% sort

# loop over tickers  
for (tt in tickers){
  dat0 <- stocks %>% filter(ticker == tt) %>%
    arrange(date) 
  if (any(dat0$hml == 0) | any(dat0$volume == 0)){
    print(paste("Zeros in data for", tt))
  }
  dat1 <- dat0 %>% 
    filter(hml != 0, volume != 0) %>%
    # use hml in levels, price and volume as log growth rates
    transmute(date, hml, ret = lgr(price), 
              vol = lgr(volume)) %>%
    filter(date >= "2017-12-04")
  # merge with fred data (vola indices)
  dat2 <- merge(dat1, fred, all.x = TRUE)
  
  # fill NAs for all variables except return and date
  # (last obs. carried forward)
  var_inds <- which(!(names(dat2) %in% c("date", "ret")))
  for (kk in var_inds){
    dat2[,kk] <- zoo::na.locf(dat2[,kk])
  }
  
  # make matrix with dependent and independent variables
  dat3 <- matrix(NA, nrow(dat2), 19)
  # absolute return (dependent var.) in first column
  dat3[, 1] <- abs(dat2$ret) 
  # use six regressors (ret, vol, hml, VXVCLS, VXNCLS, VXDCLS),
  # with three lags for each regressor
  for (jj in 23:nrow(dat2)){
    inds <- (jj-22):(jj-1)
    dat3[jj, 2:4] <- corsi_lags(dat2$ret[inds]) # lagged abs. returns
    dat3[jj, 5:7] <- corsi_lags(dat2$vol[inds]) # lagged abs. volume
    dat3[jj, 8:10] <- corsi_lags(dat2$hml[inds]) # lagged high minus low
    dat3[jj, 11:13] <- corsi_lags(dat2$VXVCLS[inds]) # lagged vola indices
    dat3[jj, 14:16] <- corsi_lags(dat2$VXNCLS[inds]) 
    dat3[jj, 17:19] <- corsi_lags(dat2$VXDCLS[inds]) 
  }
  
  # put matrix into data frame (for better handling with ranger package)
  # skip first 22 rows that contain NAs
  dat4 <- data.frame(date = dat2$date[23:nrow(dat3)], 
                     y = dat3[23:nrow(dat3),1], 
                     x = dat3[23:nrow(dat3),-1]) 
  
  # column names
  names(dat4) <- c("date", "aret", 
                   paste0("aret_l", c(1, 5, 22)),
                   paste0("vol_l", c(1, 5, 22)),
                   paste0("hml_l", c(1, 5, 22)), 
                   paste0("vxv_l", c(1, 5, 22)), 
                   paste0("vxn_l", c(1, 5, 22)), 
                   paste0("vxd_l", c(1, 5, 22)))
  
  # save data
  write.csv(dat4, file = paste0("dat_final/", tt, ".csv"),
            row.names = FALSE)
  
  # briefly compare lm and ranger predictions
  
  # get indexes of evaluation dates
  eval_inds <- which(dat4$date >= "2022-01-01")
  date_col <- which(names(dat4) == "date")
  
  # fit linear model, compute forecasts and errors
  fit_lm <- lm(aret~., data = dat4[-eval_inds, -date_col])
  e_lm <- dat4$aret[eval_inds] - 
    predict(fit_lm, newdata = dat4[eval_inds, ])
  
  # fit RF, compute forecasts and errors
  fit_rf <- ranger(aret~., data = dat4[-eval_inds, -date_col])
  pred_rf <- predict(fit_rf, 
                     data = dat4[eval_inds, ], 
                     type = "response")
  e_rf <- dat4$aret[eval_inds] - pred_rf$predictions
  
  # print result
  print(tt)
  rel_mse <- mean((e_rf)^2)/mean((e_lm)^2)
  print(paste("Relative MSE RF versus linear model:", 
              round(rel_mse, 2)))
}