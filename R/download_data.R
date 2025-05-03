rm(list = ls())

setwd("/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/rf_thesis")

library(quantmod)

tickers <- c("AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", 
             "CVX", "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO",
             "MCD", "MMM", "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW", 
             "TRV", "UNH", "V", "VZ", "WMT")

getSymbols(tickers, src = "yahoo", verbose = TRUE)

all <- data.frame()

for (jj in tickers){
  tmp1 <- get(jj)
  nms <- colnames(tmp1)
  ind_high <- which(nms == paste0(jj, ".High"))
  ind_low <- which(nms == paste0(jj, ".Low"))
  ind_price <- which(nms == paste0(jj, ".Adjusted"))
  ind_vol <- which(nms == paste0(jj, ".Volume"))
  tmp_df <- data.frame(date = unname(time(tmp1)), 
                       ticker = jj, 
                       hml = unname(tmp1[,ind_high] - tmp1[,ind_low]), 
                       volume = unname(tmp1[, ind_vol]), 
                       price = unname(tmp1[, ind_price]))
  all <- rbind(all, tmp_df)
}
rownames(all) <- NULL
svnm <- paste0("stockdata_", 
               gsub(" ", "_", substr(Sys.time(), 1, 10)),
               ".csv")
write.csv(all, file = svnm, row.names = FALSE)