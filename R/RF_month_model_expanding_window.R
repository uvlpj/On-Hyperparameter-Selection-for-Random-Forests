# Code für den Expanding Window Ansatz, Monatliche Werte 

rm(list = ls())

library(ranger)
library(dplyr)
library(scoringRules)
library(quantregForest)

set.seed(2024)

# set working directory
# set working directory
setwd("/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code")

if (!dir.exists("res_different_mtry_monthly_model")) {
  dir.create("res_different_mtry_monthly_model/")
}


# Daten laden und vorbereiten
dat <- read.csv("/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/Data/rf_data_1823_clean.csv") %>%
  mutate(date = as.Date(date))



dat <- na.omit(dat)

# Einstellungen
n_trees <- 100
n <- 35056  # Größe des initialen Trainingsdatensatzes
dat_train <- dat[1:n, ]
dat_test <- dat[(n + 1):nrow(dat), ]

# Monate im Testdatensatz identifizieren
test_months <- unique(format(dat_test$date, "%Y-%m"))

# Parameterkombinationen
packages <- c("ranger", "quantregForest")
time_trend_options <- c(TRUE, FALSE)
day_of_year_options <- c(TRUE, FALSE)


# Schleife über Monate
for (month in test_months) {
  cat("Prognose für Monat:", month, "\n")
  
  # Expanding Window: Trainingsdaten bis zum Ende des Vormonats
  end_of_previous_month <- as.Date(paste0(month, "-01")) - 1
  current_train <- dat %>% filter(date <= end_of_previous_month)
  
  # Testdaten für den aktuellen Monat
  current_test <- dat %>% filter(format(date, "%Y-%m") == month)
  cat("Monat:", month, "\n")
  y_train <- current_train$load
  y_test <- current_test$load
  
  for (which_package in packages) {
    for (time_trend in time_trend_options) {
      for (day_of_year in day_of_year_options) {
        for (load_lag1 in load_lag1_options) {
          
          # Formel anpassen
          fml <- as.formula("load ~ holiday + hour_int + weekday_int")
          if (day_of_year) {
            fml <- update(fml, . ~ . + yearday)
          } else {
            fml <- update(fml, . ~ . + month_int)
          }
          if (time_trend) {
            fml <- update(fml, . ~ . + time_trend)
          
          
          x_train <- model.matrix(fml, data = current_train)[, -1] %>% as.matrix
          x_test <- model.matrix(fml, data = current_test)[, -1] %>% as.matrix
          
          n_quantiles <- 1e2
          grid_quantiles <- (2 * (1:n_quantiles) - 1) / (2 * n_quantiles)
          p <- ncol(x_train)
          
          for (m_try in 1:p) {
            if (which_package == "ranger") {
              fit <- ranger(fml, data = current_train, 
                            mtry = m_try,
                            quantreg = TRUE, 
                            keep.inbag = TRUE,
                            min.node.size = 1,
                            num.trees = n_trees) 
              pred <- predict(fit, type = "quantiles",
                              quantiles = grid_quantiles,
                              data = current_test)$predictions
            } else {
              fit <- quantregForest(x = x_train, 
                                    y = y_train, 
                                    ntree = n_trees,
                                    mtry = m_try)
              pred <- predict(fit, what = grid_quantiles, newdata = x_test)
            }
            
            res <- data.frame(date = current_test$date, crps = NA, ae = NA, se = NA)
            for (jj in 1:nrow(current_test)) {
              res$crps[jj] <- crps_sample(y = y_test[jj], dat = pred[jj, ])
              res$ae[jj] <- abs(y_test[jj] - median(pred[jj,]))
              res$se[jj] <- (y_test[jj] - mean(pred[jj,]))^2
            }
            
            save_name <- paste0("res_different_mtry_monthly_model/",
                                which_package, "_",
                                if (time_trend) "tt_" else "nott_", 
                                if (day_of_year) "day_" else "month_", 
                                if (load_lag1) "lagged_" else "notlagged_", 
                                "mtry", m_try, "_", month, ".csv")
            
            write.table(res, file = save_name, sep = ",", row.names = FALSE)
            cat("Gespeichert:", save_name, "\n")
          }
        }
      }
    }
  }
}
