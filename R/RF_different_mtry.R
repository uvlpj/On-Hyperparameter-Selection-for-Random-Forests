# Code 
# Random Forest in ranger und quantregForest
# verschiedenen Feature Kombinationen
# und mtry = 1, ..., p
# Berechne dann CRPS, SE, AE

rm(list = ls())

library(ranger)
library(dplyr)
library(scoringRules)
library(quantregForest)
library(knitr)

set.seed(2024)
#set.seed(4000)

# set working directory
setwd("/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code")

if (!dir.exists("res_different_mtry_max")) {
  dir.create("res_different_mtry_max/")
}

#if (!dir.exists("res_different_mtry_a1_v4/predictions")) {
#  dir.create("res_different_mtry_a1_v4/predictions/")
#}



# read training data
dat <- read.csv("/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/Data/rf_data_1823_clean.csv") %>%
  mutate(date = as.Date(date))

dat <- dat %>%
  arrange(date) %>%
  mutate(load_lag1 = lag(load, 1))

dat <- na.omit(dat)
# Einstellungen
n_trees <- 100  # Anzahl der Bäume
n <- 35056  # Größe des Trainingsdatensatzes
dat_train <- dat[1:n, ]
dat_test <- dat[(n + 1):nrow(dat), ]
y_train <- as.vector(dat_train$load)
y_test <- dat_test$load
#dat_test <- dat_test %>% select(-load)
dat_test <- dat_test[, !names(dat_test) %in% "load"]

dim(dat)
dim(dat_train)  
dim(dat_test)

last_value <- y_train[length(y_train)]
print(last_value)

last_value <- y_test[length(y_test)]
print(last_value)

# Formel für das Modell konstruieren
fml_base <- as.formula("load ~ holiday + hour_int + weekday_int")

# Parameterkombinationen festlegen
packages <- c("ranger", "quantregForest")
time_trend_options <- c(TRUE, FALSE)
day_of_year_options <- c(TRUE, FALSE)
load_lag1_options <- c(TRUE, FALSE)

# Initialisiere Liste zum Speichern der Gesamtergebnisse
results <- list()

# Schleife über alle Kombinationen der Parameter
for (which_package in packages) {
  for (time_trend in time_trend_options) {
    for (day_of_year in day_of_year_options) {
      for (load_lag1 in load_lag1_options) {
        
        # Ausgabe der aktuellen Kombination
        cat("Verwendetes Paket:", which_package, "\n")
        cat("Zeittrend:", time_trend, "\n")
        cat("Tag des Jahres:", day_of_year, "\n")
        cat("Lagged Load:", load_lag1, "\n\n")
        
        # Aktualisiere die Formel je nach Parameter
        fml <- fml_base
        if (day_of_year) {
          fml <- update(fml, . ~ . + yearday)
        } else {
          fml <- update(fml, . ~ . + month_int)
        }
        if (time_trend) {
          fml <- update(fml, . ~ . + time_trend)
        }
        if (load_lag1) {
          fml <- update(fml, . ~ . + load_lag1)
        }
        
        # Erstelle die Matrix ohne Interzept
        x_train <- model.matrix(fml, data = dat_train)[, -1] %>% as.matrix
        x_test <- model.matrix(fml, data = data.frame(load = y_test, dat_test))[, -1] %>% as.matrix
        
        n_quantiles <- 1e2
        grid_quantiles <- (2 * (1:n_quantiles) - 1) / (2 * n_quantiles)
        p <- ncol(x_train)  # Anzahl der Variablen in den Trainingsdaten
        print(head(x_train))
        
        # Schleife über mtry von 1 bis zur Anzahl der Variablen
        for (m_try in 1:p) {
          if (m_try > p) {
            cat("mtry =", m_try, "überschreitet die Anzahl der Variablen. Überspringe diese Iteration.\n")
            next  # überspringe diese Iteration, falls m_try größer ist
          }
          cat("Trainiere mit mtry =", m_try, "\n")
          
          if (which_package == "ranger") {
            fit <- ranger(fml, data = dat_train, 
                          mtry = m_try,
                          quantreg = TRUE, 
                          keep.inbag = TRUE,
                          min.node.size = 1,
                          min.bucket = 1,
                          max.depth = NULL,
                          num.trees = n_trees) 
            pred <- predict(fit, type = "quantiles",
                            quantiles = grid_quantiles,
                            data = dat_test)$predictions
            cat("predictions ranger:\n")
            print(pred)
          } else {
            fml_no_intercept <- update(fml, . ~ . - 1)  # Formel ohne Intercept
            x_train <- model.matrix(fml_no_intercept, data = dat_train) %>% as.matrix
            x_test <- model.matrix(fml_no_intercept, data = data.frame(load = y_test, dat_test)) %>% as.matrix
            cat("x_train Matrix ohne Intercept:\n")
            print(head(x_train))
            fit <- quantregForest(x = x_train, 
                                  y = y_train, 
                                  ntree = n_trees,
                                  mtry = m_try,
                                  max_depth = NULL,
                                  nodesize = 2)
            pred <- predict(fit, what = grid_quantiles, newdata = x_test)
            cat("predictions quantregForest:\n")
            print(pred)
          }
          
          # Berechne CRPS und Fehlermaße und speichere alle Werte
          res <- data.frame(date = dat_test$date, crps = NA, ae = NA, se = NA)
          for (jj in 1:nrow(dat_test)) {
            res$crps[jj] <- crps_sample(y = y_test[jj], dat = pred[jj, ])
            res$ae[jj] <- abs(y_test[jj] - median(pred[jj,]))       # ------> Median der 100 quantile vorhersagen
            res$se[jj] <- (y_test[jj] - mean(pred[jj,]))^2          # -----> Mean der 100 quantile vorhersagen 
          }
          
          pred_df <- data.frame(date = dat_test$date, pred)
          
          # Speichername generieren und Ergebnisse speichern
          save_name_all <- paste0("res_different_mtry_max/", 
                                  which_package, "_",
                                  if (time_trend) "tt_" else "nott_", 
                                  if (day_of_year) "day_" else "month_", 
                                  if (load_lag1) "lagged_" else "notlagged_", "mtry", m_try, ".csv")
          
          #save_name_pred <- paste0("res_different_mtry_a1_v4/predictions/", 
          #                        which_package, "_",
          #                        if (time_trend) "tt_" else "nott_", 
          #                        if (day_of_year) "day_" else "month_", 
          #                        if (load_lag1) "lagged_" else "notlagged_", "mtry", m_try, ".csv")
          
          tryCatch({
            write.table(res, file = save_name_all, sep = ",", row.names = FALSE)
            #write.table(pred_df, file = save_name_pred, sep = ",", row.names = FALSE)
            #cat("Quantilvorhersagen gespeichert:", save_name_pred, "\n")
            cat("Gespeichert:", save_name_all, "\n")
          }, error = function(e) {
            cat("Fehler beim Speichern der Datei:", save_name_all, "\n")
            cat("Fehlermeldung:", e$message, "\n")
          })
          
          # Berechne und speichere die Gesamtergebnisse (CRPS, MAE, SE)
          overall_results <- res %>%
            summarise(mean_crps = mean(crps, na.rm = TRUE),  # average CRPS
                      mean_ae = mean(ae, na.rm = TRUE),      # average AE (MAE)
                      mean_se = mean(se, na.rm = TRUE),      # average SE (MSE)
                      root_mse = sqrt(mean_se))              # root mse
          
          results[[m_try]] <- overall_results
        }
      }
    }
  }
}

# Gesamtergebnisse ausgeben
results_df <- do.call(rbind, results)
print(results_df)
