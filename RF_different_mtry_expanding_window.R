# Code für den Expanding Window Ansatz 

rm(list = ls())

sink("/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/console_output.txt")


library(ranger)
library(dplyr)
library(scoringRules)
library(quantregForest)
library(knitr)
library(lubridate)

set.seed(2024)

# Setze das Arbeitsverzeichnis
setwd("/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code")

if (!dir.exists("res_different_mtry_exp_window_a")) {
  dir.create("res_different_mtry_exp_window_a/")
}

# Lese Trainingsdaten
dat <- read.csv("/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/Data/rf_data_1823_clean.csv") %>%
  mutate(date = as.Date(date))

dat <- na.omit(dat)

# Einstellungen
n_trees <- 100  # Anzahl der Bäume
n <- 35056  # Größe des Trainingsdatensatzes (erste 35.056 Zeilen)
dat_train <- dat[1:n, ]
dat_test <- dat[(n + 1):nrow(dat), ]
y_train <- as.vector(dat_train$load)
y_test <- dat_test$load
#dat_test <- dat_test %>% select(-load)

# Formel für das Modell konstruieren
fml_base <- as.formula("load ~ holiday + hour_int + weekday_int")

# Parameterkombinationen festlegen
packages <- c("ranger", "quantregForest")
time_trend_options <- c(TRUE, FALSE)
day_of_year_options <- c(TRUE, FALSE)

start_date <- min(dat_test$date)
end_date <- max(dat_test$date)
#months <- seq.Date(from = start_date, to = end_date, by = "month")

months <- seq.Date(from = as.Date("2022-01-01"), to = as.Date("2023-11-01"), by = "month")

# Liste zum Speichern der Ergebnisse
results <- list()

# Iteration für jedes Monat im Testdatensatz
for (month_start in months) {
  
  for (which_package in packages) {
    for (time_trend in time_trend_options) {
      for (day_of_year in day_of_year_options) {
        
        # Berechne das Monatsende
        month_start <- as.Date(month_start)
        print(paste("Monatsanfang:", month_start))
        
        month_end <- ceiling_date(month_start, "month") - days(1)
        print(paste("Monatsende:", month_end))
        
      
        # Filtere Testdaten für den aktuellen Monat
        dat_test_month <- dat_test %>% filter(date >= month_start & date <= month_end)
        
        # Trainingsdaten für das aktuelle "expanding window" (erste 35.056 Daten + alle Daten bis zum aktuellen Monat)
        #dat_train_expanding <- dat %>% filter(date <= month_end) %>% slice_head(n = n)
        dat_train_expanding <- dat %>% filter(date < month_start)
        
        
        cat("\n==== Verwendete Daten ====\n")
        cat("Trainingsdaten (Expanding Window):\n")
        cat("Die ersten 5 Zeilen des Trainingssets")
        print(head(dat_train_expanding, 5))  # Zeigt die ersten 5 Zeilen der Trainingsdaten
        cat("Die letzten 5 Zeilen des Trainingssets")
        print(tail(dat_train_expanding, 5))  # Zeigt eine Zusammenfassung der Trainingsdaten
        cat("Testdaten (Monat):\n")
        print(head(dat_test_month, 5))  # Zeigt die ersten 5 Zeilen der Testdaten
        print(tail(dat_test_month, 5))
        
      
        
        
        
        
        # Ausgabe der aktuellen Kombination
        cat("Verwendetes Paket:", which_package, "\n")
        cat("Zeittrend:", time_trend, "\n")
        cat("Tag des Jahres:", day_of_year, "\n\n")
        
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
        
        x_train <- as.matrix(model.matrix(fml, data = dat_train_expanding)[, -1])
        
        # x_test basiert jetzt nur auf den Daten von dat_test_month
        x_test <- as.matrix(model.matrix(fml, data = dat_test_month)[, -1])
        
        # Überprüfe die Dimensionen der Matrizen, um sicherzustellen, dass sie korrekt sind
        cat("Dimensionen von x_train:", dim(x_train), "\n")
        cat("Dimensionen von x_test:", dim(x_test), "\n")
        
        y_train <- dat_train_expanding$load
        y_test <- dat_test_month$load
        print(length(y_train))
        print(length(y_test))
      
        
    
        
        cat("Daten die für das Training und Testing dann ausgwählt wurden")
        cat("Traindaten (Monat):\n")
        print(head(x_train, 5))
        cat("Testdaten (Monat):\n")
        print(head(x_test, 5))
        cat("Dimension der Trainingsdaten")
        print(dim(x_train))
        cat("Dimension der Testdaten")
        print(dim(x_test))
        
        
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
            fit <- ranger(fml, data = dat_train_expanding, 
                          mtry = m_try,
                          quantreg = TRUE, 
                          keep.inbag = TRUE,
                          min.node.size = 1,
                          min.bucket = 1,
                          max.depth = NULL,
                          num.trees = 100) 
            pred <- predict(fit, type = "quantiles",
                            quantiles = grid_quantiles,
                            data = dat_test_month)$predictions
          } else {
            #fml_no_intercept <- update(fml, . ~ . - 1)  # Formel ohne Intercept
            #x_train <- as.matrix(model.matrix(fml_no_intercept, data = dat_train_expanding))
            #x_test <- as.matrix(model.matrix(fml_no_intercept, data = dat_test_month))
          
            #cat("x_train Matrix:\n")
            #print(head(x_train))
            #cat("x_test Matrix:\n")
            #print(head(x_test))
            
            # Umwandlung von y_train und y_test in numerische Vektoren
            #y_train <- as.vector(dat_train_expanding$load)
            #y_test <- as.vector(dat_test_month$load)
            
            fit <- quantregForest(x = x_train, 
                                  y = y_train, 
                                  ntree = n_trees,
                                  nodezize = 2,
                                  max_depth = NULL,
                                  mtry = m_try)
            pred <- predict(fit, what = grid_quantiles, newdata = x_test)
          }
          
          # Berechne CRPS und Fehlermaße und speichere alle Werte
          res <- data.frame(date = dat_test_month$date, crps = NA, ae = NA, se = NA)
          for (jj in 1:nrow(dat_test_month)) {
            res$crps[jj] <- crps_sample(y = y_test[jj], dat = pred[jj, ])
            res$ae[jj] <- abs(y_test[jj] - median(pred[jj,]))
            res$se[jj] <- (y_test[jj] - mean(pred[jj,]))^2
          }
          
          # Speichername generieren und Ergebnisse speichern
          save_name_all <- paste0("res_different_mtry_exp_window_a/", 
                                  which_package, "_",
                                  if (time_trend) "tt_" else "nott_", 
                                  if (day_of_year) "day_" else "month_", 
                                  "mtry", m_try,"_" ,month_end ,".csv")
          
          tryCatch({
            write.table(res, file = save_name_all, sep = ",", row.names = FALSE)
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

#sink()
