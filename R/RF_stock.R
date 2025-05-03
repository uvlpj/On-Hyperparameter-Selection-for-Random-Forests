# Code für den Stock Datensatz
# RF in ranger und quantregForest wird trainiert
# mtry = 1, 4, 6, 12, 18

rm(list = ls())

library(ranger)
library(dplyr)
library(scoringRules)
library(quantregForest)
library(knitr)

set.seed(2024)

# set working directory
setwd("/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code")

if (!dir.exists("res_stock_different_mtry_2")) {
  dir.create("res_stock_different_mtry_2/")
}

# read training data
# Setze den Pfad zum Ordner, in dem die Datensätze liegen
ordner_path <- "/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/rf_thesis/dat_final" # Passe den Pfad an

# Alle Dateien im Ordner auflisten
dateien <- list.files(path = ordner_path, pattern = "*.csv", full.names = TRUE)
# Datensätze und Dateinamen einlesen
datensaetze <- lapply(dateien, read.csv)

# Dateien einlesen und in eine Liste speichern


# Überprüfe die Struktur des ersten Datensatzes
str(datensaetze[[1]])
nrow(datensaetze[[1]]) # Datensatz hat die Länge 1723

# Einstellungen
n_trees <- 100  # Anzahl der Bäume
n <- 1206  # Größe des Trainingsdatensatzes ca 70% des gesamten Datensatzes


# Formel für das Modell konstruieren
fml_base <- as.formula("aret ~ aret_l1 + aret_l5 + aret_l22 + 
                       vol_l1 + vol_l5 + vol_l22 +
                       hml_l1 + hml_l5 + hml_22 +
                       vxv_l1 + vxv_l5 + vxv_l22 +
                       vxn_l1 + vxn_l5 + vxn_l22 +
                       vxd_l1 + vxd_l5 + vxd_l22")

# Parameterkombinationen festlegen
packages <- c("ranger", "quantregForest")
mtry_values <- c(1, 4, 6, 12, 18)

for (dat_index in seq_along(datensaetze)) {
  
  dat <- datensaetze[[dat_index]]  # Wähle den aktuellen Datensatz
  
  # Hole den echten Dateinamen 
  dat_name <- tools::file_path_sans_ext(basename(dateien[dat_index]))

  
  # Aufteilen in Trainings- und Testdaten
  dat_train <- dat[1:n, ]
  dat_test <- dat[(n + 1):nrow(dat), ]
  y_train <- as.vector(dat_train$aret)
  y_test <- dat_test$aret
  dat_test <- dat_test %>% select(-aret)
  
  # Formel für das Modell konstruieren
  fml_base <- as.formula("aret ~ aret_l1 + aret_l5 + aret_l22 + 
                         vol_l1 + vol_l5 + vol_l22 +
                         hml_l1 + hml_l5 + hml_l22 +
                         vxv_l1 + vxv_l5 + vxv_l22 +
                         vxn_l1 + vxn_l5 + vxn_l22 +
                         vxd_l1 + vxd_l5 + vxd_l22")
  
  # Schleife über die Pakete
  for (which_package in packages) {
    
    # Schleife über die mtry-Werte
    for (m_try in mtry_values) {
      cat("Datensatz:", dat_index, "Paket:", which_package, "mtry:", m_try, "\n")
      
      # Training des Modells je nach Paket
      if (which_package == "ranger") {
        fit <- ranger(fml_base, data = dat_train, 
                      mtry = m_try,
                      quantreg = TRUE, 
                      keep.inbag = TRUE,
                      min.node.size = 2,
                      min.bucket = 1,
                      max.depth = NULL,
                      num.trees = n_trees) 
        pred <- predict(fit, type = "quantiles", 
                        quantiles = (2 * (1:100) - 1) / (2 * 100), 
                        data = dat_test)$predictions
      } else if (which_package == "quantregForest") {
        fml_no_intercept <- update(fml_base, . ~ . - 1)  # Formel ohne Intercept
        x_train <- model.matrix(fml_no_intercept, data = dat_train) %>% as.matrix
        x_test <- model.matrix(fml_no_intercept, data = data.frame(aret = y_test, dat_test)) %>% as.matrix
        fit <- quantregForest(x = x_train, y = y_train, ntree = n_trees, mtry = m_try)
        pred <- predict(fit, what = (2 * (1:100) - 1) / (2 * 100), newdata = x_test)
      }
      
      # Fehlermaße berechnen
      res <- data.frame(date = dat_test$date, crps = NA, ae = NA, se = NA)
      for (jj in 1:nrow(dat_test)) {
        res$crps[jj] <- crps_sample(y = y_test[jj], dat = pred[jj, ])
        res$ae[jj] <- abs(y_test[jj] - median(pred[jj,]))
        res$se[jj] <- (y_test[jj] - mean(pred[jj,]))^2
      }
      
      # Berechne und speichere die Gesamtergebnisse für diesen mtry-Wert
      overall_results <- res %>%
        summarise(mean_crps = mean(crps, na.rm = TRUE),
                  mean_ae = mean(ae, na.rm = TRUE),
                  mean_se = mean(se, na.rm = TRUE),
                  root_mse = sqrt(mean(se, na.rm = TRUE)))
      
      # Speichername generieren und Ergebnisse speichern
      save_name <- paste0("res_stock_different_mtry_2/",which_package,"_" ,dat_name, "_mtry", m_try, ".csv")
      
      tryCatch({
        write.table(res, file = save_name, sep = ",", row.names = FALSE)
        cat("Ergebnisse gespeichert in:", save_name, "\n")
      }, error = function(e) {
        cat("Fehler beim Speichern der Datei:", save_name, "\n")
        cat("Fehlermeldung:", e$message, "\n")
      })
    }
  }
}
