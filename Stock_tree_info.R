# Code 
# Gleicher Code wie im Stock fall, nur dass hier die jeweilige Baumstruktur ausgegeben wird
rm(list = ls())

library(ranger)
library(dplyr)
library(scoringRules)
library(quantregForest)
library(knitr)

set.seed(2024)

# Verzeichnis für Ergebnisse erstellen
if (!dir.exists("res_stock_different_with_treestructure")) {
  dir.create("res_stock_different_with_treestructure/")
}

# Verzeichnis mit den Datensätzen definieren
ordner_path <- "/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/rf_thesis/dat_final"
dateien <- list.files(path = ordner_path, pattern = "*.csv", full.names = TRUE)
datensaetze <- lapply(dateien, read.csv)

# Einstellungen
n_trees <- 100  # Anzahl der Bäume
n <- 1206  # Größe des Trainingsdatensatzes (70% der Daten)

# Formel für das Modell
fml_base <- as.formula("aret ~ aret_l1 + aret_l5 + aret_l22 + 
                       vol_l1 + vol_l5 + vol_l22 +
                       hml_l1 + hml_l5 + hml_l22 +
                       vxv_l1 + vxv_l5 + vxv_l22 +
                       vxn_l1 + vxn_l5 + vxn_l22 +
                       vxd_l1 + vxd_l5 + vxd_l22")

# Parameter
packages <- c("ranger")
mtry_values <- c(1, 4, 6, 12, 18)

results <- list()

# Funktionen für Baumstrukturanalyse
get_tree_depth_iterative <- function(tree_info) {
  if (nrow(tree_info) == 1) {
    return(0)  # Falls der Baum nur aus einem Root-Leaf besteht, ist die Tiefe 0
  }
  
  depth_list <- rep(0, nrow(tree_info))  # Speichert die Tiefe jedes Knotens
  depth_list[1] <- 0  # Root-Node hat Tiefe 0 (wie in sklearn)
  
  for (i in 2:nrow(tree_info)) {
    parent_id <- match(tree_info$nodeID[i], tree_info$leftChild)  # Elternknoten links finden
    if (is.na(parent_id)) {
      parent_id <- match(tree_info$nodeID[i], tree_info$rightChild)  # Falls nicht links, dann rechts
    }
    
    if (!is.na(parent_id)) {
      depth_list[i] <- depth_list[parent_id] + 1
    }
  }
  
  return(max(depth_list))  # Maximale Tiefe des Baumes zurückgeben
}

get_internal_node_count <- function(tree_info) {
  return(sum(!tree_info$terminal))  # Alle nicht-terminalen Knoten zählen
}

get_leaf_count <- function(tree_info) {
  return(sum(tree_info$terminal))  # Alle terminalen Knoten (Blätter) zählen
}

# Iteration über alle Datensätze
for (dat_index in seq_along(datensaetze)) {
  dat <- datensaetze[[dat_index]]
  dat_name <- tools::file_path_sans_ext(basename(dateien[dat_index]))
  
  # Trainings- und Testdaten aufteilen
  dat_train <- dat[1:n, ]
  dat_test <- dat[(n + 1):nrow(dat), ]
  y_train <- dat_train$aret
  y_test <- dat_test$aret
  dat_test <- dat_test[, !names(dat_test) %in% "aret"]  # Zielvariable entfernen
  
  # Iteration über die Pakete (hier nur "ranger")
  for (which_package in packages) {
    for (m_try in mtry_values) {
      cat("Datensatz:", dat_index, "Paket:", which_package, "mtry:", m_try, "\n")
      
      # Features extrahieren
      x_train <- model.matrix(fml_base, data = dat_train)[, -1] %>% as.matrix
      x_test <- model.matrix(fml_base, data = data.frame(aret = y_test, dat_test))[, -1] %>% as.matrix
      
      # Modell trainieren
      fit <- ranger(fml_base, data = dat_train, 
                    mtry = m_try,
                    quantreg = TRUE, 
                    min.node.size = 1,
                    max.depth = NULL,
                    replace = TRUE,
                    sample.fraction = 1,
                    num.trees = n_trees) 
      
      # Baumstruktur analysieren für alle Bäume
      tree_depths <- numeric(n_trees)
      internal_nodes <- numeric(n_trees)
      leaf_counts <- numeric(n_trees)
      
      for (i in 1:n_trees) {
        tree_data <- tryCatch(treeInfo(fit, tree = i), error = function(e) NULL)
        
        if (!is.null(tree_data)) {
          tree_depths[i] <- get_tree_depth_iterative(tree_data)
          internal_nodes[i] <- get_internal_node_count(tree_data)
          leaf_counts[i] <- get_leaf_count(tree_data)
        } else {
          tree_depths[i] <- NA
          internal_nodes[i] <- NA
          leaf_counts[i] <- NA
        }
      }
      
      # Durchschnittswerte der Baumstruktur
      cat("tree_depths")
      print(tree_depths)
      avg_tree_depth <- mean(tree_depths, na.rm = TRUE)
      avg_internal_nodes <- mean(internal_nodes, na.rm = TRUE)
      avg_leaves <- mean(leaf_counts, na.rm = TRUE)
      
      cat("Durchschnittliche Baumtiefe für mtry =", m_try, ":", avg_tree_depth, "\n")
      cat("Durchschnittliche Anzahl interner Knoten:", avg_internal_nodes, "\n")
      cat("Durchschnittliche Anzahl der Blätter:", avg_leaves, "\n")
      
      # Vorhersagen
      pred <- predict(fit, type = "quantiles", 
                      quantiles = (2 * (1:100) - 1) / (2 * 100), 
                      data = dat_test)$predictions
      
      # Fehlermaße berechnen
      res <- data.frame(date = dat_test$date, crps = NA, ae = NA, se = NA)
      for (jj in 1:nrow(dat_test)) {
        res$crps[jj] <- crps_sample(y = y_test[jj], dat = pred[jj, ])
        res$ae[jj] <- abs(y_test[jj] - median(pred[jj, ]))
        res$se[jj] <- (y_test[jj] - mean(pred[jj, ]))^2
      }
      
      # Gesamtergebnisse für diesen mtry-Wert
      overall_results <- res %>%
        summarise(mean_crps = mean(crps, na.rm = TRUE),
                  mean_ae = mean(ae, na.rm = TRUE),
                  mean_se = mean(se, na.rm = TRUE),
                  root_mse = sqrt(mean(se, na.rm = TRUE)),
                  avg_tree_depth = avg_tree_depth,
                  avg_internal_nodes = avg_internal_nodes,
                  avg_leaves = avg_leaves)
      
      # Ergebnisse speichern
      save_name <- paste0("res_stock_different_with_treestructure/", which_package, "_", dat_name, "_mtry", m_try, ".csv")
      write.table(res, file = save_name, sep = ",", row.names = FALSE)
      
      cat("Ergebnisse gespeichert in:", save_name, "\n")
    }
  }
}

