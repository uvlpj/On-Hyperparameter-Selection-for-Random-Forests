# Code
# Erster Abschnitt ----
#   RF in ranger ohne Bootstrapping mit mtry = 1 und mtry = p
#   gibt die durchschnittliche Tiefe, Anzahl an interner Knoten und Blätter an 
#   gibt die Frequency Feature Selction an 
#   für eine Bestimmte Feature kombinations

# Zweiter Abschnitt  ----
#   Plottet die ersten zwei Ebenen des Baums
#   gibt die Predictions über den gesamten Testzeitraum aus

# Dritter Abschnitt ----
#   Plottet den gesamten Baum
#   gibt Baumstruktur wieder wie Baumtiefe, Anzahl an Blätter und interne Knoten sowie Frequency of Feature Selctions

rm(list = ls())

library(ranger)
library(dplyr)
library(scoringRules)
library(quantregForest)
library(knitr)
library(ggplot2)

set.seed(2024)

# set working directory
setwd("/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code")

if (!dir.exists("res_different_mtry1_with_treestructure")) {
  dir.create("res_different_mtry1_with_treestructure")
}

if (!dir.exists("tree_structure")) {
  dir.create("tree_structure")
}

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

# Formel für das Modell konstruieren
fml_base <- as.formula("load ~ holiday + hour_int + weekday_int")

# Parameterkombinationen festlegen
packages <- c("ranger")
time_trend_options <- c(FALSE)
day_of_year_options <- c(TRUE)
load_lag1_options <- c(TRUE)

# Initialisiere Liste zum Speichern der Gesamtergebnisse
results <- list()

get_tree_depth_iterative <- function(tree_info) {
  if (nrow(tree_info) == 1) {
    return(0)  # Falls der Baum nur aus einem einzigen Knoten besteht, ist die Tiefe 0
  }
  
  depth_list <- rep(0, nrow(tree_info))  # Initialisiere Liste zur Speicherung der Tiefe jedes Knotens
  depth_list[1] <- 0  # Root-Knoten hat Tiefe 0
  
  for (i in 2:nrow(tree_info)) {
    parent_id <- match(tree_info$nodeID[i], tree_info$leftChild)  # Suche Elternknoten links
    if (is.na(parent_id)) {
      parent_id <- match(tree_info$nodeID[i], tree_info$rightChild)  # Falls nicht links, dann rechts
    }
    
    if (!is.na(parent_id)) {
      depth_list[i] <- depth_list[parent_id] + 1  # Tiefe des Elternknotens + 1
    }
  }
  
  return(max(depth_list))  # Maximale Tiefe des Baums zurückgeben
}

# Funktion zur Berechnung der Anzahl interner Knoten
get_internal_node_count <- function(data) {
  # Filtere alle terminalen Knoten heraus und zähle die verbleibenden (internen) Knoten
  internal_nodes <- subset(data, terminal == FALSE)
  return(nrow(internal_nodes))
}


get_split_feature_frequencies <- function(tree_data) {
  internal_nodes <- subset(tree_data, terminal == FALSE)
  table(internal_nodes$splitvarName)
}

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
                          num.trees = n_trees,
                          min.node.size = 1,
                          max.depth = NULL,
                          min.bucket = 1,
                          #keep.inbag = TRUE,
                          quantreg = TRUE, 
                          sample.fraction = 1,
                          replace = FALSE, # Bootstrapping ist deaktiviert
                          node.stats = TRUE) 
            pred <- predict(fit, type = "quantiles",
                            quantiles = grid_quantiles,
                            data = dat_test)$predictions
            
            all_tree_splits <- list()
            tree_depths <- numeric(n_trees)  # Speichern der Tiefe für jeden Baum
            internal_node_counts <- numeric(n_trees)  # Speichern der Anzahl der internen Knoten für jeden Baum
            feature_split_list <- list()
            
            for (i in 1:n_trees) {
              tree_data <- treeInfo(fit, tree = i)
              tree_data$Tree_ID <- i  # Füge Baum-ID hinzu
              
              tree_depths[i] <- get_tree_depth_iterative(tree_data)
              # Berechne die Anzahl der internen Knoten für den aktuellen Baum
              internal_node_counts[i] <- get_internal_node_count(tree_data)
              
              all_tree_splits[[i]] <- tree_data
              
              split_counts <- get_split_feature_frequencies(tree_data)
              split_counts_df <- as.data.frame(split_counts)
              names(split_counts_df) <- c("feature", "count")
              split_counts_df$tree <- i
              feature_split_list[[i]] <- split_counts_df
              
            }
            
            get_leaf_count <- function(data) {
              # Filter terminal nodes (i.e., leaves)
              leaf_nodes <- subset(data, terminal == TRUE)
              return(nrow(leaf_nodes))
            }
            
            
            avg_tree_depth <- mean(tree_depths)
            avg_internal_nodes <- mean(internal_node_counts)
            leaf_counts <- numeric(n_trees)
            
            cat("Durchschnittliche Baumtiefe für mtry =", m_try, ":", avg_tree_depth, "\n")
            cat("Durchschnittliche Anzahl interner Knoten für mtry =", m_try, ":", avg_internal_nodes, "\n")
            
            split_feature_df <- bind_rows(feature_split_list)
            
            total_feature_usage <- split_feature_df %>%
              group_by(feature) %>%
              summarise(total_splits = sum(count)) %>%
              arrange(desc(total_splits))
            
            cat("Totale Splits pro Feature (summe aller Bäume):\n")
            print(total_feature_usage)
            
            avg_feature_usage <- split_feature_df %>%
              group_by(feature) %>%
              summarise(total_splits = sum(count)/n_trees) %>%
              arrange(desc(total_splits))
            
            cat("Durchschnittliche Splits pro Feature (über alle Bäume):\n")
            print(as.data.frame(avg_feature_usage), digits = 10)
            
            # Ausgabe der Tiefe und der Anzahl der internen Knoten für jeden Baum
            for (i in 1:n_trees) {
              cat("Baum", i, "hat eine Tiefe von", tree_depths[i], "und", internal_node_counts[i], "interne Knoten.\n")
              tree_data <- treeInfo(fit, tree = i)
              tree_data$Tree_ID <- i  # Add Tree ID
              
              # Calculate the number of leaves for the current tree
              leaf_counts[i] <- get_leaf_count(tree_data)
            }
            avg_leaves <- mean(leaf_counts)
            
            cat("Durchschnittliche Anzahl der Blätter für mtry =", m_try, ":", avg_leaves, "\n")
            # Alle Daten zusammenfügen
            split_data <- do.call(rbind, all_tree_splits)
            
            tree_filename <- paste0("tree_structure/",
                                    which_package, "_",
                                    if (time_trend) "tt_" else "nott_", 
                                    if (day_of_year) "day_" else "month_", 
                                    if (load_lag1) "lagged_" else "notlagged_", 
                                    "mtry", m_try, "_trees.csv")
            write.csv(split_data, file = tree_filename, row.names = FALSE)
            cat("Baumstruktur gespeichert unter:", tree_filename, "\n")
            
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
                                  nodesize = 2)
            pred <- predict(fit, what = grid_quantiles, newdata = x_test)
          }
          
          # Berechne CRPS und Fehlermaße und speichere alle Werte
          res <- data.frame(date = dat_test$date, crps = NA, ae = NA, se = NA)
          for (jj in 1:nrow(dat_test)) {
            res$crps[jj] <- crps_sample(y = y_test[jj], dat = pred[jj, ])
            res$ae[jj] <- abs(y_test[jj] - median(pred[jj,]))
            res$se[jj] <- (y_test[jj] - mean(pred[jj,]))^2
          }
          
          # Speichername generieren und Ergebnisse speichern
          save_name_all <- paste0("res_different_mtry1_with_treestructure/", 
                                  which_package, "_",
                                  if (time_trend) "tt_" else "nott_", 
                                  if (day_of_year) "day_" else "month_", 
                                  if (load_lag1) "lagged_" else "notlagged_", "mtry", m_try, ".csv")
          
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










#==================================================================================
library(dplyr)
# read training data
dat <- read.csv("/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/Data/rf_data_1823_clean.csv") %>%
  mutate(date = as.Date(date))
set.seed(2024)


output_dir <- "predictions/"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

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

dat_train <- dat_train[c("load", "holiday", "hour_int", "weekday_int", "load_lag1", "yearday")]
dat_test <- dat_test[c("holiday", "hour_int", "weekday_int", "load_lag1", "yearday")]


# Formel für das Modell konstruieren
fml_base <- as.formula("load ~ holiday + hour_int + weekday_int + load_lag1 + yearday")
str(dat_train)  
str(dat_test)
# Parameterkombinationen festlegen
packages <- c("ranger")
time_trend_options <- c(FALSE)
day_of_year_options <- c(TRUE)
load_lag1_options <- c(TRUE)

print(nrow(dat_test))

index_to_predict <- 1
single_point_X_test <- dat_test[index_to_predict, , drop = FALSE] 


rf_ranger <- ranger(
  load ~ holiday + hour_int + weekday_int + load_lag1 + yearday , 
  data = dat_train,
  mtry = 5,
  num.trees = 1,  
  min.node.size = 1,
  min.bucket = 1,
  max.depth = NULL,  
  keep.inbag = TRUE,
  replace = FALSE,             
  sample.fraction = 1    
)

cat("dat_train")
print(head(dat_train, 10))
print(tail(dat_train, 10))
#prediction <- predict(rf_ranger, data = single_point_X_test)
prediction <- predict(rf_ranger, data = dat_test)
#cat("single_point_X_test")
#print(single_point_X_test)


cat("prediction")
print(prediction$predictions) 
#write.csv(prediction$predictions, "predictions_mntry_1.csv", row.names = FALSE)



#inbag_counts <- rf_ranger$inbag.counts[[1]]  # Anzahl, wie oft jeder Datenpunkt gezogen wurde

# Ausgabe: Wie oft wurde jeder Datenpunkt gezogen?
#cat("\nInbag counts (Wie oft wurde jeder Datenpunkt gezogen):\n")
#print(inbag_counts)

# Bestimme die Indizes der Datenpunkte, die mindestens einmal gezogen wurden
#inbag_indices <- which(inbag_counts > 0)
#cat("\nIndices der in-Bag Datenpunkte:\n")
#print(inbag_indices)

# Ausgabe: Welche Datenpunkte wurden verwendet?
#cat("\nVerwendete in-Bag Datenpunkte:\n")
#print(train_data[inbag_indices, ])

tree_info <- treeInfo(rf_ranger, tree = 1)

# 5. Berechnung der Baumtiefe (nur die ersten beiden Ebenen)
compute_depth <- function(tree_info, max_depth = 2) {
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
  
  # Nur die ersten zwei Ebenen zurückgeben
  return(depth_list <= max_depth)
}

# 6. Berechnung der Knoten für die ersten zwei Ebenen
valid_depth_nodes <- compute_depth(tree_info, max_depth = 2)

# 7. Baum visualisieren (nur die ersten zwei Ebenen)
tree_graph <- "digraph Tree {\nnode [shape=box];\n"

leaf_count <- sum(tree_info$terminal)

for (i in 1:nrow(tree_info)) {
  if (valid_depth_nodes[i]) {  # Nur Knoten der ersten zwei Ebenen berücksichtigen
    node <- tree_info[i, ]
    
    # Falls es ein Leaf ist, füge die Anzahl der Samples hinzu
    if (node$terminal) {
      num_samples <- ifelse(node$nodeID %in% names(leaf_counts), leaf_counts[as.character(node$nodeID)], 0)
      node_label <- paste0( "samples =  ", num_samples, "\\nvalue = ", round(node$prediction, 2))
    } else {
      node_label <- paste0( "split: ", node$splitvarName, " <= ", node$splitval)
    }
    
    tree_graph <- paste0(tree_graph, node$nodeID, " [label=\"", node_label, "\"];\n")
    
    if (!is.na(node$leftChild)) {
      tree_graph <- paste0(tree_graph, node$nodeID, " -> ", node$leftChild, " [label=\"Yes\"];\n")
    }
    if (!is.na(node$rightChild)) {
      tree_graph <- paste0(tree_graph, node$nodeID, " -> ", node$rightChild, " [label=\"No\"];\n")
    }
  }
}

tree_graph <- paste0(tree_graph, "}")

# Baum plotten
grViz(tree_graph)














# Plottet den Gesamten Baum und extrahiert die Baumstruktur ---

# 4. Baumstruktur extrahieren
tree_info <- treeInfo(rf_ranger, tree = 1)

# 5. Anzahl Samples pro Leaf berechnen
pred <- predict(rf_ranger, dat_train, type = "terminalNodes")
leaf_counts <- table(pred$predictions)  # Anzahl der Samples pro Leaf-Knoten

# 6. Berechnung der Baumtiefe (analog zu sklearn)
compute_depth <- function(tree_info) {
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

tree_depth <- compute_depth(tree_info)

# 7. Berechnung der Anzahl interner Knoten
if (nrow(tree_info) == 1) {
  internal_nodes_count <- 0  # Falls der Baum nur aus einem Root-Leaf besteht
} else {
  internal_nodes_count <- sum(!tree_info$terminal)  # Alle nicht-terminalen Knoten
}

# 📌 8. Baum visualisieren mit Anzahl der Samples pro Leaf
tree_graph <- "digraph Tree {\nnode [shape=box];\n"

leaf_count <- sum(tree_info$terminal)

for (i in 1:nrow(tree_info)) {
  node <- tree_info[i, ]
  
  # Falls es ein Leaf ist, füge die Anzahl der Samples hinzu
  if (node$terminal) {
    num_samples <- ifelse(node$nodeID %in% names(leaf_counts), leaf_counts[as.character(node$nodeID)], 0)
    node_label <- paste0( "samples =  ", num_samples, "\\nvalue = ", round(node$prediction, 2))
  } else {
    node_label <- paste0( "split: ", node$splitvarName, " <= ", node$splitval)
  }
  
  tree_graph <- paste0(tree_graph, node$nodeID, " [label=\"", node_label, "\"];\n")
  
  if (!is.na(node$leftChild)) {
    tree_graph <- paste0(tree_graph, node$nodeID, " -> ", node$leftChild, " [label=\"Yes\"];\n")
  }
  if (!is.na(node$rightChild)) {
    tree_graph <- paste0(tree_graph, node$nodeID, " -> ", node$rightChild, " [label=\"No\"];\n")
  }
}

tree_graph <- paste0(tree_graph, "}")
feature_counts <- table(tree_info$splitvarName)  # Zählt, wie oft jedes Feature verwendet wurde
print("Häufigkeit der Feature-Auswahl zum Splitten:")
print(feature_counts)

cat(" Baumtiefe:", tree_depth, "\n")
cat("Anzahl interner Knoten:", internal_nodes_count, "\n")
cat(" Anzahl der Blätter:", leaf_count, "\n")

# 9. Baum plotten
grViz(tree_graph)










# =================================================================================
# Vergleich ob die Trainigsdaten in ranger und sklearn gleich sind
X_train = read.csv("/Users/sophiasiefert/Desktop/X_train.csv")
dat_train_ohne_spalten <- dat_train[, !names(dat_train) %in% c("load")]

gleich <- all(X_train$hour_int == dat_train_ohne_spalten$hour_int)
cat("Sind die Einträge in der Spalte 'hour_int' gleich? ", gleich, "\n")

gleich <- all(X_train$weekday_int == dat_train_ohne_spalten$weekday_int)
cat("Sind die Einträge in der Spalte 'weekday_int' gleich? ", gleich, "\n")

gleich <- all(X_train$holiday == dat_train_ohne_spalten$holiday)
cat("Sind die Einträge in der Spalte 'holiday' gleich? ", gleich, "\n")

gleich <- all(X_train$yearday == dat_train_ohne_spalten$yearday)
cat("Sind die Einträge in der Spalte 'yearday' gleich? ", gleich, "\n")

gleich <- all(X_train$load_lag1 == dat_train_ohne_spalten$load_lag1)
cat("Sind die Einträge in der Spalte 'load_lag1' gleich? ", gleich, "\n")
