# Code für Gesetz der Großen Zahlen
# Trainiert RF mit increasing number of estimators 
# zusätzlich BT und RF mit influential feature vergleichen


rm(list = ls())
library(MASS) 
library(ranger)
library(dplyr)
set.seed(42)



# Daten B ---
train_data <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Normalverteilung_unterschiedliche_Korrelation_B/Daten/train_gaussian_n300_B.csv')
test_data <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Normalverteilung_unterschiedliche_Korrelation_B/Daten/test_gaussian_B.csv')

# Daten C ---
train_data <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Normalverteilung_gleiche_Korrelation_C/Daten/train_gaussian_n300_C.csv')
test_data <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Normalverteilung_gleiche_Korrelation_C/Daten/test_gaussian_C.csv')

# Daten D ---
# Uni mit 4 Features, wobei nur eine eine starke korrelation mit y hat die anderen 3 Features haben keine wirkliche Information
# 350 train daten
# 150 test daten
train_data <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Uniform_Sin_verteilung_A/Data/train_uni_sin_n300_4.csv')
test_data <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Uniform_Sin_verteilung_A/Data/test_uni_sin_4.csv')
#test_data <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Uniform_Sin_verteilung_A/Data/test_uni_sin_150.csv')


# ========================================
# Ranger ----
results <- data.frame()

# 3. Iteriere über 1 bis 100 Bäume
for (num_trees in 1:500) {
   
  # Trainiere das Ranger-Modell
  rf_ranger <- ranger(
    y ~ ., 
    data = train_data,
    mtry = 4,
    num.trees = num_trees,  
    min.node.size = 1,
    min.bucket = 1,
    max.depth = NULL,  
    keep.inbag = TRUE,
    replace = TRUE,
    sample.fraction = 1
  )
  
  # Mache eine Vorhersage auf den Testdaten
  predictions <- predict(rf_ranger, test_data)$predictions
  
  # Speichere die Ergebnisse in einem DataFrame
  iter_results <- data.frame(
    num_trees = num_trees,
    X1 = test_data$X1,
    X2 = test_data$X2,
    y_actual = test_data$y,
    y_predicted = round(predictions, 2)
  )
  
  # Ergebnisse anhängen
  results <- rbind(results, iter_results)
}


# Speichere Predictions mit Daten B ---
write.csv(results, '/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Normalverteilung_unterschiedliche_Korrelation_B/Daten/predictions_ranger.csv', row.names = FALSE)
 
# Speichere Predictions mit Daten C ---
write.csv(results, '/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Normalverteilung_gleiche_Korrelation_C/Daten/predictions_ranger_C.csv', row.names = FALSE)

# Speichere Predictions mit Daten D ---
write.csv(results, '/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Prediction_D/prediction_ranger_BT.csv', row.names = FALSE)

#=====================================================================================
# Comparing BT vs RF
rf_ranger <- ranger(
  y ~ ., 
  data = train_data,
  mtry = 1,
  num.trees = 100,  
  min.node.size = 1,
  min.bucket = 1,
  max.depth = NULL,  
  keep.inbag = TRUE,
  replace = TRUE,
  sample.fraction = 1
)

predictions <- predict(rf_ranger, test_data)$predictions

results <- data.frame(
  y_true = test_data$y,       # Tatsächliche Werte
  y_pred = predictions        # Vorhergesagte Werte
)

write.csv(results, '/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Prediction_D/prediction_ranger_RF.csv', row.names = FALSE)


# ================================================================================
library(quantregForest)
set.seed(42)
# quantregForest ----
results <- data.frame()



# Daten A ---
train_data <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Uniform_Sin_verteilung_A/Data/train_uni_sin_n300.csv')
test_data <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Uniform_Sin_verteilung_A/Data/test_uni_sin.csv')

# Daten B ---
train_data <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Normalverteilung_unterschiedliche_Korrelation_B/Daten/train_gaussian_n300_B.csv')
test_data <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Normalverteilung_unterschiedliche_Korrelation_B/Daten/test_gaussian_B.csv')

# Daten C ---
train_data <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Normalverteilung_gleiche_Korrelation_C/Daten/train_gaussian_n300_C.csv')
test_data <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Normalverteilung_gleiche_Korrelation_C/Daten/test_gaussian_C.csv')

# Daten D ---
train_data <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Uniform_Sin_verteilung_A/Data/train_uni_sin_n300_4.csv')
test_data <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Uniform_Sin_verteilung_A/Data/test_uni_sin_4.csv')
test_data <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Uniform_Sin_verteilung_A/Data/test_uni_sin_150.csv')






# Extrahiere X und y für quantregForest
x_train <- train_data[, !(names(train_data) %in% "y")]  # Alle Spalten außer "y"
y_train <- train_data$y

x_test <- test_data[, !(names(test_data) %in% "y")]

# 📌 3. Iteriere über 1 bis 500 Bäume
for (num_trees in 1:500) {
  
  # Trainiere das quantregForest-Modell
  rf_qrf <- quantregForest(
    x = x_train,
    y = y_train,
    mtry = 1,
    ntree = num_trees,  # Anzahl Bäume
    nodesize = 1        # Kleinste Knotengröße, analog zu min.node.size
  )
  
  # Mache eine Vorhersage auf den Testdaten (conditional mean)
  predictions <- predict(rf_qrf, x_test, what = mean)  # Conditional Mean
  
  # Speichere die Ergebnisse in einem DataFrame
  iter_results <- data.frame(
    num_trees = num_trees,
    X1 = test_data$X1,
    X2 = test_data$X2,
    y_actual = test_data$y,
    y_predicted = round(predictions, 2)
  )
  
  # Ergebnisse anhängen
  results <- rbind(results, iter_results)
}


# ===============================
# Comparing BT vs RF

x_train <- train_data[, !(names(train_data) %in% "y")]  # Alle Spalten außer "y"
y_train <- train_data$y

x_test <- test_data[, !(names(test_data) %in% "y")]

# 📌 3. Iteriere über 1 bis 500 Bäume

  
  # Trainiere das quantregForest-Modell
rf_qrf <- quantregForest(
    x = x_train,
    y = y_train,
    mtry = 1,
    ntree = 100,  # Anzahl Bäume
    nodesize = 2        # Kleinste Knotengröße, analog zu min.node.size
)

  # Mache eine Vorhersage auf den Testdaten (conditional mean)
predictions <- predict(rf_qrf, x_test, what = mean)  # Conditional Mean
  
results <- data.frame(
     y_true = test_data$y,       # Tatsächliche Werte
     y_pred = predictions        # Vorhergesagte Werte
   )


write.csv(results, '/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Prediction_D/prediction_quantregForest_RF.csv', row.names = FALSE)




# Speichern der Vorhersagen in einer CSV-Datei
# Daten A ---
write.csv(results, '/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Uniform_Sin_verteilung_A/Data/predictions_quantregForest.csv', row.names = FALSE)

# Daten B ---
write.csv(results, '/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Normalverteilung_unterschiedliche_Korrelation_B/Daten/predictions_quantregForest.csv', row.names = FALSE)

# Speichere Predictions mit Daten C ---
write.csv(results, '/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Normalverteilung_gleiche_Korrelation_C/Daten/predictions_quantregForest_C.csv', row.names = FALSE)

# Speichere Predictions mit Daten D ---
write.csv(results, '/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Prediction_D/prediction_quantregForest.csv', row.names = FALSE)






# Ranger
bt <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Prediction_D/prediction_ranger_BT.csv')
rf <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Prediction_D/prediction_ranger_RF.csv')

# quantregForest
bt <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Prediction_D/prediction_quantregForest_BT.csv')
rf <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Prediction_D/prediction_quantregForest_RF.csv')




mse <- function(y_true, y_pred) {
  mean((y_true - y_pred)^2)
}

# MSE für Boosting Trees
mse_bt <- mse(bt$y_true, bt$y_pred)
print(paste("MSE BT:", mse_bt))

# MSE für Random Forest
mse_rf <- mse(rf$y_true, rf$y_pred)
print(paste("MSE RF:", mse_rf))


