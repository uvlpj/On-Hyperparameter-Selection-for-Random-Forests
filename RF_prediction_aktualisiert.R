# Der Code simuliert die Ausgangssituation
# Zwei Schritte werden Simuliert 
# Schritt 1:
#        => Hyperparameter sind nicht angepasst und in quantregForest ist die Konstante vorhanden 
# Schritt 2:
#        => Hyperparameter sind angepasst und in quantregForest ist die Konstante entfernt wurden


# Load libraries
library(ranger)
library(dplyr)
library(scoringRules)
library(quantregForest)
library(knitr)

set.seed(2024)

# Configuration Parameters
config <- list(
  working_directory = "/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code",
  output_directory = "res_rf_bt_a1/",
  data_file = "/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/Data/rf_data_1823_clean.csv",
  n_trees = 100,
  training_size = 35056,  # size of training sample (2018-2021)
  n_quantiles = 1e2
)

# Parameter combinations
grid <- expand.grid(
  which_package = c("ranger", "quantregForest"),
  bagged_trees = c(TRUE, FALSE),
  time_trend = c(TRUE, FALSE),
  day_of_year = c(TRUE, FALSE),
  load_lag1 = c(TRUE, FALSE)
)

# Set working directory
setwd(config$working_directory)

# Create output directory if not exists
if (!dir.exists(config$output_directory)) {
  dir.create(config$output_directory)
}

# Load and preprocess data
dat <- read.csv(config$data_file) %>%
  mutate(date = as.Date(date)) %>%
  arrange(date) %>%
  mutate(load_lag1 = lag(load, 1)) %>%
  na.omit()

# Split data
dat_train <- dat[1:config$training_size, ]
#dat_test <- dat[(config$training_size + 1):nrow(dat), ] %>% select(-load)
#dat_test <- dat_test[, !names(dat_test) %in% "load"]
dat_test <- dat[(config$training_size + 1):nrow(dat), ]
dat_test <- dat_test[, !names(dat_test) %in% "load"]

y_train <- as.vector(dat_train$load)
y_test <- dat[(config$training_size + 1):nrow(dat), ]$load

# show range of testing sample
range(dat_test$date)
# show dimension of testing sample (16449, 15)
dim(dat_test)
dim(dat_train)

# Loop over parameter combinations
results <- list()

for (i in 1:nrow(grid)) {
  params <- grid[i, ]
  
  # Construct formula
  fml <- as.formula("load ~ holiday + hour_int + weekday_int")
  if (params$day_of_year) {
    fml <- update(fml, . ~ . + yearday)
  } else {
    fml <- update(fml, . ~ . + month_int)
  }
  if (params$time_trend) {
    fml <- update(fml, . ~ . + time_trend)
  }
  if (params$load_lag1) {
    fml <- update(fml, . ~ . + load_lag1)
  }
  
  fml_no_intercept <- update(fml, . ~ . - 1)
  x_train <- model.matrix(fml_no_intercept, data = dat_train) %>% as.matrix()
  x_test <- model.matrix(fml_no_intercept, data = data.frame(load = y_test, dat_test)) %>% as.matrix()
  
  # Ausgabe der Matrizen x_train und x_test in der Konsole
  #cat("\n--- x_train (Iteration", i, ") ---\n")
  #print(head(x_train))  # Ausgabe der ersten Zeilen von x_train
  #cat("\n--- x_test (Iteration", i, ") ---\n")
  #print(head(x_test))
  # Fit model
  fit <- NULL
  pred <- NULL
  print(dat_train)
  
  if (params$which_package == "ranger") {
    m_try <- ifelse(params$bagged_trees, identity, function(d) floor(d / 3))
    cat("\n--- mtry (Iteration", i, ") ---\n")
    if (is.function(m_try)) {
      # Berechne m_try mit einer Beispielgröße, z.B. der Anzahl der Spalten in x_train
      m_try_value <- m_try(ncol(x_train))  # Beispiel mit ncol(x_train) als d
      print(m_try_value)  # Gibt den berechneten Wert aus
    } else {
      print(m_try)  # Falls m_try eine Zahl ist, wird diese direkt ausgegeben
    }
    fit <- ranger(
      fml, data = dat_train,
      mtry = m_try,
      max.depth = NULL,
      min.node.size = 1,
      min.bucket = 1,
      num.trees = config$n_trees,
      quantreg = TRUE,
      keep.inbag = TRUE )
    
    cat("\n--- mtry ranger ---\n")
    print(m_try)
    cat("\n--- dat_train ranger ---\n")
    print(dat_train[, 1:5]) 
    
    pred <- predict(
      fit, type = "quantiles",
      quantiles = (2 * (1:config$n_quantiles) - 1) / (2 * config$n_quantiles),
      data = dat_test
    )$predictions
  } else {
    x_train <- model.matrix(fml_no_intercept, data = dat_train) %>% as.matrix
    cat("\n--- dat_train quantregForest ---\n")
    print(x_train[1:5,]) 
    x_test <- model.matrix(fml_no_intercept, data = data.frame(load = y_test, dat_test)) %>% as.matrix
    print(x_test)
    m_try <- ifelse(params$bagged_trees, ncol(x_train), floor(ncol(x_train) / 3))
    fit <- quantregForest(
      x = x_train, 
      y = y_train, 
      nodesize = 2,
      max_depth = NULL,
      ntree = config$n_trees,
      mtry = m_try
    )
    pred <- predict(
      fit, what = (2 * (1:config$n_quantiles) - 1) / (2 * config$n_quantiles),
      newdata = x_test
    )
  }
  
  # Evaluate performance
  res <- data.frame(
    date_time = dat_test$date_time,
    crps = NA, ae = NA, se = NA, 
    median_pred = NA, mean_pred = NA,
    year = dat_test$year
  )
  for (jj in 1:nrow(dat_test)) {
    res$crps[jj] <- crps_sample(y = y_test[jj], dat = pred[jj, ])
    res$ae[jj] <- abs(y_test[jj] - median(pred[jj, ]))
    res$se[jj] <- (y_test[jj] - mean(pred[jj, ]))^2
    res$median_pred[jj] <- median(pred[jj, ])
    res$mean_pred[jj] <- mean(pred[jj, ])
  }
  
  # Summarize results
  overall_results <- res %>%
    summarise(
      mean_crps = mean(crps, na.rm = TRUE),
      mean_ae = mean(ae, na.rm = TRUE),
      mean_se = mean(se, na.rm = TRUE),
      root_mse = sqrt(mean_se)
    )
  
  # Save results
  save_name <- paste0(
    config$output_directory, params$which_package, "_",
    if (params$time_trend) "tt_" else "nott_", 
    if (params$day_of_year) "day_" else "month_", 
    if (params$load_lag1) "lagged_" else "notlagged_", 
    if (params$bagged_trees) "bt.csv" else "rf.csv"
  )
  write.table(res, file = save_name, sep = ",", row.names = FALSE)
  
  # Store overall results
  results[[i]] <- cbind(params, overall_results)
}

# Combine all results
final_results <- do.call(rbind, results)
print(final_results)
