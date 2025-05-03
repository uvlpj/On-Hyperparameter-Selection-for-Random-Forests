# Code für Combined Forecasts---
#     die jeweiligen 100 quantile werden jeweils mit gewicht 0.25 aus den vier verschiedenne Modellen kombiniert

library(ranger)
library(dplyr)
library(scoringRules)
library(quantregForest)
library(knitr)



config <- list(
  data_file = "/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/Data/rf_data_1823_clean.csv",
  n_trees = 100,
  training_size = 35056,  # size of training sample (2018-2021)
  n_quantiles = 1e2
)

dat <- read.csv(config$data_file) %>%
  mutate(date = as.Date(date)) %>%
  arrange(date) %>%
  mutate(load_lag1 = lag(load, 1)) %>%
  na.omit()

# Split data
dat_train <- dat[1:config$training_size, ]
dat_test <- dat[(config$training_size + 1):nrow(dat), ] %>% select(-load)
y_train <- as.vector(dat_train$load)
y_test <- dat[(config$training_size + 1):nrow(dat), ]$load

# Laden der Quantil-Vorhersagen

# quantregForest ===> nott_day_lagged
quantregForest_nott_day_lagged_mtry1 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_nott_day_lagged_mtry1.csv')
quantregForest_nott_day_lagged_mtry2 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_nott_day_lagged_mtry2.csv')
quantregForest_nott_day_lagged_mtry3 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_nott_day_lagged_mtry3.csv')
quantregForest_nott_day_lagged_mtry4 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_nott_day_lagged_mtry4.csv')
quantregForest_nott_day_lagged_mtry5 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_nott_day_lagged_mtry5.csv')


# quantregForest ===> nott_month_lagged
quantregForest_nott_month_lagged_mtry1 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_nott_month_lagged_mtry1.csv')
quantregForest_nott_month_lagged_mtry2 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_nott_month_lagged_mtry2.csv')
quantregForest_nott_month_lagged_mtry3 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_nott_month_lagged_mtry3.csv')
quantregForest_nott_month_lagged_mtry4 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_nott_month_lagged_mtry4.csv')
quantregForest_nott_month_lagged_mtry5 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_nott_month_lagged_mtry5.csv')


# quantregForest ===> tt_day_lagged
quantregForest_tt_day_lagged_mtry1 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_tt_day_lagged_mtry1.csv')
quantregForest_tt_day_lagged_mtry2 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_tt_day_lagged_mtry2.csv')
quantregForest_tt_day_lagged_mtry3 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_tt_day_lagged_mtry3.csv')
quantregForest_tt_day_lagged_mtry4 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_tt_day_lagged_mtry4.csv')
quantregForest_tt_day_lagged_mtry5 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_tt_day_lagged_mtry5.csv')
quantregForest_tt_day_lagged_mtry6 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_tt_day_lagged_mtry6.csv')



# quantregForest ===> tt_month_lagged
quantregForest_tt_month_lagged_mtry1 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_tt_month_lagged_mtry1.csv')
quantregForest_tt_month_lagged_mtry2 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_tt_month_lagged_mtry2.csv')
quantregForest_tt_month_lagged_mtry3 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_tt_month_lagged_mtry3.csv')
quantregForest_tt_month_lagged_mtry4 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_tt_month_lagged_mtry4.csv')
quantregForest_tt_month_lagged_mtry5 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_tt_month_lagged_mtry5.csv')
quantregForest_tt_month_lagged_mtry6 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/quantregForest_tt_month_lagged_mtry6.csv')



# ============================================

# ranger ===> nott_day_lagged
ranger_nott_day_lagged_mtry1 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_nott_day_lagged_mtry1.csv')
ranger_nott_day_lagged_mtry2 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_nott_day_lagged_mtry2.csv')
ranger_nott_day_lagged_mtry3 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_nott_day_lagged_mtry3.csv')
ranger_nott_day_lagged_mtry4 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_nott_day_lagged_mtry4.csv')
ranger_nott_day_lagged_mtry5 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_nott_day_lagged_mtry5.csv')


# quantregForest ===> nott_month_lagged
ranger_nott_month_lagged_mtry1 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_nott_month_lagged_mtry1.csv')
ranger_nott_month_lagged_mtry2 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_nott_month_lagged_mtry2.csv')
ranger_nott_month_lagged_mtry3 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_nott_month_lagged_mtry3.csv')
ranger_nott_month_lagged_mtry4 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_nott_month_lagged_mtry4.csv')
ranger_nott_month_lagged_mtry5 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_nott_month_lagged_mtry5.csv')


# quantregForest ===> tt_day_lagged
ranger_tt_day_lagged_mtry1 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_tt_day_lagged_mtry1.csv')
ranger_tt_day_lagged_mtry2 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_tt_day_lagged_mtry2.csv')
ranger_tt_day_lagged_mtry3 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_tt_day_lagged_mtry3.csv')
ranger_tt_day_lagged_mtry4 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_tt_day_lagged_mtry4.csv')
ranger_tt_day_lagged_mtry5 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_tt_day_lagged_mtry5.csv')
ranger_tt_day_lagged_mtry6 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_tt_day_lagged_mtry6.csv')



# quantregForest ===> tt_month_lagged
ranger_tt_month_lagged_mtry1 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_tt_month_lagged_mtry1.csv')
ranger_tt_month_lagged_mtry2 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_tt_month_lagged_mtry2.csv')
ranger_tt_month_lagged_mtry3 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_tt_month_lagged_mtry3.csv')
ranger_tt_month_lagged_mtry4 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_tt_month_lagged_mtry4.csv')
ranger_tt_month_lagged_mtry5 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_tt_month_lagged_mtry5.csv')
ranger_tt_month_lagged_mtry6 <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code/res_different_mtry_a1_v4/quantile_pred/ranger_tt_month_lagged_mtry6.csv')







df1 <- ranger_nott_day_lagged_mtry1 %>% select(-date)
df2 <- ranger_tt_day_lagged_mtry1 %>% select(-date)
df3 <- ranger_nott_month_lagged_mtry1 %>% select(-date)
df4 <- ranger_tt_month_lagged_mtry1 %>% select(-date)



quantiles_combined <- 0.25 * df1 + 0.25 * df2 + 0.25 * df3 + 0.25 * df4

quantiles_combined <- as.matrix(quantiles_combined)

res <- data.frame(date = dat_test$date)
# Berechnung des CRPS für jedes Testbeispiel basierend auf der gewichteten Verteilung
crps_combined <- numeric(nrow(dat_test))  # Leerer Vektor für CRPS-Ergebnisse

for (jj in 1:nrow(dat_test)) {
  # Für jedes Testbeispiel (jj) berechne den CRPS für die gewichteten Quantilvorhersagen
  crps_combined[jj] <- crps_sample(y = y_test[jj], dat = quantiles_combined[jj, ])
}

# Speichern der CRPS-Ergebnisse
res$crps_combined <- crps_combined

mean_crps_combined <- mean(crps_combined)




setwd("/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code")
if (!dir.exists("res_combined_crps")) {
  dir.create("res_combined_crps/")
}

# Kombinierte Daten als CSV speichern
write.csv(res, file = "res_combined_crps/ranger_mtry1.csv", row.names = FALSE)






res_2 <- data.frame(date = dat_test$date)
crps_model1_vec <- numeric(nrow(dat_test))
crps_model2_vec <- numeric(nrow(dat_test))
crps_model3_vec <- numeric(nrow(dat_test))
crps_model4_vec <- numeric(nrow(dat_test))

# Schleife über jedes Testbeispiel
for (jj in 1:nrow(dat_test)) {
  
  # Berechnung des CRPS für jedes Testbeispiel basierend auf den vier Modellen
  crps_model1_vec[jj] <- crps_sample(y = y_test[jj], dat = as.numeric(df1[jj, ]))
  crps_model2_vec[jj] <- crps_sample(y = y_test[jj], dat = as.numeric(df2[jj, ]))
  crps_model3_vec[jj] <- crps_sample(y = y_test[jj], dat = as.numeric(df3[jj, ]))
  crps_model4_vec[jj] <- crps_sample(y = y_test[jj], dat = as.numeric(df4[jj, ]))
}

# Kontrolle: die ersten paar Werte ausgeben
head(crps_model1_vec)
head(crps_model2_vec)
head(crps_model3_vec)
head(crps_model4_vec)


mean_crps_model1 <- mean(crps_model1_vec, na.rm = TRUE)
mean_crps_model2 <- mean(crps_model2_vec, na.rm = TRUE)
mean_crps_model3 <- mean(crps_model3_vec, na.rm = TRUE)
mean_crps_model4 <- mean(crps_model4_vec, na.rm = TRUE)




library(ggplot2)

# Erstelle einen DataFrame mit den CRPS-Werten
data <- data.frame(
  Model = rep(c("Model 1", "Model 2", "Model 3", "Model 4", "Combined Model"), each = nrow(dat_test)),
  CRPS = c(crps_model1_vec, crps_model2_vec, crps_model3_vec, crps_model4_vec, crps_combined),
  Type = rep(c("Individual", "Individual", "Individual", "Individual", "Combined"), each = nrow(dat_test))
)

# Erstelle den Boxplot
ggplot(data, aes(x = Model, y = CRPS, fill = Model)) +
  geom_boxplot() +
  ggtitle("CRPS Comparison of Individual Models vs Combined Model") +
  ylab("CRPS") +
  theme_minimal()




# Überprüfung des ≤-Kriteriums für den Vergleich
if (mean_crps_combined <= mean(mean_crps_model1, mean_crps_model2, mean_crps_model3, mean_crps_model4)) {
  print("Das ≤-Kriterium ist erfüllt.")
} else {
  print("Das ≤-Kriterium wurde verletzt!")
}



