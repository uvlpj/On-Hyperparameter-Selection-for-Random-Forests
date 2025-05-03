# Code erstellt verschiedenste Simulationsdatensätze
# aus verschiedenen Verteilungen und mit verschiedenen Korrelationen
# Der Decision Tree wird dann geplottet für n = 1,....,8

rm(list = ls())
library(ranger)
library(DiagrammeR)
library(dplyr)


# ------------------------
# Simulierter Datensatz (1) ---
# Uniform Distribution 
# X1 hat eine starke Korrelation zu y
# X2 hat keine Korrelation zu y
# X1 und X2 sind nicht miteinander korreliert
rm(list = ls())
set.seed(42)
n <- 500
X <- as.data.frame(matrix(runif(n * 2), ncol = 2))  
#X <- as.data.frame(matrix(runif(n * 4), ncol = 4))  
colnames(X) <- c("X1", "X2")
#colnames(X) <- c("X1", "X2", "X3", "X4")

y <- X$X1 * 10 + sin(X$X2 * 10) + rnorm(n, 0, 0.1)
#y <- X$X1 * 10 + sin(X$X2 * 10) + sin(X$X3 * 8) + sin(X$X4 * 5) + rnorm(n, 0, 0.1)


X <- round(X, 2)
y <- round(y, 2)
data <- cbind(X, y)
c_X2_y <- cor(data$X2, data$y)
c_X1_y <- cor(data$X1, data$y)
#c_X3_y <- cor(data$X3, data$y)
#c_X4_y <- cor(data$X4, data$y)
#c_X1_X2 <- cor(data$X1, data$X2)

# 2. Ziehe genau size zufällige Trainingspunkte
train_idx <- sample(1:n, size = 3)  
train_data <- data[train_idx, ]
test_data <- data[-train_idx, ]

one_test_point <- test_data[sample(1:nrow(test_data), size = 1), ]
one_test_point

write.csv(train_data, '/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Uniform_Sin_verteilung_A/Data/train_uni_sin_n300_4.csv', row.names = FALSE)
write.csv(one_test_point, '/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Uniform_Sin_verteilung_A/Data/test_uni_sin_150.csv', row.names = FALSE)


# ------------------------
# Simulierter Datensatz (2) ---
# Multivariaten Normalverteilung
# X1 hat eine starke Korrelation zu y
# X2 hat keine Korrelation zu y
# X1 und X2 sind nicht miteinander korreliert

rm(list = ls())
library(MASS)
set.seed(42)
n <- 500
Sigma <- matrix(c(1, 0.2, 0.2, 1), nrow = 2)
#Sigma <- matrix(c(1, 0.01, 0.01, 1), nrow = 2)

X <- as.data.frame(mvrnorm(n, mu = c(0, 0), Sigma = Sigma))
colnames(X) <- c("X1", "X2")

# Erzeuge y basierend auf X1 und X2
y <- X$X1 * 10 + sin(X$X2 * 10) + rnorm(n, 0, 0.01)

X <- round(X, 2)
y <- round(y, 2)
data <- cbind(X, y)

# Berechne die Korrelation zwischen X2 und y
c_X2_y <- cor(data$X2, data$y) 
c_X1_y <- cor(data$X1, data$y)
c_X1_X2 <- cor(data$X1, data$X2)

train_idx <- sample(1:n, size = 5)  
train_data <- data[train_idx, ]
test_data <- data[-train_idx, ]

one_test_point <- test_data[sample(1:nrow(test_data), size = 1), ]
one_test_point

#write.csv(train_data, '/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Normalverteilung_sehr_starke_Korrelation_D/Daten_D/train_gaussian_n5_D.csv', row.names = FALSE)


write.csv(train_data, '/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Normalverteilung_unterschiedliche _Korrelation_B/Daten/train_gaussian_n5_B.csv', row.names = FALSE)
write.csv(one_test_point, '/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Normalverteilung_unterschiedliche _Korrelation_B/Daten/test_gaussian_B.csv', row.names = FALSE)


# ------------------------
# Simulierter Datensatz (3) ---
# Multivariate Normalverteilung
rm(list = ls())
library(MASS)
set.seed(42)
n <- 500

# Definiere die Kovarianzmatrix mit Varianz 1 und Korrelation 0.2
Sigma <- matrix(c(1, 0.2, 0.2, 1), nrow = 2)

# Multivariaten Normalverteilung mit Mittelwert 0 und der definierten Kovarianzmatrix
X <- as.data.frame(mvrnorm(n, mu = c(0, 0), Sigma = Sigma))
colnames(X) <- c("X1", "X2")

y <- 0.1*X$X1 + 2*X$X2 + rnorm(n, 0, 0.01)

X <- round(X, 2)
y <- round(y, 2)


data <- cbind(X, y)
c_X2_y <- cor(data$X2, data$y)
c_X1_y <- cor(data$X1, data$y)
c_X1_X2 <- cor(data$X1, data$X2)

train_idx <- sample(1:n, size = 5)
train_data <- data[train_idx, ]


test_data <- data[-train_idx, ]
one_test_point <- test_data[sample(1:nrow(test_data), size = 1), ]
print(one_test_point)

write.csv(train_data, '/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Normalverteilung_gleiche_Korrelation_C/Daten/train_gaussian_unterschiedlich.csv', row.names = FALSE)
write.csv(one_test_point, '/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Normalverteilung_gleiche_Korrelation_C/Daten/test_gaussian_C.csv', row.names = FALSE)



#===============================================================================
# Die Decision Trees plotten
rm(list = ls())
set.seed(42)

# Einlesen des jeweiligen Simulations-Datensaetzes
# Daten A ---
train_data <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Uniform_Sin_verteilung_A/Data/train_uni_sin_n5.csv')
test_point <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Uniform_Sin_verteilung_A/Data/test_uni_sin_one.csv')

# Daten B ---
#train_data <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Normalverteilung_unterschiedliche_Korrelation_B/Daten/train_gaussian_n5_B.csv')

# Daten C ---
#train_data <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Normalverteilung_gleiche_Korrelation_C/Daten/train_gaussian_unterschiedlich.csv')

# Daten D ---
#train_data <- read.csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/plots/Simulated_data_one_tree/Normalverteilung_sehr_starke_Korrelation_D/Daten_D/train_gaussian_n5_D.csv')

# 3. Ranger-Modell mit 1 Baum trainieren
rf_ranger <- ranger(
  y ~ ., 
  data = train_data,
  mtry = 2, #mtry = p
  num.trees = 1,  
  min.node.size = 1,
  min.bucket = 1,
  max.depth = NULL,  
  keep.inbag = TRUE,
  replace = FALSE,              # ==> Bootstrapping aktivieren (Zufallsstichprobe mit Zurücklegen)
  sample.fraction = 1     # ==> Optionale Einstellung: 100% der Daten verwenden, aber mit Bootstrapping
)

prediction <- predict(rf_ranger, data = test_point)
cat("prediction")
print(prediction$predictions) 

inbag_counts <- rf_ranger$inbag.counts[[1]]  # Anzahl, wie oft jeder Datenpunkt gezogen wurde

# Ausgabe: Wie oft wurde jeder Datenpunkt gezogen?
cat("\nInbag counts (Wie oft wurde jeder Datenpunkt gezogen):\n")
print(inbag_counts)

# Bestimme die Indizes der Datenpunkte, die mindestens einmal gezogen wurden
inbag_indices <- which(inbag_counts > 0)
cat("\nIndices der in-Bag Datenpunkte:\n")
print(inbag_indices)

# Ausgabe: Welche Datenpunkte wurden verwendet?
cat("\nVerwendete in-Bag Datenpunkte:\n")
print(train_data[inbag_indices, ])


# 4. Baumstruktur extrahieren
tree_info <- treeInfo(rf_ranger, tree = 1)

# 5. Anzahl Samples pro Leaf berechnen
pred <- predict(rf_ranger, train_data, type = "terminalNodes")
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

# 8. Baum visualisieren mit Anzahl der Samples pro Leaf
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

cat("📏 Baumtiefe:", tree_depth, "\n")
cat("🔢 Anzahl interner Knoten:", internal_nodes_count, "\n")
cat("🌿 Anzahl der Blätter:", leaf_count, "\n")

# 9. Baum plotten
grViz(tree_graph)



