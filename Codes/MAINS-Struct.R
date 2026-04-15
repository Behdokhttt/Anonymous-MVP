# =========================
# MAINS-Struct.R
# ENA-based co-occurrence modeling for MAINS
# =========================

if (!require(readxl)) install.packages("readxl")
if (!require(rENA)) install.packages("rENA")
if (!require(tma)) install.packages("tma")

library(readxl)
library(rENA)
library(tma)

# -------------------------
# Load data
# -------------------------
data <- read_excel("C:/Users/X/Data/Group1_Combined_coded_turn_level_format.xlsx",
                   sheet = 1)

# -------------------------
# Columns
# -------------------------
unitCols <- c("Pair_ID", "Speaker")
codeCols <- c(
  "Engagement",
  "Self_Disclosure",
  "Agreement",
  "Interpersonal_Coordination",
  "Alignment",
  "Breakdown",
  "Positive",
  "Neutral",
  "Negative"
)

# Use stanza as the local conversational context
conversationCols <- c("Stanza_ID")

# Optional metadata to keep in the ENA set
metaCols <- c("Topic_Label", "Avg_Confidence")

data[, codeCols] <- lapply(data[, codeCols], as.numeric)

# Create unique ENA unit label
data$ENA_UNIT <- paste(data$Pair_ID, data$Speaker, sep = ".")

# -------------------------
# Accumulate ENA data
# -------------------------
# window.size.back = 0 lets co-occurrence be defined by the stanza grouping
# rather than a moving temporal window
accum.ena <- ena.accumulate.data(
  text_data = data[, "Raw_Data", drop = FALSE],
  units = data[, unitCols, drop = FALSE],
  conversation = data[, conversationCols, drop = FALSE],
  codes = data[, codeCols, drop = FALSE],
  meta.data = data[, metaCols, drop = FALSE],
  window.size.back = 0
)

# -------------------------
# Build ENA model
# -------------------------
set.ena <- ena.make.set(
  enadata = accum.ena,
  rotation.by = ena.rotate.by.mean
)

# -------------------------
# Overall mean network
# -------------------------
overall.lineweights <- as.matrix(set.ena$line.weights$ENA_UNIT)
overall.mean <- as.vector(colMeans(overall.lineweights))

ena.plot(set.ena, title = "MAINS-Struct mean network") |>
  ena.plot.network(network = overall.mean, colors = c("purple"))

# -------------------------
# Participant points
# -------------------------
overall.points <- as.matrix(set.ena$points$ENA_UNIT)

ena.plot(set.ena, title = "MAINS-Struct participant points, mean, and CI") |>
  ena.plot.points(points = overall.points, colors = c("purple")) |>
  ena.plot.group(point = overall.points,
                 colors = c("purple"),
                 confidence.interval = "box")

# -------------------------
# Mean network + participant points
# -------------------------
ena.plot(set.ena, title = "MAINS-Struct mean network and participant points") |>
  ena.plot.network(network = overall.mean, colors = c("purple")) |>
  ena.plot.points(points = overall.points, colors = c("purple")) |>
  ena.plot.group(point = overall.points,
                 colors = c("purple"),
                 confidence.interval = "box")

# -------------------------
# Example individual network
# -------------------------
example_unit <- rownames(set.ena$points$ENA_UNIT)[1]

unit.line.weights <- as.matrix(set.ena$line.weights$ENA_UNIT[[example_unit]])
unit.point <- as.matrix(set.ena$points$ENA_UNIT[[example_unit]])

ena.plot(set.ena, title = paste("Individual network:", example_unit)) |>
  ena.plot.network(network = unit.line.weights, colors = c("darkgreen")) |>
  ena.plot.points(points = unit.point, colors = c("darkgreen"))

# -------------------------
write.csv(set.ena$points, "MAINS_Struct_points.csv", row.names = FALSE)

# Optional: inspect variance explained
print(set.ena$model$variance)

# Optional: correlations with metadata if needed
# ena.correlations(set.ena)
