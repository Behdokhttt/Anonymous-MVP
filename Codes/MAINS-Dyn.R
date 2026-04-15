# =========================
# MAINS-Dyn.R
# ONA-based temporal modeling for MAINS
# =========================

if (!require(readxl)) install.packages("readxl")
if (!require(ona)) install.packages("ona")
if (!require(tma)) install.packages("tma")

library(readxl)
library(ona)
library(tma)

# -------------------------
# Load data
# -------------------------
data <- read_excel("C:/Users/X/Data/Group1_Combined_coded_turn_level_format.xlsx",
                   sheet = 1)

# -------------------------
# Columns
# -------------------------
my_units <- c("Pair_ID", "Speaker")

my_codes <- c(
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

# Keep turn order within each participant stream
# If Section_ID is global and already ordered, this works well
data <- data[order(data$Pair_ID, data$Speaker, data$Section_ID), ]

# Optional metadata
metaCols <- c("Topic_Label", "Avg_Confidence")

# Ensure codes are numeric
data[, my_codes] <- lapply(data[, my_codes], as.numeric)

# -------------------------
# Context rule
# -------------------------
# Restrict context to turns that belong to the same participant stream
my_hoo_rules <- conversation_rules(
  (Pair_ID %in% UNIT$Pair_ID) &
  (Speaker %in% UNIT$Speaker)
)

# Window size for temporal ground -> response links
window_size <- 3

# -------------------------
# Accumulate ONA contexts
# -------------------------
accum.ona <-
  contexts(
    data,
    units_by = my_units,
    hoo_rules = my_hoo_rules
  ) |>
  accumulate_contexts(
    codes = my_codes,
    decay.function = decay(simple_window, window_size = window_size),
    meta.data = metaCols,
    return.ena.set = FALSE
  )

# -------------------------
# Build ONA model
# -------------------------
set.ona <- model(accum.ona)

# -------------------------
# Plot parameters
# -------------------------
node_size_multiplier <- 0.45
node_position_multiplier <- 1
edge_arrow_saturation_multiplier <- 1.5
edge_size_multiplier <- 1

# -------------------------
# Overall mean dynamic network
# -------------------------
plot(set.ona, title = "MAINS-Dyn mean network") |>
  edges(
    weights = colMeans(set.ona$line.weights$ENA_UNIT),
    edge_size_multiplier = edge_size_multiplier,
    edge_arrow_saturation_multiplier = edge_arrow_saturation_multiplier,
    node_position_multiplier = node_position_multiplier,
    edge_color = c("purple")
  ) |>
  nodes(
    node_size_multiplier = node_size_multiplier,
    node_position_multiplier = node_position_multiplier,
    self_connection_color = c("purple")
  )

# -------------------------
# Participant points
# -------------------------
plot(set.ona, title = "MAINS-Dyn participant points, mean, and CI") |>
  units(
    points = set.ona$points$ENA_UNIT,
    points_color = c("purple"),
    show_mean = TRUE,
    show_points = TRUE,
    with_ci = TRUE
  )

# -------------------------
# Mean network + participant points
# -------------------------
plot(set.ona, title = "MAINS-Dyn mean network and participant points") |>
  units(
    points = set.ona$points$ENA_UNIT,
    points_color = c("purple"),
    show_mean = TRUE,
    show_points = TRUE,
    with_ci = TRUE
  ) |>
  edges(
    weights = colMeans(set.ona$line.weights$ENA_UNIT),
    edge_size_multiplier = edge_size_multiplier,
    edge_arrow_saturation_multiplier = edge_arrow_saturation_multiplier,
    node_position_multiplier = node_position_multiplier,
    edge_color = c("purple")
  ) |>
  nodes(
    node_size_multiplier = node_size_multiplier,
    node_position_multiplier = node_position_multiplier,
    self_connection_color = c("purple")
  )

# -------------------------
# Example individual dynamic network
# -------------------------
example_unit <- names(set.ona$line.weights$ENA_UNIT)[1]

plot(set.ona, title = paste("Individual dynamic network:", example_unit)) |>
  edges(
    weights = set.ona$line.weights$ENA_UNIT[[example_unit]],
    edge_size_multiplier = edge_size_multiplier,
    edge_arrow_saturation_multiplier = edge_arrow_saturation_multiplier,
    node_position_multiplier = node_position_multiplier,
    edge_color = c("darkgreen")
  ) |>
  nodes(
    node_size_multiplier = node_size_multiplier,
    node_position_multiplier = node_position_multiplier,
    self_connection_color = c("darkgreen")
  ) |>
  units(
    points = set.ona$points$ENA_UNIT[[example_unit]],
    points_color = c("darkgreen"),
    show_mean = TRUE,
    show_points = TRUE,
    with_ci = FALSE
  )

# -------------------------
# Save useful outputs
# -------------------------
write.csv(set.ona$points, "MAINS_Dyn_points.csv", row.names = FALSE)

# Optional: variance explained
print(set.ona$model$variance)
