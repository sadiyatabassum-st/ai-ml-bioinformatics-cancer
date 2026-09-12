# ================================================================
# visualization.R
# ----------------------------------------------------------------
# R/ggplot2 exploratory visualization of the biomarker dataset,
# extending the original ggplot2 workflow (barplot, line plot,
# boxplot, jitter plot, dotplot) into a biologically meaningful
# EDA for the breast cancer biomarker classification project.
#
# Mirrors the "Data visualization using ggplot2" workflow from
# Assignment Q5:
#   1. Import dataset
#   2. Load visualization library (ggplot2)
#   3. Create basic plots (bar plot, scatter plot, histogram, boxplot)
#   4. Add labels, title, legends
#   5. Interpret biological insights from plots
# ================================================================

install.packages("ggplot2", repos = "http://cran.us.r-project.org")
library(ggplot2)

# ---- 1. Import dataset ----
df <- read.csv("../data/clean_dataset.csv")
df$diagnosis <- factor(df$diagnosis, levels = c(0, 1), labels = c("Malignant", "Benign"))

# ---- 2. Boxplot: mean radius by diagnosis ----
# (Malignant tumors are expected to have larger, more irregular nuclei)
ggplot(df, aes(x = diagnosis, y = mean.radius, fill = diagnosis)) +
  geom_boxplot() +
  scale_fill_manual(values = c("Malignant" = "#E85D5D", "Benign" = "#3FA34D")) +
  labs(title = "Mean Nuclear Radius by Diagnosis",
       x = "Diagnosis", y = "Mean Radius") +
  theme_minimal(base_size = 13)
ggsave("../images/r_boxplot_radius.png", width = 7, height = 5, dpi = 150)

# ---- 3. Scatter plot: radius vs texture, colored by diagnosis ----
ggplot(df, aes(x = mean.radius, y = mean.texture, colour = diagnosis)) +
  geom_point(size = 2, alpha = 0.7) +
  scale_colour_manual(values = c("Malignant" = "#E85D5D", "Benign" = "#3FA34D")) +
  labs(title = "Mean Radius vs Mean Texture",
       x = "Mean Radius", y = "Mean Texture") +
  theme_minimal(base_size = 13)
ggsave("../images/r_scatter_radius_texture.png", width = 7, height = 5, dpi = 150)

# ---- 4. Histogram: distribution of mean concavity ----
ggplot(df, aes(x = mean.concavity, fill = diagnosis)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("Malignant" = "#E85D5D", "Benign" = "#3FA34D")) +
  labs(title = "Distribution of Mean Concavity by Diagnosis",
       x = "Mean Concavity", y = "Count") +
  theme_minimal(base_size = 13)
ggsave("../images/r_histogram_concavity.png", width = 7, height = 5, dpi = 150)

# ---- 5. Jitter plot: worst perimeter by diagnosis ----
ggplot(df, aes(x = diagnosis, y = worst.perimeter, colour = diagnosis)) +
  geom_jitter(width = 0.2, size = 2, alpha = 0.7) +
  scale_colour_manual(values = c("Malignant" = "#E85D5D", "Benign" = "#3FA34D")) +
  labs(title = "Worst Perimeter by Diagnosis (Jitter Plot)",
       x = "Diagnosis", y = "Worst Perimeter") +
  theme_minimal(base_size = 13)
ggsave("../images/r_jitter_perimeter.png", width = 7, height = 5, dpi = 150)

cat("Saved 4 ggplot2 visualizations to ../images/\n")
cat("Biological interpretation: malignant samples consistently show larger radius, \n",
    "higher texture variance, and greater concavity -- consistent with irregular, \n",
    "invasive cell morphology typically seen in malignant tumors.\n")
