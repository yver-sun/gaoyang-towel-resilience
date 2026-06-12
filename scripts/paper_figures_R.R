# Paper figures: R ggplot2 publication-quality plots
.libPaths(c("C:/Users/Yver/R/library", .libPaths()))
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(scales)
library(patchwork)
library(reshape2)

setwd("c:/Users/Yver/Desktop/史岩林/高阳毛巾")
dir.create("output/figures_R", showWarnings = FALSE, recursive = TRUE)

theme_paper <- theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.title = element_text(size = 11),
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    plot.margin = margin(10, 15, 10, 10)
  )

# Load data
mp <- read_csv("output/master_panel_data_v2.csv", show_col_types = FALSE)
ps <- read_csv("output/policy_scores_panel.csv", show_col_types = FALSE)
scm <- read_csv("analysis/scm_results.csv", show_col_types = FALSE)
ti <- read_csv("output/textile_indices_annual.csv", show_col_types = FALSE)
ti_sub <- read_csv("output/gaoyang_textile_registration.csv", show_col_types = FALSE)

gy_ps <- ps %>% filter(as.character(County_Code) == "130628")
m <- mp %>%
  select(Year, textile_firms) %>%
  inner_join(gy_ps %>% select(Year, policy_intensity_total), by = "Year")

# ====== Fig1: Dual-axis time series ======
cat("Fig1: Time series...\n")
coeff <- max(m$textile_firms, na.rm = TRUE) / max(m$policy_intensity_total, na.rm = TRUE)

p1 <- ggplot(m, aes(x = Year)) +
  geom_bar(aes(y = textile_firms), stat = "identity", fill = "#2c3e50", alpha = 0.85, width = 0.7) +
  geom_line(aes(y = policy_intensity_total * coeff), color = "#e74c3c", linewidth = 1.2) +
  geom_point(aes(y = policy_intensity_total * coeff), color = "#e74c3c", size = 2.5) +
  geom_area(aes(y = policy_intensity_total * coeff), fill = "#e74c3c", alpha = 0.08) +
  geom_vline(xintercept = 2017, linetype = "dashed", color = "gray50", linewidth = 0.8) +
  annotate("text", x = 2017.5, y = max(m$textile_firms) * 0.95,
           label = "2017 Environmental\nInspection", hjust = 0, size = 3.5, color = "gray40") +
  scale_y_continuous(
    name = "Textile Firms (Annual Registration)",
    sec.axis = sec_axis(~ . / coeff, name = "LLM Policy Intensity Score")
  ) +
  labs(title = "Gaoyang County: Textile Firm Registrations and LLM Policy Intensity (2000-2026)",
       x = "Year") +
  theme_paper +
  theme(axis.title.y.right = element_text(color = "#e74c3c"),
        axis.text.y.right = element_text(color = "#e74c3c"),
        axis.title.y.left = element_text(color = "#2c3e50"),
        axis.text.y.left = element_text(color = "#2c3e50"))

ggsave("output/figures_R/fig1_timeseries.png", p1, width = 10, height = 5.5, dpi = 300)

# ====== Fig2: SCM gap ======
cat("Fig2: SCM...\n")
p2a <- ggplot(scm, aes(x = Year)) +
  geom_line(aes(y = Gaoyang_actual, color = "Gaoyang (actual)"), linewidth = 1.2) +
  geom_line(aes(y = Synthetic_Gaoyang, color = "Synthetic Gaoyang"), linewidth = 1.2, linetype = "dashed") +
  geom_vline(xintercept = 2017, linetype = "dashed", color = "gray50", linewidth = 0.8) +
  annotate("text", x = 2017.5, y = max(scm$Gaoyang_actual) * 0.95,
           label = "2017", hjust = 0, size = 3.5, color = "gray40") +
  scale_color_manual(values = c("Gaoyang (actual)" = "#2c3e50", "Synthetic Gaoyang" = "#e74c3c")) +
  labs(y = "Textile Firms", color = "") +
  theme_paper

scm <- scm %>% mutate(gap_color = ifelse(Gap >= 0, "Positive", "Negative"))
p2b <- ggplot(scm, aes(x = Year, y = Gap, fill = gap_color)) +
  geom_bar(stat = "identity", alpha = 0.85, width = 0.7) +
  geom_hline(yintercept = 0, linewidth = 0.4) +
  geom_vline(xintercept = 2017, linetype = "dashed", color = "gray50", linewidth = 0.8) +
  scale_fill_manual(values = c("Positive" = "#3498db", "Negative" = "#e74c3c")) +
  labs(y = "Gap (Actual - Synthetic)", x = "Year", fill = "") +
  theme_paper

p2 <- p2a / p2b + plot_annotation(
  title = "Synthetic Control: Gaoyang vs Synthetic Counterpart (Li County)"
)
ggsave("output/figures_R/fig2_scm.png", p2, width = 10, height = 7, dpi = 300)

# ====== Fig3: Coefficient comparison ======
cat("Fig3: Coefficient comparison...\n")
dims <- c("policy_intensity_total", "equipment_index", "environment_index",
          "ecommerce_index", "brandquality_index", "cluster_index", "finance_index", "education_index")
labels <- c("Total Policy", "Equipment", "Environment", "Ecommerce",
            "Brand Quality", "Cluster", "Finance", "Education (placebo)")

m2 <- mp %>%
  select(Year, textile_firms) %>%
  inner_join(gy_ps %>% select(Year, all_of(dims)), by = "Year") %>%
  filter(Year <= 2024) %>%
  mutate(time = Year - 2000)

get_coefs <- function(data, dims, include_time) {
  results <- data.frame()
  for (i in seq_along(dims)) {
    y <- data$textile_firms
    if (include_time) {
      fit <- lm(y ~ data[[dims[i]]] + data$time)
    } else {
      fit <- lm(y ~ data[[dims[i]]])
    }
    s <- summary(fit)$coefficients
    results <- rbind(results, data.frame(
      dim = dims[i], label = labels[i],
      coef = s[2, 1], se = s[2, 2],
      ci_low = s[2, 1] - 1.96 * s[2, 2],
      ci_high = s[2, 1] + 1.96 * s[2, 2],
      significant = s[2, 4] < 0.05
    ))
  }
  results
}

coefs_no_time <- get_coefs(m2, dims, FALSE)
coefs_with_time <- get_coefs(m2, dims, TRUE)

plot_coefs <- function(df, title) {
  df$label <- factor(df$label, levels = rev(labels))
  ggplot(df, aes(x = coef, y = label, fill = ci_low > 0)) +
    geom_bar(stat = "identity", alpha = 0.85, width = 0.6) +
    geom_errorbar(aes(xmin = ci_low, xmax = ci_high), width = 0.2) +
    geom_vline(xintercept = 0, linewidth = 0.4) +
    scale_fill_manual(values = c("TRUE" = "#2c3e50", "FALSE" = "#bdc3c7"), guide = "none") +
    labs(x = "Coefficient (firms per unit)", y = "", title = title) +
    theme_paper
}

p3 <- plot_coefs(coefs_no_time, "Without Time Control (Spurious)") +
  plot_coefs(coefs_with_time, "With Time Control (Correct Specification)") +
  plot_annotation(title = "Policy Dimension Effects: The Danger of Omitted Time Trend")
ggsave("output/figures_R/fig3_model_comparison.png", p3, width = 12, height = 5.5, dpi = 300)

# ====== Fig4: Correlation heatmap ======
cat("Fig4: Correlation heatmap...\n")
corr_cols <- c("textile_firms", "total_firms", "policy_intensity_total",
               "equipment_index", "environment_index", "ecommerce_index",
               "brandquality_index", "cluster_index", "finance_index", "education_index", "time")
corr_labels <- c("Textile Firms", "Total Firms", "Policy Total", "Equipment", "Environment",
                 "Ecommerce", "Brand Quality", "Cluster", "Finance", "Education", "Time Trend")

m3 <- mp %>%
  select(Year, textile_firms, total_firms) %>%
  inner_join(gy_ps %>% select(Year, all_of(setdiff(corr_cols, c("textile_firms", "total_firms", "time")))), by = "Year") %>%
  mutate(time = Year - 2000) %>%
  select(all_of(corr_cols))

corr_mat <- cor(m3, use = "complete.obs")
corr_melt <- melt(corr_mat)
corr_melt <- corr_melt %>%
  mutate(Var1 = factor(Var1, levels = corr_cols, labels = corr_labels),
         Var2 = factor(Var2, levels = corr_cols, labels = corr_labels))

corr_melt_upper <- corr_melt %>%
  filter(as.numeric(Var1) < as.numeric(Var2))

p4 <- ggplot(corr_melt_upper, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", value)), size = 3.5) +
  scale_fill_gradient2(low = "#e74c3c", mid = "white", high = "#2c3e50",
                       midpoint = 0, limits = c(-1, 1), name = "Pearson r") +
  labs(title = "Correlation Matrix: Key Variables", x = "", y = "") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "bold"))

ggsave("output/figures_R/fig4_correlation_heatmap.png", p4, width = 9, height = 7, dpi = 300)

# ====== Fig5: Construct validity scatter ======
cat("Fig5: Construct validity...\n")
m5 <- ti %>%
  select(Year, index_policy_support) %>%
  inner_join(gy_ps %>% select(Year, policy_intensity_total), by = "Year")
corr_val <- cor(m5$index_policy_support, m5$policy_intensity_total)

p5 <- ggplot(m5, aes(x = index_policy_support, y = policy_intensity_total, label = Year)) +
  geom_point(aes(fill = Year), size = 3.5, shape = 21, stroke = 0.3) +
  geom_text(vjust = -1, hjust = 0.5, size = 3.5) +
  scale_fill_viridis_c() +
  annotate("text", x = min(m5$index_policy_support) + 0.2,
           y = max(m5$policy_intensity_total) * 0.9,
           label = sprintf("r = %.3f (n = %d)", corr_val, nrow(m5)),
           size = 5, hjust = 0) +
  labs(x = "Official Policy Support Index (Textile Association)",
       y = "LLM Policy Intensity Score",
       title = "Construct Validity: Official Index vs LLM Score (2020-2026)") +
  theme_paper

ggsave("output/figures_R/fig5_construct_validity.png", p5, width = 7.5, height = 5.5, dpi = 300)

# ====== Fig6: Policy heatmap ======
cat("Fig6: Policy heatmap...\n")
dims_6 <- c("equipment_index", "environment_index", "ecommerce_index",
            "brandquality_index", "cluster_index", "finance_index", "education_index")
labels_6 <- c("Equipment", "Environment", "Ecommerce", "Brand", "Cluster", "Finance", "Education")

heatmap_df <- gy_ps %>%
  select(Year, all_of(dims_6)) %>%
  filter(Year >= 2009) %>%
  pivot_longer(-Year, names_to = "Dimension", values_to = "Score") %>%
  mutate(Dimension = factor(Dimension, levels = rev(dims_6), labels = rev(labels_6)))

p6 <- ggplot(heatmap_df, aes(x = factor(Year), y = Dimension, fill = Score)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.0f", Score)), size = 3.5) +
  scale_fill_gradientn(colors = c("#fff5f0", "#fcbba1", "#fc9272", "#ef3b2d", "#99000d"),
                       name = "Score") +
  labs(x = "Year", y = "", title = "Policy Dimension Scores by Year (Gaoyang County)") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        panel.grid = element_blank())

ggsave("output/figures_R/fig6_policy_heatmap.png", p6, width = 11, height = 4.5, dpi = 300)

# ====== Fig7: Textile subindustry ======
cat("Fig7: Textile subindustry...\n")
pivot_sub <- ti_sub %>%
  filter(Year >= 2000) %>%
  pivot_wider(names_from = industry_level2, values_from = firm_count, values_fill = 0) %>%
  pivot_longer(-Year, names_to = "Industry", values_to = "Firms")

p7 <- ggplot(pivot_sub, aes(x = Year, y = Firms, fill = Industry)) +
  geom_bar(stat = "identity", alpha = 0.85, width = 0.8) +
  geom_vline(xintercept = 2017, linetype = "dashed", color = "gray50", linewidth = 0.8) +
  annotate("text", x = 2017.5, y = max(pivot_sub %>% group_by(Year) %>% summarise(s = sum(Firms)) %>% pull(s)) * 0.95,
           label = "2017", hjust = 0, size = 3.5, color = "gray40") +
  scale_fill_manual(values = c("#2c3e50", "#e74c3c")) +
  labs(x = "Year", y = "Firm Registrations",
       title = "Textile Industry Registrations by Sub-category (Gaoyang)",
       fill = "") +
  theme_paper

ggsave("output/figures_R/fig7_textile_subindustry.png", p7, width = 10, height = 5.5, dpi = 300)

# ====== Fig8: Pre vs Post 2017 boxplot ======
cat("Fig8: Pre vs Post boxplot...\n")
m8 <- mp %>%
  select(Year, textile_firms, total_firms) %>%
  inner_join(gy_ps %>% select(Year, policy_intensity_total), by = "Year") %>%
  mutate(Period = ifelse(Year < 2017, "Pre-2017\n(2000-2016)", "Post-2017\n(2017-2026)"))

plot_box <- function(data, var, ylab, title) {
  ggplot(data, aes(x = Period, y = .data[[var]], fill = Period)) +
    geom_boxplot(alpha = 0.7, width = 0.5, outlier.shape = NA) +
    geom_jitter(width = 0.08, alpha = 0.4, size = 1.8) +
    scale_fill_manual(values = c("Pre-2017\n(2000-2016)" = "#3498db", "Post-2017\n(2017-2026)" = "#e74c3c"), guide = "none") +
    labs(y = ylab, title = title) +
    theme_paper
}

p8a <- plot_box(m8, "textile_firms", "Textile Firms", "Textile Firms")
p8b <- plot_box(m8, "total_firms", "Total Firms", "Total Firms")
p8c <- plot_box(m8, "policy_intensity_total", "Policy Intensity (LLM Score)", "Policy Intensity")

p8 <- p8a + p8b + p8c + plot_annotation(title = "Pre vs Post 2017: Key Variables")
ggsave("output/figures_R/fig8_pre_post_boxplot.png", p8, width = 12, height = 4.5, dpi = 300)

cat("\nAll 8 R figures saved to output/figures_R/\n")
