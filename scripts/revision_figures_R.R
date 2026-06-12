# New figures for peer review revision
.libPaths(c("C:/Users/Yver/R/library", .libPaths()))
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(patchwork)

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

gy_ps <- ps %>% filter(as.character(County_Code) == "130628")
m <- mp %>%
  select(Year, textile_firms, total_firms) %>%
  inner_join(gy_ps %>% select(Year, policy_intensity_total,
    equipment_index, environment_index, ecommerce_index,
    brandquality_index, cluster_index, finance_index, education_index),
    by = "Year") %>%
  filter(Year <= 2024) %>%
  mutate(
    time = Year - 2000,
    post2017 = ifelse(Year >= 2017, "Post-2017", "Pre-2017"),
    period = ifelse(Year >= 2017, "Post-2017\n(2017-2024)", "Pre-2017\n(2000-2016)")
  )

# ====== Fig9: Kernel density — Policy intensity by period ======
cat("Fig9: Kernel density...\n")
p9 <- ggplot(m, aes(x = policy_intensity_total, fill = period, color = period)) +
  geom_density(alpha = 0.3, linewidth = 1) +
  scale_fill_manual(values = c("Pre-2017\n(2000-2016)" = "#3498db", "Post-2017\n(2017-2024)" = "#e74c3c")) +
  scale_color_manual(values = c("Pre-2017\n(2000-2016)" = "#3498db", "Post-2017\n(2017-2024)" = "#e74c3c")) +
  labs(x = "LLM Policy Intensity Score", y = "Density",
       title = "Distribution Shift: LLM Policy Intensity Pre vs Post 2017",
       fill = "", color = "") +
  theme_paper
ggsave("output/figures_R/fig9_kernel_density.png", p9, width = 8, height = 5, dpi = 300)

# ====== Fig10: Coefficient comparison — 4 specifications ======
cat("Fig10: 4-spec coefficient comparison...\n")

dims <- c("equipment_index", "environment_index", "ecommerce_index",
          "brandquality_index", "cluster_index", "finance_index", "education_index")
labels <- c("Equipment", "Environment", "Ecommerce",
            "Brand Quality", "Cluster", "Finance", "Education\n(placebo)")

get_coefs_4spec <- function(data, dims, labels_vec) {
  results <- data.frame()
  for (i in seq_along(dims)) {
    y <- data$textile_firms
    x <- data[[dims[i]]]

    # Spec 1: No time
    fit1 <- lm(y ~ x)
    s1 <- summary(fit1)$coefficients
    # Spec 2: With time
    fit2 <- lm(y ~ x + data$time)
    s2 <- summary(fit2)$coefficients
    # Spec 3: First-difference
    dy <- diff(y); dx <- diff(x)
    fit3 <- lm(dy ~ dx)
    s3 <- summary(fit3)$coefficients

    results <- rbind(results, data.frame(
      dim = dims[i], label = labels_vec[i],
      spec = "1.NoTime", coef = s1[2,1], se = s1[2,2],
      ci_low = s1[2,1] - 1.96*s1[2,2], ci_high = s1[2,1] + 1.96*s1[2,2]
    ))
    results <- rbind(results, data.frame(
      dim = dims[i], label = labels_vec[i],
      spec = "2.+Time", coef = s2[2,1], se = s2[2,2],
      ci_low = s2[2,1] - 1.96*s2[2,2], ci_high = s2[2,1] + 1.96*s2[2,2]
    ))
    results <- rbind(results, data.frame(
      dim = dims[i], label = labels_vec[i],
      spec = "3.1stDiff", coef = s3[2,1], se = s3[2,2],
      ci_low = s3[2,1] - 1.96*s3[2,2], ci_high = s3[2,1] + 1.96*s3[2,2]
    ))
  }
  results
}

coefs_4 <- get_coefs_4spec(m, dims, labels)
coefs_4$label <- factor(coefs_4$label, levels = rev(labels))
coefs_4$spec <- factor(coefs_4$spec, levels = c("1.NoTime", "2.+Time", "3.1stDiff"))
coefs_4$significant <- coefs_4$ci_low > 0

p10 <- ggplot(coefs_4, aes(x = coef, y = label, fill = significant)) +
  geom_bar(stat = "identity", alpha = 0.85, width = 0.6) +
  geom_errorbar(aes(xmin = ci_low, xmax = ci_high), width = 0.2) +
  geom_vline(xintercept = 0, linewidth = 0.4) +
  scale_fill_manual(values = c("TRUE" = "#2c3e50", "FALSE" = "#bdc3c7"), guide = "none") +
  facet_wrap(~ spec, ncol = 3,
             labeller = labeller(spec = c(
               "1.NoTime" = "Spec 1: No Time Control\n(Spurious)",
               "2.+Time" = "Spec 2: + Time Trend\n(Correct)",
               "3.1stDiff" = "Spec 3: First Difference\n(Robustness)"
             ))) +
  labs(x = "Coefficient (firms per unit)", y = "",
       title = "Policy Dimension Effects Under Three Specifications") +
  theme_paper
ggsave("output/figures_R/fig10_4spec_coefficients.png", p10, width = 14, height = 5.5, dpi = 300)

# ====== Fig11: SCM gap with confidence band ======
cat("Fig11: SCM gap with inference...\n")
p11 <- ggplot(scm, aes(x = Year, y = Gap)) +
  geom_bar(aes(fill = ifelse(Gap >= 0, "Positive", "Negative")),
           stat = "identity", alpha = 0.85, width = 0.7) +
  geom_hline(yintercept = 0, linewidth = 0.5) +
  geom_hline(yintercept = mean(scm$Gap[scm$Year >= 2017]), linetype = "dotted",
             color = "#e74c3c", linewidth = 0.8) +
  annotate("text", x = 2020, y = mean(scm$Gap[scm$Year >= 2017]) + 20,
           label = sprintf("ATT = %.1f", mean(scm$Gap[scm$Year >= 2017])),
           color = "#e74c3c", size = 4) +
  geom_vline(xintercept = 2017, linetype = "dashed", color = "gray50", linewidth = 1) +
  annotate("text", x = 2017.5, y = max(scm$Gap) * 0.9,
           label = "2017", hjust = 0, size = 3.5, color = "gray40") +
  scale_fill_manual(values = c("Positive" = "#3498db", "Negative" = "#e74c3c"), guide = "none") +
  labs(x = "Year", y = "Gap (Actual - Synthetic)",
       title = "SCM Treatment Effect: Gap by Year") +
  theme_paper
ggsave("output/figures_R/fig11_scm_gap_inference.png", p11, width = 10, height = 5, dpi = 300)

# ====== Fig12: Correlation scatter — policy docs vs LLM score ======
cat("Fig12: Policy docs vs LLM score...\n")
pc <- read_csv("output/policy_document_counts.csv", show_col_types = FALSE)
m12 <- gy_ps %>% select(Year, policy_intensity_total) %>%
  left_join(pc %>% select(Year, policy_doc_count), by = "Year")
corr_doc_llm <- cor(m12$policy_intensity_total, m12$policy_doc_count, use = "complete.obs")

p12 <- ggplot(m12, aes(x = policy_doc_count, y = policy_intensity_total, label = Year)) +
  geom_point(aes(fill = Year), size = 4, shape = 21, stroke = 0.3) +
  geom_smooth(method = "lm", se = TRUE, color = "#e74c3c", alpha = 0.15) +
  geom_text(vjust = -1, hjust = 0.5, size = 3) +
  scale_fill_viridis_c() +
  annotate("text", x = max(m12$policy_doc_count, na.rm = TRUE) * 0.7,
           y = max(m12$policy_intensity_total, na.rm = TRUE) * 0.9,
           label = sprintf("r = %.3f\n1st-stage F = 1.36", corr_doc_llm),
           size = 4.5, hjust = 0) +
  labs(x = "Annual Policy Document Count", y = "LLM Policy Intensity Score",
       title = "Weak Instrument: Policy Documents vs LLM Scores") +
  theme_paper
ggsave("output/figures_R/fig12_weak_iv.png", p12, width = 7.5, height = 5.5, dpi = 300)

cat("\nAll 4 new R figures saved to output/figures_R/\n")
