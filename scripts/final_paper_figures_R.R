# Final paper figures — publication-quality ggplot2 plots
# Data source: rigorous_analysis.py outputs + CSV files
.libPaths(c("C:/Users/Yver/R/library", .libPaths()))
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(patchwork)
library(ggrepel)

setwd("c:/Users/Yver/Desktop/史岩林/高阳毛巾")
dir.create("output/figures_R", showWarnings = FALSE, recursive = TRUE)

theme_paper <- theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(size = 10, color = "grey40"),
    axis.title = element_text(size = 10),
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = 0.3, color = "grey90"),
    plot.margin = margin(12, 15, 10, 10)
  )

# ═══════════════════════════════════════════════════════
# Load and prepare data
# ═══════════════════════════════════════════════════════
ps2 <- read_csv("output/policy_scores_panel_v2.csv", show_col_types = FALSE)
mp <- read_csv("output/master_panel_data_v2.csv", show_col_types = FALSE)
ti <- read_csv("output/textile_indices_annual.csv", show_col_types = FALSE)
media_pol <- read_csv("data/policies/高阳县毛巾产业政策文件_全量_20260517_195152.csv", show_col_types = FALSE)

# Gaoyang subset (2015-2024 estimation window)
gy <- ps2 %>% filter(as.character(County_Code) == "130628", Year >= 2015, Year <= 2024)
m <- mp %>%
  filter(as.character(County_Code) == "130628", Year >= 2015, Year <= 2024) %>%
  select(Year, textile_firms, total_firms, textile_ratio, textile_firms_ma3) %>%
  inner_join(gy %>% select(Year, policy_intensity_total, policy_intensity_concentration,
    equipment_sum, environment_sum, ecommerce_sum, brandquality_sum,
    cluster_sum, finance_sum, education_sum, total_chunks),
    by = "Year")

# Media counts
library(stringr)
media_pol <- media_pol %>%
  mutate(yr = as.integer(str_extract(发文时间, "\\d{4}"))) %>%
  filter(yr >= 2015, yr <= 2024)
media_counts <- media_pol %>% count(yr, name = "media_count")
m <- m %>% left_join(media_counts %>% rename(Year = yr), by = "Year")
m$media_count[is.na(m$media_count)] <- 0

# Residualization
fit_resid <- lm(policy_intensity_total ~ total_chunks, data = m)
m <- m %>% mutate(
  policy_predicted = predict(fit_resid),
  policy_residual = policy_intensity_total - policy_predicted
)

# Residualize each dimension
dim_names <- c("equipment_sum", "environment_sum", "ecommerce_sum",
               "brandquality_sum", "cluster_sum", "finance_sum", "education_sum")
dim_labels <- c("Equipment", "Environment", "Ecommerce",
                "Brand&Quality", "Cluster", "Finance", "Education")
for (i in seq_along(dim_names)) {
  col <- dim_names[i]
  vals <- m[[col]]
  vals[is.na(vals)] <- 0
  fit_d <- lm(vals ~ m$total_chunks)
  m[[paste0(col, "_resid")]] <- vals - predict(fit_d)
}

# PCA on residualized dimensions
dim_mat <- as.matrix(m[, paste0(dim_names, "_resid")])
pca_fit <- prcomp(dim_mat, center = TRUE, scale. = TRUE)
m$PC1 <- pca_fit$x[, 1]
m$PC2 <- pca_fit$x[, 2]
loadings_pc1 <- pca_fit$rotation[, 1]
loadings_pc2 <- pca_fit$rotation[, 2]

# Derived variables
m <- m %>% mutate(
  ln_textile = log(textile_firms),
  time = Year - 2014,
  period = ifelse(Year >= 2017, "Post-2017", "Pre-2017")
)

# Annualized quarterly indices for construct validity
ti_annual <- ti %>%
  select(Year, index_policy_support, index_competitiveness, index_industrial_development) %>%
  inner_join(m %>% select(Year, policy_intensity_total, policy_residual, total_chunks), by = "Year")

cat("Data loaded:", nrow(m), "years,", ncol(m), "columns\n")

# ═══════════════════════════════════════════════════════
# FIGURE 1: Dual-axis time series (Textile Firms + Policy Residual)
# ═══════════════════════════════════════════════════════
cat("Figure 1: Time series with residualized policy...\n")
m_long <- m %>%
  select(Year, textile_firms, policy_residual) %>%
  mutate(policy_residual_scaled = policy_residual * (max(textile_firms) / max(abs(policy_residual) + 0.1)))

fig1 <- ggplot(m, aes(x = Year)) +
  geom_rect(aes(xmin = 2019.8, xmax = 2022.2, ymin = -Inf, ymax = Inf),
            fill = "grey90", alpha = 0.5) +
  annotate("text", x = 2021, y = max(m$textile_firms) * 0.95,
           label = "COVID-19", size = 3.5, color = "grey50", fontface = "italic") +
  geom_bar(aes(y = textile_firms), stat = "identity", fill = "#2c3e50", alpha = 0.85, width = 0.7) +
  geom_line(aes(y = policy_residual * (max(textile_firms) / max(abs(policy_residual) + 0.1))),
            color = "#e74c3c", linewidth = 1.2) +
  geom_point(aes(y = policy_residual * (max(textile_firms) / max(abs(policy_residual) + 0.1))),
             color = "#e74c3c", size = 3) +
  geom_hline(yintercept = 0 * (max(textile_firms) / max(abs(m$policy_residual) + 0.1)),
             linetype = "dotted", color = "#e74c3c", linewidth = 0.5) +
  geom_vline(xintercept = 2017, linetype = "dashed", color = "grey50", linewidth = 0.8) +
  annotate("text", x = 2017.3, y = max(m$textile_firms) * 0.9,
           label = "2017\nEnvironmental\nInspection", hjust = 0, size = 3.2, color = "grey40") +
  scale_x_continuous(breaks = 2015:2024) +
  scale_y_continuous(
    name = "New Textile Firm Registrations",
    sec.axis = sec_axis(~ . / (max(m$textile_firms) / max(abs(m$policy_residual) + 0.1)),
                        name = "Residualized Policy Intensity\n(Excess Policy Effort)")
  ) +
  labs(title = "Figure 1: Textile Firm Entry and Abnormal Policy Effort (2015–2024)",
       subtitle = "Residualized policy = policy intensity orthogonal to report length (r = 0.000 with paragraph count)") +
  theme_paper +
  theme(axis.title.y.right = element_text(color = "#e74c3c"),
        axis.text.y.right = element_text(color = "#e74c3c"),
        axis.title.y.left = element_text(color = "#2c3e50"))

ggsave("output/figures_R/fig1_timeseries_resid.png", fig1, width = 10, height = 5.5, dpi = 300)

# ═══════════════════════════════════════════════════════
# FIGURE 2: Residualization scatter (Policy vs Chunks)
# ═══════════════════════════════════════════════════════
cat("Figure 2: Residualization...\n")
cor_raw <- round(cor(m$policy_intensity_total, m$total_chunks), 3)
cor_resid <- round(cor(m$policy_residual, m$total_chunks), 3)

fig2 <- ggplot(m, aes(x = total_chunks, y = policy_intensity_total)) +
  geom_smooth(method = "lm", se = TRUE, color = "#e74c3c", fill = "#e74c3c", alpha = 0.12, linewidth = 0.8) +
  geom_segment(aes(xend = total_chunks, yend = policy_predicted),
               color = "grey70", linewidth = 0.5, linetype = "dotted") +
  geom_point(aes(fill = policy_residual), size = 4, shape = 21, stroke = 0.3, color = "grey30") +
  geom_text_repel(aes(label = Year), size = 3.2, box.padding = 0.5, max.overlaps = 10) +
  scale_fill_gradient2(low = "#3498db", mid = "white", high = "#e74c3c",
                        midpoint = 0, name = "Residual\n(Excess Effort)") +
  annotate("text", x = max(m$total_chunks) * 0.7, y = max(m$policy_intensity_total) * 0.95,
           label = paste0("r = ", cor_raw, "\nPolicy = ", round(coef(fit_resid)[1], 1),
                          " + ", round(coef(fit_resid)[2], 2), " * Chunks\nResidualized r = ", cor_resid),
           size = 3.5, hjust = 0, color = "grey30") +
  labs(x = "Report Paragraph Count", y = "Raw Policy Intensity Score",
       title = "Figure 2: The Text-Length Confound — Raw Policy vs Paragraph Count",
       subtitle = "Dotted lines show residuals (vertical distance from regression line). Colored by residual sign.") +
  theme_paper

ggsave("output/figures_R/fig2_residualization.png", fig2, width = 8.5, height = 6, dpi = 300)

# ═══════════════════════════════════════════════════════
# FIGURE 3: PCA Loadings Biplot
# ═══════════════════════════════════════════════════════
cat("Figure 3: PCA loadings...\n")
pca_loadings <- data.frame(
  Dimension = dim_labels,
  PC1 = loadings_pc1,
  PC2 = loadings_pc2
)

fig3 <- ggplot(pca_loadings, aes(x = PC1, y = PC2)) +
  geom_hline(yintercept = 0, linewidth = 0.3, color = "grey80") +
  geom_vline(xintercept = 0, linewidth = 0.3, color = "grey80") +
  geom_segment(aes(x = 0, y = 0, xend = PC1, yend = PC2),
               arrow = arrow(length = unit(0.25, "cm")),
               color = "#2c3e50", linewidth = 1) +
  geom_text_repel(aes(label = Dimension), size = 3.8, box.padding = 0.5,
                   force = 2, fontface = ifelse(abs(pca_loadings$PC1) > 0.35 | abs(pca_loadings$PC2) > 0.35, "bold", "plain")) +
  annotate("text", x = 0.3, y = 0.95, label = "Environmental\nGovernance", size = 3, color = "#e74c3c", hjust = 0) +
  annotate("text", x = -0.95, y = 0.15, label = "Cluster\nCoordination", size = 3, color = "#3498db", hjust = 0) +
  annotate("text", x = 0.15, y = -0.9, label = "Education &\nTraining", size = 3, color = "#3498db", hjust = 0) +
  labs(x = paste0("PC1 (", round(summary(pca_fit)$importance[2,1] * 100, 1),
                  "%): Environment—Cluster Axis"),
       y = paste0("PC2 (", round(summary(pca_fit)$importance[2,2] * 100, 1),
                  "%): Cluster—Education Axis"),
       title = "Figure 3: PCA of Residualized Policy Dimensions",
       subtitle = paste0("Cumulative variance explained: ",
                         round(sum(summary(pca_fit)$importance[2, 1:2]) * 100, 1),
                         "%  |  Bold = |loading| > 0.35")) +
  coord_fixed(xlim = c(-1.05, 1.05), ylim = c(-1.05, 1.05)) +
  theme_paper

ggsave("output/figures_R/fig3_pca_loadings.png", fig3, width = 8.5, height = 6.5, dpi = 300)

# ═══════════════════════════════════════════════════════
# FIGURE 4: Permutation Test Results — Coefficient Comparison
# ═══════════════════════════════════════════════════════
cat("Figure 4: Permutation test results...\n")
# Manual permutation (replicating Python logic for R figure)
set.seed(42)
n <- nrow(m)
y <- m$ln_textile
time <- m$time
post <- m$post

N_PERM <- 5000

perm_results <- data.frame()
test_vars <- list(
  "Raw Policy" = m$policy_intensity_total,
  "Resid. Policy" = m$policy_residual,
  "PC1 (Resid. Dims)" = m$PC1,
  "Environment (Resid.)" = m$environment_sum_resid,
  "Media Events" = m$media_count
)

for (vname in names(test_vars)) {
  X <- cbind(1, test_vars[[vname]], time, post)
  k <- ncol(X)
  b_obs <- solve(t(X) %*% X) %*% t(X) %*% y
  beta_obs <- b_obs[2]
  r <- y - X %*% b_obs
  s2 <- sum(r^2) / (n - k)
  p_ols <- 2 * (1 - pt(abs(beta_obs / sqrt(s2 * solve(t(X) %*% X)[2, 2])), n - k))

  # Block-respecting permutation
  pre_idx <- which(post == 0)
  post_idx <- which(post == 1)
  null_dist <- numeric(N_PERM)
  for (i in 1:N_PERM) {
    y_perm <- y
    y_perm[pre_idx] <- sample(y[pre_idx])
    y_perm[post_idx] <- sample(y[post_idx])
    b_perm <- solve(t(X) %*% X) %*% t(X) %*% y_perm
    null_dist[i] <- b_perm[2]
  }
  p_perm <- mean(abs(null_dist) >= abs(beta_obs))
  perm_results <- rbind(perm_results, data.frame(
    Variable = vname, Beta = beta_obs, Perm_p = p_perm, OLS_p = p_ols
  ))
}

perm_results$Variable <- factor(perm_results$Variable, levels = rev(perm_results$Variable))
perm_results$Sig <- ifelse(perm_results$Perm_p < 0.10, "p < 0.10", "n.s.")
perm_results$Sig_label <- sprintf("p = %.3f", perm_results$Perm_p)

fig4 <- ggplot(perm_results, aes(x = Beta, y = Variable)) +
  geom_vline(xintercept = 0, linewidth = 0.5, color = "grey70", linetype = "dashed") +
  geom_bar(aes(fill = Perm_p < 0.10), stat = "identity", alpha = 0.85, width = 0.6) +
  geom_text(aes(label = Sig_label, x = ifelse(Beta >= 0, Beta + 0.008, Beta - 0.008)),
            size = 3.5, hjust = ifelse(perm_results$Beta >= 0, 0, 1)) +
  scale_fill_manual(values = c("TRUE" = "#e74c3c", "FALSE" = "#bdc3c7"), guide = "none") +
  labs(x = expression(beta ~ "Coefficient (DV: ln Textile Firms)"), y = "",
       title = "Figure 4: Permutation-Based Inference — Policy Effects on Firm Entry",
       subtitle = paste0("N = 10 years (2015–2024), ", N_PERM,
                         " block-respecting permutations. Red = nominal p < 0.10")) +
  theme_paper

ggsave("output/figures_R/fig4_permutation_results.png", fig4, width = 9, height = 4.5, dpi = 300)

# ═══════════════════════════════════════════════════════
# FIGURE 5: Construct Validity — LLM vs Official Index
# ═══════════════════════════════════════════════════════
cat("Figure 5: Construct validity...\n")
if (nrow(ti_annual) >= 3) {
  n_val <- nrow(ti_annual %>% filter(!is.na(policy_residual), !is.na(index_policy_support)))
  cor_val <- round(cor(ti_annual$policy_residual, ti_annual$index_policy_support, use = "complete.obs"), 3)
  cor_val_raw <- round(cor(ti_annual$policy_intensity_total, ti_annual$index_policy_support, use = "complete.obs"), 3)

  fig5 <- ggplot(ti_annual, aes(x = index_policy_support, y = policy_residual)) +
    geom_smooth(method = "lm", se = TRUE, color = "#e74c3c", fill = "#e74c3c", alpha = 0.12, linewidth = 0.8) +
    geom_point(aes(fill = Year), size = 4.5, shape = 21, stroke = 0.3, color = "grey40") +
    geom_text_repel(aes(label = Year), size = 3.5, box.padding = 0.5) +
    scale_fill_viridis_c(option = "C") +
    geom_hline(yintercept = 0, linewidth = 0.3, color = "grey80", linetype = "dotted") +
    annotate("text", x = min(ti_annual$index_policy_support, na.rm = TRUE) +
               0.05 * diff(range(ti_annual$index_policy_support, na.rm = TRUE)),
             y = max(ti_annual$policy_residual, na.rm = TRUE) * 0.85,
             label = paste0("r_resid = ", cor_val, "  (n = ", n_val, ")\n",
                            "r_raw = ", cor_val_raw),
             size = 4, hjust = 0, color = "grey30") +
    labs(x = "Official Policy Support Index (Textile Industry Association)",
         y = "Residualized LLM Policy Score",
         title = "Figure 5: Construct Validity — Residualized LLM Score vs Official Index",
         subtitle = "2020–2024 annualized. Residualization improved r from near-zero to moderate positive.") +
    theme_paper

  ggsave("output/figures_R/fig5_construct_validity.png", fig5, width = 8, height = 5.5, dpi = 300)
}

# ═══════════════════════════════════════════════════════
# FIGURE 6: Quarterly Index Trends (supplementary)
# ═══════════════════════════════════════════════════════
cat("Figure 6: Quarterly index trends...\n")
ti_quarterly <- ti %>% filter(!is.na(`产业景气指数`)) %>%
  mutate(period = paste0(Year, "-Q", Quarter),
         time_idx = Year + (Quarter - 1) / 4)

if (nrow(ti_quarterly) > 0) {
  ti_long <- ti_quarterly %>%
    select(time_idx, `产业景气指数`, `产业竞争力指数`, `政策支持指数`) %>%
    pivot_longer(-time_idx, names_to = "Index", values_to = "Value")

  fig6 <- ggplot(ti_long, aes(x = time_idx, y = Value, color = Index)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 1.5) +
    scale_color_manual(values = c("产业景气指数" = "#2c3e50",
                                   "产业竞争力指数" = "#e74c3c",
                                   "政策支持指数" = "#3498db")) +
    geom_hline(yintercept = 50, linewidth = 0.3, color = "grey70", linetype = "dotted") +
    labs(x = "Year", y = "Index Value", color = "",
         title = "Figure 6: Quarterly Textile Monitoring Indices (2020–2025)",
         subtitle = "Source: China National Textile and Apparel Council. 50 = neutral threshold.") +
    theme_paper

  ggsave("output/figures_R/fig6_quarterly_indices.png", fig6, width = 10, height = 5, dpi = 300)
}

# ═══════════════════════════════════════════════════════
# FIGURE 7: Descriptive — Pre vs Post 2017 boxplots
# ═══════════════════════════════════════════════════════
cat("Figure 7: Pre vs Post 2017 comparison...\n")
m_period <- m %>%
  select(Year, textile_firms, total_firms, policy_residual, period) %>%
  pivot_longer(c(textile_firms, total_firms, policy_residual),
               names_to = "Variable", values_to = "Value") %>%
  mutate(Variable = factor(Variable,
    levels = c("textile_firms", "total_firms", "policy_residual"),
    labels = c("Textile Firms\n(New Reg.)", "Total Firms\n(New Reg.)", "Residualized\nPolicy Intensity")))

fig7 <- ggplot(m_period, aes(x = period, y = Value, fill = period)) +
  geom_boxplot(alpha = 0.6, width = 0.5, outlier.shape = NA) +
  geom_jitter(width = 0.08, alpha = 0.5, size = 2) +
  facet_wrap(~ Variable, scales = "free_y") +
  scale_fill_manual(values = c("Pre-2017" = "#3498db", "Post-2017" = "#e74c3c"), guide = "none") +
  labs(x = "", y = "", title = "Figure 7: Key Variables Before and After the 2017 Environmental Shock") +
  theme_paper

ggsave("output/figures_R/fig7_pre_post_boxplot.png", fig7, width = 10, height = 4, dpi = 300)

# ═══════════════════════════════════════════════════════
# FIGURE 8: Dimension-level effects (7 dimensions comparison)
# ═══════════════════════════════════════════════════════
cat("Figure 8: Dimension-level effects...\n")
dim_results <- data.frame()
for (i in seq_along(dim_names)) {
  col_resid <- paste0(dim_names[i], "_resid")
  X <- cbind(1, m[[col_resid]], m$time, m$post)
  k <- 4
  b <- solve(t(X) %*% X) %*% t(X) %*% m$ln_textile
  r <- m$ln_textile - X %*% b
  p_ols <- 2 * (1 - pt(abs(b[2] / sqrt(sum(r^2) / (n - k) * solve(t(X) %*% X)[2, 2])), n - k))
  dim_results <- rbind(dim_results, data.frame(
    Dimension = dim_labels[i], Beta = b[2], P_value = p_ols
  ))
}
dim_results$Dimension <- factor(dim_results$Dimension, levels = rev(dim_labels))
dim_results$Sig <- ifelse(dim_results$P_value < 0.05, "p < 0.05",
                          ifelse(dim_results$P_value < 0.10, "p < 0.10", "n.s."))

fig8 <- ggplot(dim_results, aes(x = Beta, y = Dimension)) +
  geom_vline(xintercept = 0, linewidth = 0.5, color = "grey70", linetype = "dashed") +
  geom_bar(aes(fill = P_value < 0.10), stat = "identity", alpha = 0.85, width = 0.6) +
  geom_text(aes(label = sprintf("p = %.3f", P_value),
                x = ifelse(Beta >= 0, Beta + 0.01, Beta - 0.01)),
            size = 3.2, hjust = ifelse(dim_results$Beta >= 0, 0, 1)) +
  scale_fill_manual(values = c("TRUE" = "#e74c3c", "FALSE" = "#bdc3c7"), guide = "none") +
  labs(x = expression(beta ~ "Coefficient (DV: ln Textile Firms)"), y = "",
       title = "Figure 8: Dimension-Level Effects (Residualized, with Time + Post2017 Controls)",
       subtitle = "Each dimension entered separately. n = 10. Red = p < 0.10 (OLS, likely optimistic).") +
  theme_paper

ggsave("output/figures_R/fig8_dimension_effects.png", fig8, width = 9, height = 4.5, dpi = 300)

cat("\nAll 8 figures saved to output/figures_R/\n")
cat("Files:\n")
for (f in dir("output/figures_R", pattern = "*.png")) {
  cat("  ", f, "\n")
}
