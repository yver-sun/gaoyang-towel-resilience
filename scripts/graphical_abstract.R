# Graphical Abstract for Sustainability (MDPI) submission
# Single integrated PDF summarizing: Polycrisis → Policy Response → Cluster Resilience
.libPaths(c("C:/Users/Yver/R/library", .libPaths()))
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(patchwork)
library(showtext)

setwd("c:/Users/Yver/Desktop/史岩林/高阳毛巾")

# ── Load data ──
ps2 <- read_csv("output/policy_scores_panel_v2.csv", show_col_types = FALSE)
gy <- ps2 %>% filter(as.character(County_Code) == "130628", Year >= 2000, Year <= 2024)

policy_long <- gy %>%
  select(Year, environment_sum, ecommerce_sum, brandquality_sum) %>%
  pivot_longer(-Year, names_to = "Dimension", values_to = "Score") %>%
  mutate(
    Dimension = recode(Dimension,
      environment_sum = "Environment",
      ecommerce_sum = "E-commerce",
      brandquality_sum = "Brand"
    ),
    Phase = case_when(
      Year <= 2008 ~ "Phase 1:\nScale Expansion",
      Year <= 2017 ~ "Phase 2:\nEnvironmental Regulation",
      TRUE ~ "Phase 3:\nDigital & Brand"
    )
  )

# ── Font setup ──
tryCatch({
  font_add("noto", "C:/Windows/Fonts/simsun.ttc")
  showtext_auto()
}, error = function(e) NULL)

# ── Color palette ──
phase_colors <- c(
  "Phase 1:\nScale Expansion" = "#4472C4",
  "Phase 2:\nEnvironmental Regulation" = "#ED7D31",
  "Phase 3:\nDigital & Brand" = "#70AD47"
)
dim_colors <- c("Environment" = "#2E86AB", "E-commerce" = "#D64045", "Brand" = "#F2A900")

# ── Panel A: Policy attention evolution ──
p1 <- ggplot(policy_long, aes(x = Year, y = Score, fill = Dimension)) +
  geom_area(position = "stack", alpha = 0.85) +
  annotate("rect", xmin = 2000, xmax = 2008.5, ymin = -2, ymax = -0.5,
           fill = phase_colors[1], alpha = 0.9) +
  annotate("rect", xmin = 2008.5, xmax = 2017.5, ymin = -2, ymax = -0.5,
           fill = phase_colors[2], alpha = 0.9) +
  annotate("rect", xmin = 2017.5, xmax = 2024, ymin = -2, ymax = -0.5,
           fill = phase_colors[3], alpha = 0.9) +
  annotate("text", x = 2004, y = -1.25, label = "Scale\nExpansion", size = 2.8, color = "white", fontface = "bold") +
  annotate("text", x = 2013, y = -1.25, label = "Environmental\nRegulation", size = 2.8, color = "white", fontface = "bold") +
  annotate("text", x = 2021, y = -1.25, label = "Digital &\nBrand", size = 2.8, color = "white", fontface = "bold") +
  annotate("segment", x = 2008.5, xend = 2008.5, y = 62, yend = 68, linewidth = 0.4, linetype = "dashed", color = "grey40") +
  annotate("segment", x = 2017.5, xend = 2017.5, y = 62, yend = 68, linewidth = 0.4, linetype = "dashed", color = "grey40") +
  geom_vline(xintercept = 2017, linewidth = 0.8, linetype = "dotted", color = "red4") +
  annotate("text", x = 2017, y = 75, label = "2017 Shock", size = 3, color = "red4", fontface = "italic") +
  scale_fill_manual(values = dim_colors) +
  scale_x_continuous(breaks = seq(2000, 2024, 4)) +
  labs(title = "Policy Attention Evolution in Gaoyang Towel Cluster (2000–2024)",
       subtitle = "BERTopic-derived topic prevalence from 25 government work reports",
       x = NULL, y = "Policy Attention Score", fill = "Policy Dimension") +
  ylim(-3, 82) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

# ── Panel B: Mechanism schematic ──
mechanism_data <- data.frame(
  x = c(1, 2, 2, 3, 3, 4),
  y = c(2, 3, 1, 3, 1, 2),
  label = c("Polycrisis\nShocks", "Compliance\nCost Socialization",
            "Transaction\nCost Reduction", "Baseline\nResilience",
            "Value\nResilience", "Sustainable\nCluster"),
  type = c("shock", "mechanism", "mechanism", "outcome", "outcome", "goal")
)

arrows <- data.frame(
  x = c(1.3, 2, 2, 2.7, 2.7),
  xend = c(1.7, 2, 2.7, 2, 2.7),
  y = c(2, 3, 1, 3, 1),
  yend = c(2, 2.7, 2.7, 2.7, 1.3)
)

p2 <- ggplot() +
  geom_rect(aes(xmin = 0.5, xmax = 4.5, ymin = 0, ymax = 4), fill = "grey98") +
  # Main pathway
  annotate("segment", x = 1, xend = 2, y = 2, yend = 2, linewidth = 2, color = "grey30",
           arrow = arrow(length = unit(0.3, "cm"), type = "closed")) +
  annotate("segment", x = 2, xend = 3, y = 2.5, yend = 2.5, linewidth = 1.5, color = "#2E86AB",
           arrow = arrow(length = unit(0.25, "cm"), type = "closed")) +
  annotate("segment", x = 2, xend = 3, y = 1.5, yend = 1.5, linewidth = 1.5, color = "#D64045",
           arrow = arrow(length = unit(0.25, "cm"), type = "closed")) +
  annotate("segment", x = 3, xend = 4, y = 2.5, yend = 2.5, linewidth = 1.5, color = "#2E86AB",
           arrow = arrow(length = unit(0.25, "cm"), type = "closed")) +
  annotate("segment", x = 3, xend = 4, y = 1.5, yend = 1.5, linewidth = 1.5, color = "#D64045",
           arrow = arrow(length = unit(0.25, "cm"), type = "closed")) +
  # Nodes
  annotate("label", x = 1, y = 2, label = "Environmental\nRegulation Shock\n(2017)", size = 3.5, fill = "#F5F5F5", color = "red4", fontface = "bold") +
  annotate("label", x = 2, y = 3.2, label = "Centralized Wastewater\nEco-Industrial Parks\n≈F/N Cost Sharing", size = 3.2, fill = "#E8F4FD", color = "#2E86AB") +
  annotate("label", x = 2, y = 0.8, label = "E-commerce Platforms\nRegional Branding\nMarket Access ↑", size = 3.2, fill = "#FDE8E8", color = "#D64045") +
  annotate("label", x = 3, y = 3.2, label = "BASELINE\nRESILIENCE\nFirm Survival", size = 3.2, fill = "#D5E8D4", color = "#2E7D32", fontface = "bold") +
  annotate("label", x = 3, y = 0.8, label = "VALUE\nRESILIENCE\nUpgrading", size = 3.2, fill = "#D5E8D4", color = "#2E7D32", fontface = "bold") +
  annotate("label", x = 4, y = 2, label = "Sustainable\nIndustrial\nCluster", size = 3.5, fill = "#4472C4", color = "white", fontface = "bold") +
  # Sequencing arrow
  annotate("segment", x = 4.2, y = 3, xend = 4.2, yend = 1, linewidth = 1, color = "grey50",
           arrow = arrow(length = unit(0.2, "cm"), ends = "both", type = "closed")) +
  annotate("text", x = 4.7, y = 2, label = "Sequential", size = 2.8, color = "grey40", angle = 90) +
  xlim(0.3, 5.2) + ylim(-0.2, 4.2) +
  labs(title = "Transmission Mechanisms: From Shock to Sustainable Resilience",
       subtitle = "Microeconomic channels through which local government policy fosters cluster adaptability") +
  theme_void() +
  theme(
    plot.title = element_text(face = "bold", size = 11, hjust = 0.5),
    plot.subtitle = element_text(size = 9, color = "grey40", hjust = 0.5),
    plot.margin = margin(10, 5, 5, 5)
  )

# ── Assemble ──
ga <- p1 / p2 +
  plot_layout(heights = c(1, 0.65)) +
  plot_annotation(
    title = "Navigating the Polycrisis: Institutional Responses and Sustainable Cluster Resilience",
    subtitle = "Gaoyang Towel Cluster, China (2000–2024)  |  Mixed Methods: BERTopic + SCM + ITS",
    caption = "Yiwen Sun, Yanlin Shi, Jiarui Liang  |  Submitted to Sustainability (MDPI)",
    theme = theme(
      plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
      plot.subtitle = element_text(size = 10, color = "grey30", hjust = 0.5),
      plot.caption = element_text(size = 8, color = "grey50", hjust = 0.5)
    )
  )

# ── Save ──
ggsave("paper_latex_en/figures/Graphical_Abstract.pdf", ga,
       width = 10, height = 7.5, dpi = 300, device = "pdf")
ggsave("paper_latex_en/figures/Graphical_Abstract.png", ga,
       width = 10, height = 7.5, dpi = 300, device = "png")

cat("Graphical abstract saved to paper_latex_en/figures/Graphical_Abstract.pdf\n")
cat("Size: 10 x 7.5 inches at 300 dpi\n")
