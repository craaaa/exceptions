### Analysis script for Experiment 1A of Leong and Linzen 2024
### December 2024
### CS Leong & T Linzen

here::i_am("analysis/exp1_analysis.R")
library(here)
library(lme4)
source(here("analysis", "utils.R"))

# PARAMETERS
MAX_UNEXPECTED_FILLERS <- 15
SAVE_GRAPHS <- TRUE
set.seed(123)

### Read in human data
raw_data_human <- read_csv(here("data", "human", "results.csv"),
  comment = "#",
  col_names = c(
    "", "", "", "order_no", "", "trial_type", "", "element_type", "element_name",
    "", "score", "", "", "group", "list", "frame", "verb", "sentence_type",
    "sentence", "presentation_order", "is_passivizable", "id", "", ""
  )
) %>%
  select(id, verb, score, list, frame, order_no, sentence, sentence_type, trial_type, element_type) %>%
  filter(element_type == "Scale") %>%
  mutate(
    score = as.numeric(score),
    frame = as.numeric(frame),
    verb = factor(verb),
    verb_class = get_verb_class(verb)
  )

participants_to_remove <- raw_data_human %>%
  filter(element_type == "Scale") %>%
  filter((sentence_type == "check_accept" & score < 50) |
    (sentence_type == "check_unaccept" & score > 50)) %>%
  count(id) %>%
  filter(n >= MAX_UNEXPECTED_FILLERS) %>%
  select(id)

data_human <- raw_data_human %>% filter(!id %in% participants_to_remove$id)
num_participants_human <- length(unique(data_human$id))

### Obtain test and filler dfs
test_human <- data_human %>%
  filter(trial_type == "experimental-trial")

fillers_human <- data_human %>%
  filter(trial_type == "attention-check") %>%
  filter(!str_detect(sentence_type, "Attention")) %>%
  mutate(frame = order_no)

### Compute Cousineau-corrected SEs for purposes of plotting
grand_means_human <- test_human %>%
  group_by(verb, sentence_type) %>%
  summarise(global_mean = mean(score, na.rm = TRUE))

participant_means_human <- test_human %>%
  group_by(id, verb, sentence_type) %>%
  summarise(participant_mean = mean(score, na.rm = TRUE))

test_human <- test_human %>%
  left_join(grand_means_human) %>%
  left_join(participant_means_human) %>%
  mutate(cousineau_score = score - participant_mean + global_mean)

mean_test_human <- test_human %>%
  group_by(verb, verb_class, sentence_type) %>%
  summarise(
    result = Hmisc::smean.cl.boot(cousineau_score) %>% t() %>% as.data.frame()
  ) %>%
  unnest(result)

mean_test_human %>%
  ggplot(aes(x = sentence_type, y = Mean, colour = verb_class)) +
  facet_grid(cols = vars(verb_class)) +
  geom_point(aes(group = verb)) +
  geom_line(aes(group = interaction(verb))) +
  geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.1) +
  geom_text_repel(
    data = subset(mean_test_human, sentence_type == "passive"),
    aes(label = verb, segment.linetype = 2),
    size = 3.5, family = "fira", hjust = "right", vjust = 0,
    nudge_x = 1.5, color = "black"
  ) +
  scale_colour_manual(values = verb_class_colours) +
  labs(
    # title="Human ratings of active and passive sentences by verb",
    caption = paste("Human n=", toString(num_participants_human))
  ) +
  xlab(element_blank()) +
  ylab("Mean sentence score") +
  theme(
    legend.position = "none",
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
  ) +
  scale_y_continuous(limits = c(0, 100)) +
  scale_x_discrete(labels = c("Active", "Passive"), expand = expansion(mult = 0.5)) # +
# theme(axis.text.x = element_text(angle = 45, hjust = 1))

if (SAVE_GRAPHS) {
  ggsave(filename = here("analysis", "figures", "exp1a_human-duckbill.pdf"), device = "pdf", width = 12, height = 4, units = "in")
}

fillers_human %>%
  ggplot(aes(x = score, fill = sentence_type)) +
  geom_histogram() +
  facet_wrap(vars(sentence))

filler_variance <- fillers_human %>%
  group_by(sentence) %>%
  summarise(variance = sd(score)) %>%
  mutate(sentence_type = "filler")

variance_histogram <- test_human %>%
  group_by(verb, frame, sentence_type) %>%
  summarise(variance = sd(score)) %>%
  bind_rows(filler_variance) %>%
  mutate(sentence_type = factor(sentence_type, levels = c("active", "passive", "filler"))) %>%
  ggplot(aes(x = variance, fill = sentence_type)) +
  geom_histogram() +
  facet_wrap(vars(sentence_type)) +
  xlab("Inter-rater standard deviation") +
  ylab("Frequency") +
  scale_fill_manual(values = sent_type_colours) +
  theme(legend.position = "none")

variance_histogram

test_human %>%
  filter(sentence_type == "passive") %>%
  ggplot(aes(x = score, fill = frame)) +
  geom_histogram() +
  facet_wrap(vars(sentence))

score_histogram <- test_human %>%
  filter(frame == 24) %>%
  ggplot(aes(x = score, fill = sentence_type)) +
  geom_histogram() +
  facet_grid(rows = vars(sentence_type), cols = vars(verb)) +
  xlab("Score") +
  ylab("Frequency") +
  scale_fill_manual(values = sent_type_colours, name = element_blank()) +
  theme(legend.position = "none")

variance_histogram + score_histogram

if (SAVE_GRAPHS) {
  ggsave(filename = here("analysis", "figures", "exp1a_human-score-histogram.pdf"), device = "pdf", width = 12, height = 4, units = "in")
}

## Calculate passive drop for human data
mean_pass_drop_human <- mean_test_human %>%
  pivot_wider(names_from = sentence_type, values_from = c(Mean, Lower, Upper)) %>%
  mutate(
    Mean = Mean_active - Mean_passive,
    Lower = Lower_active - Upper_passive,
    Upper = Upper_active - Lower_passive
  ) %>%
  select(verb, verb_class, Mean, Lower, Upper)

### Model results
data_models <- list.files(path = here("scores", "exp1b"), pattern = "*.csv", full.names = TRUE) %>%
  read_csv(id = "model_id") %>%
  mutate(verb_class = get_verb_class(verb))

data_models %>%
  select(model_id) %>%
  distinct()

## Calculate Cousineau-corrected SEs for models
grand_means_models <- data_models %>%
  group_by(verb) %>%
  summarise(global_mean = mean(pass_drop, na.rm = TRUE))

participant_means_models <- data_models %>%
  group_by(model_id, verb) %>%
  summarise(participant_mean = mean(pass_drop, na.rm = TRUE))

data_models <- data_models %>%
  left_join(grand_means_models) %>%
  left_join(participant_means_models) %>%
  mutate(cousineau_pass_drop = pass_drop - participant_mean + global_mean)

mean_test_models <- data_models %>%
  group_by(verb, verb_class) %>%
  summarise(result = Hmisc::smean.cl.boot(cousineau_pass_drop) %>% t() %>% as.data.frame()) %>%
  unnest(result)

mean_test_combined <- mean_test_models %>%
  left_join(mean_pass_drop_human, by = c("verb", "verb_class"), suffix = c("_model", "_human"))

mean_test_combined %>%
  ggplot(aes(x = Mean_human, y = Mean_model)) +
  geom_smooth(aes(alpha = 0.8), method = "lm") +
  geom_point(aes(color = verb_class)) +
  geom_errorbar(aes(ymin = Lower_model, ymax = Upper_model, color = verb_class, alpha = 0.8), width = 0) +
  geom_errorbarh(aes(xmin = Lower_human, xmax = Upper_human, color = verb_class, alpha = 0.8), width = 0) +
  xlab("Mean human passive drop") +
  ylab("Mean model passive drop") +
  geom_text_repel(aes(label = verb), min.segment.length = 0, size = 3.5, max.overlaps = Inf, segment.linetype = 2, family = "firal", nudge_x = 1, nudge_y = 0.4) +
  scale_x_continuous(limits = c(-12, 89), breaks = seq(0, 80, by = 20)) +
  scale_y_continuous(limits = c(-10, 40), breaks = seq(0, 30, by = 10)) +
  theme(legend.position = "bottom", legend.title = element_blank()) +
  scale_color_manual(values = verb_class_colours) +
  ylim(-10, 30) +
  guides(alpha = "none", colour = guide_legend(nrow = 1))

## Correlation if we remove outlying verbs

mean_test_combined %>%
  ungroup() %>%
  filter(verb != "lasted", verb != "cost", verb != "took") %>%
  select(Mean_human, Mean_model) %>%
  cor(method = "pearson")


if (SAVE_GRAPHS) {
  ggsave(filename = here("analysis", "figures", "exp1b_human-model-correlation-scatter.pdf"), device = "pdf", width = 10, height = 6, units = "in")
}

### Linear models

test_human <- test_human %>%
  mutate(verb_class = relevel(verb_class, ref = "Agent-Patient"))
verb_class_lmm <- lmer(
  score ~ sentence_type + verb_class + sentence_type:verb_class + (1 | frame) + (1 + sentence_type | id) + (1 | verb),
  data = test_human,
  REML = FALSE
)
summary(verb_class_lmm)

scores_human_duration <- test_human %>%
  filter(verb_class == "Duration") %>%
  mutate(verb = relevel(verb, ref = "required"))
lmm_duration <- lmer(
  score ~ sentence_type + verb + sentence_type:verb + (1 | frame) + (1 | id),
  data = scores_human_duration,
  REML = FALSE
)
summary(lmm_duration)

scores_human_price <- test_human %>%
  filter(verb_class == "Price") %>%
  mutate(verb = relevel(verb, ref = "earned"))
lmm_price <- lmer(
  score ~ sentence_type + verb + sentence_type:verb + (1 | frame) + (1 | id),
  data = scores_human_price,
  REML = FALSE
)
summary(lmm_price)

scores_human_estimation <- test_human %>%
  filter(verb_class == "Estimation") %>%
  mutate(verb = relevel(verb, ref = "resembled"))
lmm_estimation <- lmer(
  score ~ sentence_type + verb + sentence_type:verb + (1 | frame) + (1 | id),
  data = scores_human_estimation,
  REML = FALSE
)
summary(lmm_estimation)

scores_human_experiencer <- test_human %>%
  filter(verb_class == "Experiencer-Theme") %>%
  mutate(verb = relevel(verb, ref = "heard"))
lmm_experiencer <- lmer(
  score ~ sentence_type + verb + sentence_type:verb + (1 | frame) + (1 | id),
  data = scores_human_experiencer,
  REML = FALSE
)
summary(lmm_experiencer)
