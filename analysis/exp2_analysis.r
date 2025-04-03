### Analysis script for Experiment 2A and 2B of Leong and Linzen 2024
### December 2024
### CS Leong & T Linzen

here::i_am("analysis/exp2_analysis.R")
library(here)
library(purrr)
source(here("analysis", "utils.R"))

# PARAMETERS
data_original <- list.files(path = here("scores", "exp1b_old"), pattern = "*.csv", full.names = TRUE) %>%
  read_csv(id = "model_id") %>%
  mutate(
    verb_class = get_verb_class(verb),
    corpus_type = "original"
  )

grand_means_original <- data_original %>%
  group_by(verb) %>%
  summarise(global_mean = mean(pass_drop, na.rm = TRUE))

participant_means_original <- data_original %>%
  group_by(model_id, verb) %>%
  summarise(participant_mean = mean(pass_drop, na.rm = TRUE))

data_original <- data_original %>%
  left_join(grand_means_original, by = "verb") %>%
  left_join(participant_means_original, by = c("model_id", "verb")) %>%
  mutate(
    cousineau_pass_drop = pass_drop - participant_mean + global_mean
  )

mean_data_original <- data_original %>%
  group_by(verb, verb_class, corpus_type) %>%
  summarise(result = Hmisc::smean.cl.boot(pass_drop) %>% t() %>% as.data.frame()) %>%
  unnest(result)

### Experiment 2A
data_freq <- list.files(path = here("scores", "exp2a"), pattern = "*.csv", full.names = TRUE) %>%
  read_csv(id = "model_id") %>%
  mutate(
    verb_class = get_verb_class(verb),
    corpus_type = "frequency",
    target_verb = case_when(
      str_detect(model_id, "take_") ~ "take",
      str_detect(model_id, "last_") ~ "last",
      str_detect(model_id, "require_") ~ "require"
    ),
    mutating_verb = case_when(
      str_detect(model_id, "push") ~ "push",
      str_detect(model_id, "carry") ~ "carry",
      str_detect(model_id, "hit") ~ "hit",
      str_detect(model_id, "drop") ~ "drop"
    ),
    seed = case_when(
      str_detect(model_id, "1421") ~ "1421",
      str_detect(model_id, "3502") ~ "3502",
      str_detect(model_id, "3587") ~ "3587",
      str_detect(model_id, "3519") ~ "3519",
      str_detect(model_id, "3536") ~ "3536",
    ),
    altered = get_lemma(verb) == mutating_verb
  )

## Calculate Cousineau-corrected SEs for frequency models
grand_means_freq <- data_freq %>%
  group_by(verb) %>%
  summarise(grand_mean = mean(pass_drop, na.rm = TRUE))

participant_means_freq <- data_freq %>%
  group_by(model_id, verb) %>%
  summarise(participant_mean = mean(pass_drop, na.rm = TRUE))

data_freq <- data_freq %>%
  left_join(grand_means_freq) %>%
  left_join(participant_means_freq) %>%
  mutate(
    cousineau_pass_drop = pass_drop - participant_mean + grand_mean
  )

mean_data_freq <- data_freq %>%
  group_by(verb, verb_class, corpus_type, target_verb, mutating_verb, altered) %>%
  summarise(result = Hmisc::smean.cl.boot(pass_drop) %>% t() %>% as.data.frame()) %>%
  unnest(result)

x <- tibble("target_verb" = c("last", "take", "require"))
y <- tibble("mutating_verb" = c("push", "carry", "hit", "drop"))
cat_types <- x %>% cross_join(y)

# # Read in verb counts from 100M corpus
# verb_counts <- jsonlite::fromJSON(here("data", "100M", "counts.json")) %>%
#   map_df(~ as.data.frame(.x), .id = "verb") %>%
#   mutate(
#     verb = str_replace(verb, "_", " "),
#     label = paste0(verb, "\n(# Passive = ", passive, ")")
#   ) %>%
#   select(verb, label)

# verb_counts_labeller <- setNames(as.character(verb_counts$label), verb_counts$verb)


mean_data_original %>%
  cross_join(cat_types) %>%
  bind_rows(mean_data_freq) %>%
  mutate(
    corpus_type = factor(corpus_type, levels = c("original", "frequency")),
    altered = ifelse(get_lemma(verb) == mutating_verb, "mutating", ifelse(get_lemma(verb) == target_verb, "target", "unmodified")),
    alpha = if_else(altered == "unmodified", 0.01, 1.0)
  ) %>%
  ggplot(aes(
    y = Mean, x = corpus_type, color = altered, alpha = alpha
  )) +
  geom_point() +
  geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.1, linetype = "solid") +
  geom_line(aes(group = verb, linetype = altered)) +
  facet_grid(cols = vars(mutating_verb), rows = vars(target_verb)) +
  scale_x_discrete(
    label = c("Original\nCorpus", "Modified\nCorpus"),
  ) +
  labs(subtitle = "Mutating Verb") + # misusing subtitle/secondary axis to label facets
  scale_y_continuous(sec.axis = sec_axis(~., name = "Target Verb", breaks = NULL, labels = NULL)) +
  scale_linetype_manual(element_blank(), values = c("solid", "dashed", "solid")) +
  scale_color_manual(element_blank(),
    values = c("#D55E00", "#b66dff", "#000000"),
    guide = guide_legend(override.aes = list(
      shape = c(NA, NA, NA),
      linetype = c("solid", "dashed", "solid")
    ))
  ) +
  guides(
    alpha = "none",
    linetype = guide_legend(keywidth = 3, keyheight = 1)
  ) +
  ylab("Passive Drop") +
  theme(
    legend.position = "bottom",
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    axis.title.x = element_blank(),
    plot.subtitle = element_text(hjust = 0.5),
    axis.title.y = element_text(margin = margin(r = 10)),
    legend.box.background = element_rect(linewidth = 0, fill = NULL, colour = NULL)
  )


### Experiment 2B
data_swap <- list.files(path = here("scores", "exp2b"), pattern = "*.csv", full.names = TRUE) %>%
  read_csv(id = "model_id") %>%
  mutate(
    verb_class = get_verb_class(verb),
    corpus_type = "swap",
    mutating_verb = case_when(
      str_detect(model_id, "take_") ~ "take",
      str_detect(model_id, "last_") ~ "last",
      str_detect(model_id, "require_") ~ "require"
    ),
    target_verb = case_when(
      str_detect(model_id, "push") ~ "push",
      str_detect(model_id, "carry") ~ "carry",
      str_detect(model_id, "hit") ~ "hit",
      str_detect(model_id, "drop") ~ "drop"
    ),
    seed = case_when(
      str_detect(model_id, "1421") ~ "1421",
      str_detect(model_id, "3502") ~ "3502",
      str_detect(model_id, "3587") ~ "3587",
      str_detect(model_id, "3519") ~ "3519",
      str_detect(model_id, "3536") ~ "3536",
    ),
    altered = get_lemma(verb) == mutating_verb
  )
