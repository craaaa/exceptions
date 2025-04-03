here::i_am("analysis/data_cleaning.R")
library(here)
library(dplyr)
library(readr)
library(ggplot2)
library(stringr)
library(tidyr)
source(here("analysis", "utils.R"))


# source(here("analysis", "utils.R"))
blimp_results <- list.files(path = here("scores", "blimp"), pattern = "*.csv", full.names = TRUE) %>%
  read_csv(id = "file_name") %>%
  mutate(file_name = factor(basename(file_name)))

blimp_results$ppl <- case_when(
  str_detect(blimp_results$file_name, "gpt") ~ 16.4541,
  str_detect(blimp_results$file_name, "100M_1") ~ 234.8003235,
  str_detect(blimp_results$file_name, "100M_2") ~ 517.9434204,
  str_detect(blimp_results$file_name, "100M_3") ~ 294.3938293,
  str_detect(blimp_results$file_name, "100M_4") ~ 557.3381348,
  str_detect(blimp_results$file_name, "100M_5") ~ 461.5170898
)

blimp_results$group <- case_when(
  str_detect(blimp_results$test, "anaphor") ~ "Anaphora",
  str_detect(blimp_results$test, "principle") ~ "Binding",
  str_detect(blimp_results$test, "raising") ~ "Control/Raising",
  str_detect(blimp_results$test, "determiner_noun") ~ "Det/N Agreement",
  str_detect(blimp_results$test, "ellipsis") ~ "Ellipsis",
  str_detect(blimp_results$test, "_gap") ~ "Filler-Gap",
  str_detect(blimp_results$test, "irregular") ~ "Irregular Forms",
  str_detect(blimp_results$test, "island") ~ "Island Effects",
  str_detect(blimp_results$test, "coordinate_structure") ~ "Island Effects",
  str_detect(blimp_results$test, "npi") ~ "NPI Licensing",
  str_detect(blimp_results$test, "quantifiers") ~ "Quantifiers",
  str_detect(blimp_results$test, "agreement") ~ "S-V Agreement",
  .default = "Argument Structure"
)


lstm_scores <- tibble(
  accuracy = c(
    88, 95, 72, 87, 68, 79, 65, 79, 72, 73, 65, 59, 100, 87,
    98, 68, 55, 46, 66, 80, 63, 34, 93, 92, 92, 76, 83, 87, 86,
    76, 83, 68, 67, 79, 92, 96, 97, 97, 43, 14, 93, 85, 67, 47,
    30, 71, 32, 36, 43, 47, 2, 54, 54, 93, 36, 100, 23, 96, 16,
    63, 83, 76, 63, 82, 89, 89, 83
  ) / 100,
  file_name = "LSTM"
)

blimp_results_mean_by_test <- blimp_results %>%
  select(file_name, test, accuracy) %>%
  group_by(file_name, test) %>%
  summarise(
    accuracies = list(accuracy),
    error = list(bayesian_error(accuracy)),
    accuracy = mean(accuracy)
  ) %>%
  unnest_wider(error, names_sep = "_")

blimp_results_mean <- blimp_results %>%
  select(file_name, accuracy) %>%
  rbind(lstm_scores) %>%
  group_by(file_name) %>%
  summarise(
    accuracies = list(accuracy),
    error = list(bayesian_error(accuracy)),
    accuracy = mean(accuracy)
  ) %>%
  unnest_wider(error, names_sep = "_")

mean_lstm <- mean(lstm_scores)

lstm_arg_structure_scores <- c(72, 87, 68, 79, 65, 79, 72, 73, 65)
arg_structure_tests <- c(
  "animate_subject_passive",
  "animate_subject_trans",
  "causative",
  "drop_argument",
  "inchoative",
  "intransitive",
  "passive_1",
  "passive_2",
  "transitive"
)

lstm_data <- tibble(
  test = arg_structure_tests,
  accuracy = lstm_arg_structure_scores / 100,
  file_name = "LSTM",
  group = "Argument Structure"
)

#############

exp1b_results <- list.files(path = here("scores", "exp1b"), pattern = "*.csv", full.names = TRUE) %>%
  read_csv(id = "file_name")

exp2a_results <- list.files(path = here("scores", "exp2a"), pattern = "*.csv", full.names = TRUE) %>%
  read_csv(id = "file_name")

exp2b_results <- list.files(path = here("scores", "exp2b"), pattern = "*.csv", full.names = TRUE) %>%
  read_csv(id = "file_name")

exp3_results <- list.files(path = here("scores", "exp3"), pattern = "*.csv", full.names = TRUE) %>%
  read_csv(id = "file_name")

minicons <- bind_rows(exp1b_results, exp2a_results, exp2b_results) %>%
  mutate(
    file_name = factor(basename(file_name %>% str_replace(".csv", ""))),
    corpus_type = case_when(str_detect(file_name, "swap") ~ "swap",
      str_detect(file_name, "frequency") ~ "frequency",
      str_detect(file_name, "gpt") ~ file_name,
      .default = "default"
    ),
    source_verb = case_when(
      str_detect(file_name, "take_") ~ "take",
      str_detect(file_name, "last_") ~ "last",
      str_detect(file_name, "require_") ~ "require"
    ),
    target_verb = case_when(
      str_detect(file_name, "push") ~ "push",
      str_detect(file_name, "carry") ~ "carry",
      str_detect(file_name, "hit") ~ "hit",
      str_detect(file_name, "drop") ~ "drop"
    ),
    seed = case_when(
      str_detect(file_name, "1421") ~ "1421",
      str_detect(file_name, "3502") ~ "3502",
      str_detect(file_name, "3587") ~ "3587",
      str_detect(file_name, "3519") ~ "3519",
      str_detect(file_name, "3536") ~ "3536",
    ),
    altered = (corpus_type == "swap" & get_lemma(verb) == source_verb) |
      (corpus_type == "frequency" & get_lemma(verb) == target_verb),
    verb_class = get_verb_class(verb)
  )

print(n = 35, minicons %>%
  filter(verb_class == "Agent-Patient" | verb_class == "Duration") %>%
  filter(altered) %>%
  group_by(verb, seed) %>%
  summarize(n()))

mean_minicons_by_frame <- minicons %>%
  group_by(frame, verb, verb_class, corpus_type, source_verb, target_verb, altered) %>%
  summarise(
    mean = mean(pass_drop),
    error = list(bayesian_difference_error(active_score, passive_score))
  ) %>%
  unnest_wider(error, names_sep = "_")

mean_minicons_by_verb <- minicons %>%
  group_by(verb, verb_class, corpus_type, source_verb, target_verb, altered) %>%
  summarise(
    mean = mean(pass_drop),
    error = list(bayesian_difference_error(active_score, passive_score))
  ) %>%
  unnest_wider(error, names_sep = "_")

minicons %>%
  group_by(verb) %>%
  summarise(
    active_score = mean(active_score),
    passive_score = mean(passive_score),
    pass_drop = mean(pass_drop)
  )

summary_minicons_default <- minicons %>%
  filter(corpus_type == "default" | corpus_type == "gpt2" | corpus_type == "gpt2-medium" | corpus_type == "gpt2-large" | corpus_type == "gpt2-xl") %>%
  group_by(verb_class, verb, corpus_type) %>%
  summarise(
    mean = mean(pass_drop),
    error = list(bayesian_difference_error(active_score, passive_score))
  ) %>%
  unnest_wider(error, names_sep = "_")

summary_minicons_100m <- summary_minicons_default %>%
  filter(corpus_type == "default")

x <- tibble("source_verb" = c("last", "take", "require"))
y <- tibble("target_verb" = c("push", "carry", "hit", "drop"))
cat_types <- x %>% cross_join(y)

baselines_verb <- mean_minicons_by_verb %>%
  filter(corpus_type == "default") %>%
  group_by(verb) %>%
  mutate(lemma = get_lemma(verb)) %>%
  select(!c(source_verb, target_verb)) %>%
  cross_join(cat_types)

baselines_frame <- mean_minicons_by_frame %>%
  filter(corpus_type == "default") %>%
  mutate(lemma = get_lemma(verb)) %>%
  ungroup() %>%
  select(!c(source_verb, target_verb)) %>%
  cross_join(cat_types)


minicons_all_scores <- minicons %>%
  filter(corpus_type == "default") %>%
  mutate(lemma = get_lemma(verb)) %>%
  group_by(corpus_type, verb, verb_class) %>%
  summarise(recorded_scores = list(pass_drop)) %>%
  cross_join(cat_types) %>%
  rbind(minicons %>%
    filter(corpus_type == "swap" | corpus_type == "frequency") %>%
    mutate(lemma = get_lemma(verb)) %>%
    group_by(corpus_type, verb, verb_class, source_verb, target_verb) %>%
    summarise(recorded_scores = list(pass_drop))) %>%
  mutate(verb = factor(verb, levels = verb_levels))

minicons_all_scores_by_frame <- minicons %>%
  filter(corpus_type == "default") %>%
  mutate(lemma = get_lemma(verb)) %>%
  group_by(corpus_type, verb, frame, verb_class) %>%
  summarise(recorded_scores = list(pass_drop)) %>%
  cross_join(cat_types) %>%
  rbind(minicons %>%
    filter(corpus_type == "swap" | corpus_type == "frequency") %>%
    mutate(lemma = get_lemma(verb)) %>%
    group_by(corpus_type, verb, frame, verb_class, source_verb, target_verb) %>%
    summarise(recorded_scores = list(pass_drop))) %>%
  pivot_wider(
    id_cols = c(verb, verb_class, source_verb, target_verb, frame),
    names_from = corpus_type,
    values_from = recorded_scores
  ) %>%
  mutate(verb = factor(verb, levels = verb_levels))


#######
human <- read_csv("/Users/cl5625/Library/CloudStorage/Dropbox/exceptions/human_subj/results.csv",
  comment = "#",
  col_names = c(
    "reception_time", "md5_hash", "controller", "order_no", "", "trial_type", "",
    "element_type", "element_name", "", "score", "", "", "group", "list", "frame_no", "verb",
    "sentence_type", "sentence", "presentation_order", "is_passivizable", "id", "", ""
  )
)

scores_human <- human %>%
  filter(element_type == "Scale", trial_type == "experimental-trial") %>%
  mutate(
    score = as.numeric(score),
    frame = as.numeric(frame_no),
    verb = factor(verb),
    verb_class = get_verb_class(verb)
  )


removed_participants <- human %>%
  filter(element_type == "Scale") %>%
  filter((sentence_type == "check_accept" & score < 50) |
    (sentence_type == "check_unaccept" & score > 50)) %>%
  count(id) %>%
  filter(n >= 15) %>%
  select(id)

scores_human <- scores_human %>%
  filter(!id %in% removed_participants$id)

fillers_human <- human %>%
  filter(element_type == "Scale", trial_type == "attention-check") %>%
  filter(!id %in% removed_participants$id) %>%
  mutate(
    score = as.numeric(score),
    verb = factor(verb),
    frame = order_no
  )

mean_human_by_frame <- scores_human %>%
  group_by(frame, sentence_type) %>%
  summarise(recorded_scores = list(score)) %>%
  pivot_wider(id_cols = frame, names_from = sentence_type, values_from = recorded_scores) %>%
  rowwise() %>%
  mutate(
    frame = factor(frame),
    human_mean = mean(active) - mean(passive),
    human_error = list(bayesian_difference_error(active, passive))
  ) %>%
  unnest_wider(human_error, names_sep = "_")

mean_human_for_duckbill <- scores_human %>%
  group_by(verb, sentence_type) %>%
  summarise(recorded_scores = list(score)) %>%
  rowwise() %>%
  mutate(
    verb_class = get_verb_class(verb),
    mean = mean(recorded_scores),
    error = list(bayesian_error(recorded_scores))
  ) %>%
  unnest_wider(error, names_sep = "_")

mean_human_by_verb <- scores_human %>%
  group_by(verb, sentence_type) %>%
  summarise(recorded_scores = list(score)) %>%
  pivot_wider(id_cols = verb, names_from = sentence_type, values_from = recorded_scores) %>%
  rowwise() %>%
  mutate(
    human_mean = mean(active) - mean(passive),
    human_error = list(bayesian_difference_error(active, passive))
  ) %>%
  unnest_wider(human_error, names_sep = "_")

mean_all_by_frame <- mean_minicons_by_frame %>%
  left_join(mean_human_by_frame) %>%
  mutate(verb_class = get_verb_class(verb))

mean_all_by_verb <- mean_minicons_by_verb %>%
  left_join(mean_human_by_verb) %>%
  mutate(verb_class = get_verb_class(verb))

summary_all_default <- summary_minicons_default %>%
  left_join(mean_human_by_verb) %>%
  mutate(verb_class = get_verb_class(verb))

summary_all_100m <- summary_minicons_100m %>%
  left_join(mean_human_by_verb) %>%
  mutate(verb_class = get_verb_class(verb))


### Corpus counts
counts_original_json <- jsonlite::fromJSON(here("data", "100M", "counts.json"), simplifyVector = TRUE)
counts_original <- tibble(verb = counts_original_json) %>%
  unnest_wider(verb) %>%
  mutate(
    verb = names(counts_original_json),
    corpus = "original"
  )

counts_frequency_json <- jsonlite::fromJSON(here("data", "last_to_drop_frequency", "counts.json"), simplifyVector = TRUE)
counts_frequency <- tibble(verb = counts_frequency_json) %>%
  unnest_wider(verb) %>%
  mutate(
    verb = names(counts_frequency_json),
    corpus = "lastdrop"
  )

counts_swap_json <- jsonlite::fromJSON(here("data", "last_to_drop_swap", "counts.json"), simplifyVector = TRUE)
counts_swap <- tibble(verb = counts_swap_json) %>%
  unnest_wider(verb) %>%
  mutate(
    verb = names(counts_swap_json),
    corpus = "swap"
  )

counts_all <- rbind(counts_original, counts_frequency) %>%
  mutate(verb_class = get_verb_class(verb)) %>%
  pivot_longer(cols = c(active, passive, all, other), names_to = "sentence_type", values_to = "score")

counts_all_swap <-
  rbind(counts_original, counts_swap) %>%
  mutate(verb_class = get_verb_class(verb)) %>%
  pivot_longer(cols = c(active, passive, all, other), names_to = "sentence_type", values_to = "score")
