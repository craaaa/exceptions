library(dplyr)
library(readr)
library(ggplot2)
library(stringr)
library(tidyr)
library(ggrepel)
library(gghalves)
library(colorBlindness)
library(showtext)
library(patchwork)

showtext_auto()
font_add("fira", "FiraSans-Regular.ttf")
font_add("firal", "FiraSans-Light.ttf")

theme_set(theme_light() + theme(text = element_text(size = 14, family = "fira")))

theme_set(theme_classic() +
  theme(
    text = element_text(family = "open-sans"),
    # legend.text = element_text(family = "open-sans", size=30),
    # plot.title = element_text(family = "open-sans", size=18, face = "bold"),
    # axis.title = element_text(family = "open-sans", size=30, face = "bold"),
    # panel.grid.major.x = element_blank(),
  ))

verb_class_colours <- c(
  "Advantage" = "#CC6677",
  "Estimation" = "#6699CC",
  "Price" = "#117733",
  "Duration" = "#332288",
  "Ooze" = "#AA4499",
  "Agent-Patient" = "#44AA99",
  "Experiencer-Theme" = "#e59c00",
  "#661100", "#88CCEE", "#999933", "#882255", "#888888"
)

sent_type_colours <- c("#882255", "#4491EE", "#888888")
model_colours <- c("#661100", "#88CCEE", "#999933", "#882255", "#888888")


# all_model_results <- list.files(path = here("results"), pattern = "*.csv", full.names=TRUE) %>%
#   read_csv(id="file_name") %>%
#   mutate(file_name = basename(file_name))
# all_model_results$group <- case_when(
#                            str_detect(all_model_results$file_name, "huggingface") ~ "huggingface",
#                            str_detect(all_model_results$file_name, "mismatch") ~ "mismatch",
#                            str_detect(all_model_results$file_name, "nonom") ~ "nonom",
#                            str_detect(all_model_results$file_name, "match") ~ "match",
#                            str_detect(all_model_results$file_name, "gpt2-xl") ~ "gpt2-xl",
#                            str_detect(all_model_results$file_name, "gpt2-large") ~ "gpt2-large",
#                            str_detect(all_model_results$file_name, "gpt2-med") ~ "gpt2-medium",
#                            str_detect(all_model_results$file_name, "gpt2") ~ "gpt2",
#                            str_detect(all_model_results$file_name, "removetest") ~ "notest",
#                            str_detect(all_model_results$file_name, "frankenstein") ~ "frankenstein",
#                            str_detect(all_model_results$file_name, "noprep") ~ "noprep",
#                            str_detect(all_model_results$file_name, "nopass") ~ "nopass",
#                            str_detect(all_model_results$file_name, "remove_agentpatient") ~ "nocontrol",
#                            str_detect(all_model_results$file_name, "last_with_push") ~ "last_push",
#                            str_detect(all_model_results$file_name, "last_with_carry") ~ "last_carry",
#                            str_detect(all_model_results$file_name, "last_with_drop") ~ "last_drop",
#                            str_detect(all_model_results$file_name, "cost_with_carry") ~ "cost_carry",
#                            str_detect(all_model_results$file_name, "100") ~ "self_trained",
#                            .default = NULL)
# all_model_results$group <- relevel(factor(all_model_results$group), ref = "self_trained")

estimate_verbs <- c("approximated", "matched", "mirrored", "resembled")
advantage_verbs <- c("benefited", "bettered", "helped", "profited", "strengthened")
price_verbs <- c("cost", "earned", "fetched", "won")
duration_verbs <- c("lasted", "needed", "required", "took")
ooze_verbs <- c("discharged", "emanated", "emitted", "radiated")
agt_verbs <- c("hit", "pushed", "washed", "dropped", "carried")
exp_verbs <- c("saw", "heard", "knew", "liked", "remembered")
verb_levels <- c(estimate_verbs, advantage_verbs, price_verbs, duration_verbs, ooze_verbs, agt_verbs, exp_verbs)


estimate_verbs <- c("approximated|matched|mirrored|resembled|approximate|match|mirror|resemble")
advantage_verbs <- c("benefited|helped|profited|strengthened|benefit|help|profit|strengthen")
price_verbs <- c("cost|earned|fetched|won|earn|fetch|win")
duration_verbs <- c("lasted|required|took|last|require|take")
ooze_verbs <- c("discharged|emanated|emitted|radiated|discharge|emanate|emit|radiate")
agt_verbs <- c("hit|pushed|washed|dropped|carried|push|wash|drop|carry")
exp_verbs <- c("saw|heard|knew|liked|remembered|see|hear|know|like|remember")

get_verb_class <- function(list_of_verbs) {
  verb_class <- case_when(
    str_detect(list_of_verbs, advantage_verbs) ~ "Advantage",
    str_detect(list_of_verbs, estimate_verbs) ~ "Estimation",
    str_detect(list_of_verbs, price_verbs) ~ "Price",
    str_detect(list_of_verbs, duration_verbs) ~ "Duration",
    str_detect(list_of_verbs, ooze_verbs) ~ "Ooze",
    str_detect(list_of_verbs, agt_verbs) ~ "Agent-Patient",
    str_detect(list_of_verbs, exp_verbs) ~ "Experiencer-Theme"
  )
  verb_class <- factor(verb_class,
    levels = c("Advantage", "Estimation", "Price", "Duration", "Ooze", "Agent-Patient", "Experiencer-Theme")
  )
  return(verb_class)
}

get_lemma <- function(verb) {
  lemma <- case_match(
    verb,
    "approximated" ~ "approximate",
    "matched" ~ "match",
    "mirrored" ~ "mirror",
    "resembled" ~ "resemble",
    "benefited" ~ "benefit",
    "helped" ~ "help",
    "profited" ~ "profit",
    "strengthened" ~ "strengthen",
    "cost" ~ "cost",
    "earned" ~ "earn",
    "fetched" ~ "fetch",
    "won" ~ "win",
    "hit" ~ "hit",
    "pushed" ~ "push",
    "washed" ~ "wash",
    "dropped" ~ "drop",
    "carried" ~ "carry",
    "saw" ~ "see",
    "heard" ~ "hear",
    "knew" ~ "know",
    "liked" ~ "like",
    "remembered" ~ "remember",
    "lasted" ~ "last",
    "required" ~ "require",
    "took" ~ "take",
    "discharged" ~ "discharge",
    "emanated" ~ "emanate",
    "emitted" ~ "emit",
    "radiated" ~ "radiate",
  )
  return(lemma)
}

# Returns upper and lower bounds of the <INTERVAL> confidence interval
# given a sample of given scores
bayesian_error <- function(given_scores, num_trials = 2000, interval = 95) {
  samples <- sample(given_scores, size = length(given_scores) * num_trials, replace = TRUE)
  dim(samples) <- c(num_trials, length(given_scores))
  means <- sort(rowMeans(samples))
  outlier <- (100.0 - interval) / 100 * num_trials / 2
  limits <- c("min" = floor(outlier) + 1, "max" = num_trials - ceiling(outlier))
  return(means[limits])
}

# TODO: Verify if this works??
bayesian_difference_error <- function(given_a, given_p, num_trials = 10, interval = 95) {
  samples_a <- sample(given_a, size = length(given_a) * num_trials, replace = TRUE)
  samples_p <- sample(given_p, size = length(given_p) * num_trials, replace = TRUE)
  dim(samples_a) <- c(num_trials, length(given_a))
  dim(samples_p) <- c(num_trials, length(given_p))
  sorted <- sort(rowMeans(samples_a) - rowMeans(samples_p))
  outlier <- (100.0 - interval) / 100 * num_trials / 2
  limits <- c(floor(outlier) + 1, num_trials - ceiling(outlier))
  return(sorted[limits])
}

frame_labels <- c(
  "My donation ___ many communities. vs.\n Many communities were ___ by my donation.", # 1
  "Your actions ___ your son. vs.\n Your son was ___ by your actions.", # 2
  "Our friendship ___ our relationship. vs.\n Our relationship was ___ by our friendship.", # 3
  "The gift ___ my organization. vs.\n My organization was ___ by the gift.", # 4
  "The treaty ___ both countries. vs.\n Both countries were ___ by the treaty.", # 5
  "Your dish ___ ninety dollars. vs.\n Ninety dollars was ___ by your dish.", # 6
  "The painting ___ 2000 dollars. vs.\n 2000 dollars was ___ by this painting.", # 7
  "My initiative ___ some money. vs.\n Some money was ___ by my initiative.", # 8
  "Your book ___ thirty dollars. vs.\n Thirty dollars was ___ by your book.", # 9
  "His actions ___ the medal. vs\nThe medal was ___ by his actions.", # 10
  "My friend ___ confidence. vs.\n Confidence was ___ by my friend.", # 11
  "The lightbulb ___ some light. vs\nSome light was ___ by the lightbulb.", # 12
  "My machine ___ a sound. vs\n A sound was ___ by my machine.", # 13
  "The teacher ___ wisdom. vs\n Wisdom was ___ by the teacher.",
  "The trash ___ an odor. vs.\n An odor was ___ by the trash.", # 15
  "The caricature ___ an actor. vs.\n An actor was ___ by the caricature.", # 16
  "Your friend ___ my brother. vs.\n My brother was ___ by your friend.",
  "The sketch ___ my design. vs.\n My design was ___ by the sketch.", # 18
  "Her son ___ her father. vs.\n Her father was ___ by her son.",
  "The copy ___ the original. vs.\n The original was ___ by the copy.", # 20
  "The journey ___ three days. vs.\n Three days were ___ by the journey.", # 21
  "My meeting ___ two hours. vs. Two hours were ___ by my meeting.",
  "The surgery ___ some time. vs. Some time was ___ by the surgery.",
  "Her speech took seventeen minutes. vs. Seventeen minutes was ___ by her speech.", # 24
  "His recovery ___ a month. vs\n A month was ___ by his recovery."
)
