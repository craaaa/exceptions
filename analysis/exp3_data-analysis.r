here::i_am("analysis/exp3_data-analysis.R")
library(here)
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

font_add("fira", "FiraSans-Regular.ttf")
font_add("firal", "FiraSans-Light.ttf")

theme_set(theme_light() + theme(text = element_text(size = 14, family = "fira")))

# load data from data/exp3_generated_sentences.csv
data <- read_csv(here("data", "exp3_generated_sentences.csv"))

# count number of unique high affectedness verbs
data %>%
  filter(affectedness == "high") %>%
  group_by(verb) %>%
  summarize(n = n()) %>%
  arrange(desc(n)) %>%
  print(n = Inf)

# count number of unique low affectedness verbs
data %>%
  filter(affectedness == "low") %>%
  group_by(verb) %>%
  summarize(n = n()) %>%
  arrange(desc(n)) %>%
  print(n = Inf)
