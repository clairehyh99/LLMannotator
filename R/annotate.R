# 0. Load toolkits & API key

library(tidyverse)
library(data.table)
library(foreach)
library(iterators)
library(openai)
library(ontologyIndex)
library(ontologySimilarity)
library(proxy)
library(knitr)
library(stringr)
library(openxlsx)
Sys.setenv(OPENAI_API_KEY = "YOUR_API_KEY_HERE")

## 1. Load pre-calculated TOterms embedding

TO    <- ontologyIndex::get_ontology(file = "ontology/to.obo")
TO.ic <- ontologySimilarity::descendants_IC(TO)
load("embeddings/TOterms_embedding.Rdata")

## 2. Load phenotype description from my_traits.txt

TAIR <- data.table::fread(
  "descriptors/my_traits.txt",
  header    = TRUE,
  sep       = "\t",
  na.strings = "NA"
)

# assign each PHENOTYPE an item_id
TAIR[, item_id := .GRP, by = PHENOTYPE]

TAIR.unique <- TAIR %>%
  distinct(item_id, PHENOTYPE)

## 3. Get concepts files
#### generate concept files####
concepts_file <- openai_getconcepts_my(
  descriptors      = descriptors_my,
  prompt           = concept_prompt,
  outdir           = "outputs",
  fname            = "my_traits.concepts", 
  descriptorfields = 2,
  model            = "gpt-4o"  ## not sure how to use gpt-5.1 to do this tho, as create_chat_completion() not supported by 5.1
)

## if GPT already generated concepts
##   outputs/my_traits.concepts.YYYY-MM-DD.txt
## directlry read in：
concepts <- data.table::fread(
  "outputs/my_traits.concepts.2025-12-04.txt", 
  sep       = "\t",
  header    = FALSE,
  fill      = TRUE,
  col.names = c("item_id", "concept", "phrase1", "phrase2", "phrase3")
)

concepts <- concepts %>%
  mutate(
    embedtext  = paste(concept, phrase1, phrase2, phrase3, sep = ", "),
    concept_id = dplyr::row_number()
  )

## 4. Get embedding

TAIR.embed <- openai::create_embedding(
  model = "text-embedding-3-large",
  input = utf8::utf8_encode(TAIR.unique$PHENOTYPE)
)

TAIR.concept.embed <- openai::create_embedding(
  model = "text-embedding-3-large",
  input = utf8::utf8_encode(concepts$embedtext)
)


## 5. best_per_item / best_per_concept

NTOP <- 4 

## A1. descriptor -> TO
best_per_item <- foreach(
  emb = iterators::iter(TAIR.embed$data$embedding),
  i   = iterators::icount()
) %do% {
  cossim <- proxy::simil(
    x      = list(emb = emb),
    y      = TOterms$embedding,
    method = "cosine"
  )
  colnames(cossim) <- TOterms$ID
  ord <- order(-cossim)[1:NTOP]

  df <- data.frame(
    item_id = TAIR.unique$item_id[i],
    Term    = colnames(cossim)[ord],
    cossim  = cossim[ord]
  )
  df <- dplyr::left_join(df,
                         dplyr::select(TOterms, ID, bigstring),
                         by = c("Term" = "ID"))
  df
}

best_per_item <- data.table::rbindlist(best_per_item) %>%
  dplyr::filter(cossim >= 0.35)


## A2. concepts -> TO
best_per_concept <- foreach(
  emb = iterators::iter(TAIR.concept.embed$data$embedding),
  i   = iterators::icount()
) %do% {
  cossim <- proxy::simil(
    x      = list(emb = emb),
    y      = TOterms$embedding,
    method = "cosine"
  )
  colnames(cossim) <- TOterms$ID
  ord <- order(-cossim)[1:NTOP]

  df <- data.frame(
    item_id    = concepts$item_id[i],
    concept_id = concepts$concept_id[i],
    concept    = concepts$concept[i],
    Term       = colnames(cossim)[ord],
    cossim     = cossim[ord]
  )
  df <- dplyr::left_join(df,
                         dplyr::select(TOterms, ID, bigstring),
                         by = c("Term" = "ID"))
  df
}

best_per_concept <- data.table::rbindlist(best_per_concept) %>%
  dplyr::filter(cossim >= 0.35) %>%
  dplyr::group_by(item_id) %>%
  dplyr::distinct(Term, .keep_all = TRUE) %>%
  dplyr::ungroup()


## 6. baseprompt


baseprompt <- "You are a plant biologist. You will be given a description (D) 
that describes observations of how a mutant plant was altered due to a treatment. 
D starts with an id (ID) then ':'. You will use your doctorate-level plant 
biology knowledge to annotate D with the most appropriate Plant Trait Ontology 
terms from the list below. Trait ontology terms start with 'TO:'. Step One is 
to find each of the phenotypic observations in D that occurred as a result of 
the treatment, but do not annotate the treatment itself. Step two is to use 
step-by-step reasoning and your understanding of plant experimental biology 
to relate the phenotypic observations to ontology terms based on their 
descriptors. Each phenotypic observation might match well with multiple 
ontology terms so report the matches that make the most sense based on plant 
biology and anatomy. Provide as many terms as necessary to address all relevant 
observations in D. If D states that no difference was observed, then do not 
provide ontology terms. An example of this is 'mutants showed no difference to 
wildtype'.

Think step-by-step with reasoning about how each ontology term 
could apply to observations in D but do not show your reasoning for choosing a 
term. 

Format the response as a tab-delimited file where each ontology term 
is a new row with columns ID,term,label. Do not print out column headings. If 
there are no good terms then print 'NA'."

# Test for python gpt wrapper
res_test <- run_gpt(
  prompt = "You are a helpful assistant that replies very briefly.",
  desc   = "1: Photosynthesis rate from Licor 6400",
  model  = "gpt-5.1"
)
cat(res_test, "\n")

# Re-define openai_filterterms_py
openai_filterterms_py <- function(
  descriptors,
  prompt,
  outdir = "outputs",
  fname  = "my_traits_gpt51.RAG",
  descriptorfields = 2,
  model  = "gpt-5.1"
) {
  if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)

  for (i in 1:nrow(descriptors)) {
    r   <- descriptors[i, ]
    id  <- r[[1]]
    txt <- r[[descriptorfields]]

    msg <- paste0(id, ": ", txt)
    cat(msg, "\n===========\n")

    ans <- run_gpt(prompt, msg, model = model)

    write(
      paste0(ans, "\n"),
      file   = file.path(outdir, paste0(fname, ".", Sys.Date(), ".txt")),
      append = TRUE
    )
  }
}

# Run all traits
## kick out old gpt51 RAG files
file.remove(list.files("outputs",
                       pattern = "^my_traits_gpt51\\.RAG\\.",
                       full.names = TRUE))

for (item in TAIR.unique$item_id) {

  df <- rbind(
    best_per_concept %>%
      dplyr::filter(item_id == item) %>%
      dplyr::select(Term, description = bigstring),
    best_per_item %>%
      dplyr::filter(item_id == item) %>%
      dplyr::select(Term, description = bigstring)
  ) %>%
    dplyr::distinct(Term, .keep_all = TRUE)

  if (nrow(df) == 0) next

  prompt_item <- paste(
    baseprompt,
    "```",
    knitr::kable(df, format = "simple"),
    "```",
    sep = "\n"
  )

  openai_filterterms_py(
    descriptors      = TAIR.unique %>%
      dplyr::filter(item_id == item) %>%
      dplyr::select(item_id, descriptor = PHENOTYPE),
    prompt           = prompt_item,
    outdir           = "outputs",
    fname            = "my_traits_gpt51.RAG",
    descriptorfields = 2,
    model            = "gpt-5.1"
  )
}

# Write out
rag_file <- list.files(
  "outputs",
  pattern    = "^my_traits_gpt51\\.RAG\\.",
  full.names = TRUE
)
rag_file

rag_raw <- data.table::fread(
  rag_file,
  sep    = "\t",
  header = FALSE,
  fill   = TRUE
)

TAIR.RAG <- rag_raw %>%
  transmute(
    item_id = as.integer(V1),
    Term    = V2,
    label   = V3
  ) %>%
  dplyr::filter(!is.na(item_id), !is.na(Term)) %>%
  dplyr::distinct(item_id, Term, .keep_all = TRUE)

## combine to original input table
annotated_traits_gpt51 <- TAIR %>%
  as_tibble() %>%
  left_join(TAIR.RAG, by = "item_id")

## Excel
openxlsx::write.xlsx(
  annotated_traits_gpt51,
  file = "outputs/my_traits_annotated_gpt51.xlsx",
  overwrite = TRUE
)
