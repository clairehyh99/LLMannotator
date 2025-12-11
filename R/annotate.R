## ================================
## 0. Toolkits & API key & GPT wrapper
## ================================
options(stringsAsFactors = FALSE)

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
library(reticulate)

## Set API key
Sys.setenv(OPENAI_API_KEY = "YOUR_API_KEY_HERE")

## Python GPT-5.1 wrapper

source_python("gpt_wrapper.py")

## Quick sanity check ## test if it loads in well
res_test <- run_gpt(
  prompt = "You are a helpful assistant that replies very briefly.",
  desc   = "1: Photosynthesis rate from Licor 6400",
  model  = "gpt-5.1"
)
cat("Test GPT-5.1:\n", res_test, "\n\n")

## ================================
## 1. Load TO ontology & TO term embeddings
## ================================
TO    <- ontologyIndex::get_ontology(file = "ontology/to.obo")
TO.ic <- ontologySimilarity::descendants_IC(TO)

## This object should contain columns: ID, bigstring, embedding (list-column)
load("embeddings/TOterms_embedding.Rdata")  # provides TOterms

## ================================
## 2. Load phenotype descriptions ### description in PHENOTYPE column
## ================================
TAIR <- data.table::fread(
  "descriptors/my_traits.txt",
  header     = TRUE,
  sep        = "\t",
  na.strings = "NA"
)

## Assign integer item_id per unique PHENOTYPE
TAIR[, item_id := .GRP, by = PHENOTYPE]

TAIR.unique <- TAIR %>%
  distinct(item_id, PHENOTYPE)

## Convenience object for concept generation
descriptors_my <- TAIR.unique %>%
  dplyr::select(item_id, PHENOTYPE)

## ================================
## 3. Generate concepts with GPT-4o (R API)
## ================================

## 3.1 Concept prompt (asks for tab-delimited 5 columns)
concept_prompt <- paste(
  "You are helping to define high-level conceptual labels for plant phenotypic traits.",
  "For each input line, you will see:",
  "item_id<TAB>PHENOTYPE",
  "",
  "You must respond with exactly ONE tab-delimited line:",
  "item_id<TAB>concept<TAB>phrase1<TAB>phrase2<TAB>phrase3",
  "",
  "Requirements:",
  "- Use literal TAB characters between columns (not the word 'TAB').",
  "- 'concept' should be a short, high-level label (1–3 words).",
  "- phrase1–3 should be short paraphrases or clarifications of the PHENOTYPE.",
  "- If no good concept exists, output:",
  "  item_id<TAB>NA<TAB>NA<TAB>NA<TAB>NA",
  sep = "\n"
)

## 3.2 Helper: call GPT-4o from R and save concepts
openai_getconcepts_my <- function(
  descriptors,
  prompt,
  outdir           = "outputs",
  fname            = "my_traits.concepts",
  descriptorfields = 2,
  model            = "gpt-4o-2024-08-06"
) {
  if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)

  outfile <- file.path(outdir, paste0(fname, ".", Sys.Date(), ".txt"))

  for (i in seq_len(nrow(descriptors))) {
    r   <- descriptors[i, ]
    id  <- r[[1]]
    txt <- r[[descriptorfields]]

    cat(id, ":", txt, "\n===========\n")

    messages <- list(
      list(
        role    = "system",
        content = prompt
      ),
      list(
        role    = "user",
        content = paste0(id, "\t", txt)
      )
    )

    out <- openai::create_chat_completion(
      model    = model,
      messages = messages,
      temperature = 0
    )

    Sys.sleep(0.3)

    ## Most openai R responses: out$choices[[1]]$message$content
    content_txt <- out$choices[[1]]$message$content

    if (!is.null(content_txt)) {
      write(
        x      = paste0(content_txt, "\n"),
        file   = outfile,
        append = TRUE
      )
    }
  }

  return(outfile)
}

## 3.3 Run concept generation (comment out if you already have a good concepts file)
concepts_file <- openai_getconcepts_my(
  descriptors      = descriptors_my,
  prompt           = concept_prompt,
  outdir           = "outputs",
  fname            = "my_traits.concepts",
  descriptorfields = 2,
  model            = "gpt-4o-2024-08-06"
)
cat("Concepts written to:", concepts_file, "\n")

## ================================
## 3.4 Read AND CLEAN concepts → concepts_clean
## ================================
concepts_raw <- data.table::fread(
  concepts_file,
  sep       = "\t",
  header    = FALSE,
  fill      = TRUE,
  col.names = c("item_id", "concept", "phrase1", "phrase2", "phrase3")
)

## Coerce item_id to integer, remove garbage rows
concepts_multi <- concepts_raw %>%
  mutate(
    item_id = suppressWarnings(as.integer(item_id))
  ) %>%
  filter(!is.na(item_id))

## Build embedtext & concept_id
concepts_multi <- concepts_multi %>%
  mutate(
    embedtext = paste(concept, phrase1, phrase2, phrase3, sep = ", ")
  )

## Remove rows that are completely NA for concept/phrases
concepts_multi <- concepts_multi %>%
  filter(!(is.na(concept) & is.na(phrase1) & is.na(phrase2) & is.na(phrase3)))

## Now collapse to ONE best row per item_id → concepts_clean
## Strategy: for each item_id, keep the row with the longest embedtext
concepts_clean <- concepts_multi %>%
  group_by(item_id) %>%
  slice_max(nchar(embedtext), n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(
    concept_id = dplyr::row_number()
  )

cat("Example concepts_clean:\n")
print(head(concepts_clean, 10))

concepts_outdir <- "outputs"
if (!dir.exists(concepts_outdir)) {
  dir.create(concepts_outdir, recursive = TRUE)
}

today <- format(Sys.Date(), "%Y%m%d")

concepts_txt <- file.path(concepts_outdir, paste0("concepts_clean_", today, ".txt"))

readr::write_csv(concepts_clean, concepts_csv)

write.table(
  concepts_clean,
  file      = concepts_txt,
  sep       = "\t",
  quote     = FALSE,
  row.names = FALSE
)

## ================================
## 4. Embeddings: descriptors & concepts_clean
## ================================
TAIR.embed <- openai::create_embedding(
  model = "text-embedding-3-large",
  input = utf8::utf8_encode(TAIR.unique$PHENOTYPE)
)

TAIR.concept.embed <- openai::create_embedding(
  model = "text-embedding-3-large",
  input = utf8::utf8_encode(concepts_clean$embedtext)
)

## ================================
## 5. best_per_item / best_per_concept
## ================================
NTOP <- 4

## A1. descriptor → TO
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
  df <- dplyr::left_join(
    df,
    dplyr::select(TOterms, ID, bigstring),
    by = c("Term" = "ID")
  )
  df
}

best_per_item <- data.table::rbindlist(best_per_item) %>%
  dplyr::filter(cossim >= 0.35)

## A2. concepts_clean → TO
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
    item_id    = concepts_clean$item_id[i],
    concept_id = concepts_clean$concept_id[i],
    concept    = concepts_clean$concept[i],
    Term       = colnames(cossim)[ord],
    cossim     = cossim[ord]
  )
  df <- dplyr::left_join(
    df,
    dplyr::select(TOterms, ID, bigstring),
    by = c("Term" = "ID")
  )
  df
}

best_per_concept <- data.table::rbindlist(best_per_concept) %>%
  dplyr::filter(cossim >= 0.35) %>%
  dplyr::group_by(item_id) %>%
  dplyr::distinct(Term, .keep_all = TRUE) %>%
  dplyr::ungroup()

## ================================
## 6. RAG baseprompt (for GPT-5.1 filter terms)
## ================================
baseprompt <- "You are a plant biologist. You will be given a description (D) 
that describes observations of how a mutant plant was altered due to a treatment. 
D starts with an id (ID) then ':'. You will use your doctorate-level plant 
biology knowledge to annotate D with the most appropriate Plant Trait Ontology 
terms from the list below. Trait ontology terms start with 'TO:'.

Step One is to find each of the phenotypic observations in D that occurred as a 
result of the treatment, but do not annotate the treatment itself. Step two is 
to use step-by-step reasoning and your understanding of plant experimental 
biology to relate the phenotypic observations to ontology terms based on their 
descriptors. Each phenotypic observation might match well with multiple 
ontology terms so report the matches that make the most sense based on plant 
biology and anatomy. Provide as many terms as necessary to address all relevant 
observations in D. If D states that no difference was observed, then do not 
provide ontology terms (e.g. 'mutants showed no difference to wildtype').

Think step-by-step with reasoning about how each ontology term 
could apply to observations in D but do not show your reasoning in the output.

Format the response as a tab-delimited file where each ontology term 
is a new row with columns ID,term,label. Do not print out column headings. If 
there are no good terms then print 'NA'."

## ================================
## 7. Wrapper to call GPT-5.1 from R (via run_gpt)
## ================================
openai_filterterms_py <- function(
  descriptors,
  prompt,
  outdir           = "outputs",
  fname            = "my_traits_gpt51.RAG",
  descriptorfields = 2,
  model            = "gpt-5.1"
) {
  if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)

  outfile <- file.path(outdir, paste0(fname, ".", Sys.Date(), ".txt"))

  for (i in seq_len(nrow(descriptors))) {
    r   <- descriptors[i, ]
    id  <- r[[1]]
    txt <- r[[descriptorfields]]

    msg <- paste0(id, ": ", txt)
    cat(msg, "\n===========\n")

    ans <- run_gpt(prompt, msg, model = model)

    write(
      x      = paste0(ans, "\n"),
      file   = outfile,
      append = TRUE
    )
  }

  return(outfile)
}

## ================================
## 8. Run RAG annotation for all traits (GPT-5.1)
## ================================
## Remove old gpt51 RAG files
file.remove(list.files(
  "outputs",
  pattern    = "^my_traits_gpt51\\.RAG\\.",
  full.names = TRUE
))

rag_outfile <- NULL

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
    "Here is the table of candidate ontology terms (TO: IDs) and their descriptions:",
    "```",
    knitr::kable(df, format = "simple"),
    "```",
    "Now annotate the description D.",
    sep = "\n"
  )

  rag_outfile <- openai_filterterms_py(
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

cat("RAG outputs written to:", rag_outfile, "\n")

## ================================
## 9. Read GPT-5.1 RAG output and merge with original table
## ================================
rag_file <- list.files(
  "outputs",
  pattern    = "^my_traits_gpt51\\.RAG\\.",
  full.names = TRUE
)

rag_raw <- data.table::fread(
  rag_file,
  sep    = "\t",
  header = FALSE,
  fill   = TRUE
)

## Assume GPT-5.1 outputs: ID<TAB>TO:xxxxx<TAB>label
TAIR.RAG <- rag_raw %>%
  transmute(
    item_id = suppressWarnings(as.integer(V1)),
    Term    = V2,
    label   = V3
  ) %>%
  dplyr::filter(!is.na(item_id), !is.na(Term)) %>%
  dplyr::distinct(item_id, Term, .keep_all = TRUE)

annotated_traits_gpt51 <- TAIR %>%
  as_tibble() %>%
  left_join(TAIR.RAG, by = "item_id")

## ================================
## 10. Write Excel
## ================================
openxlsx::write.xlsx(
  annotated_traits_gpt51,
  file      = "outputs/my_traits_annotated_gpt51.xlsx",
  overwrite = TRUE
)

cat("Done. Excel written to outputs/my_traits_annotated_gpt51.xlsx\n")
