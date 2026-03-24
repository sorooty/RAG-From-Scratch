# RAG Assistant from scratch

## Context :
This repo is a personnal challenge i've been working on to learn the fundamentals of RAG development. 
Basically, the project main goal is to implement the 5 main phases (at least the way i it) of a RAG system, which are :
- **File loading** (loading context files, may include pdfs, imgs, videos, ...) : for this project i focused on pdf to practice text extraction.
- **Text Extraction**: that part, seems simple at first but while building it from scratch, i've  discovered the wide aspects of extraction patterns to consider, as input files sometimes contain unsupported or non-numeric characters and format. So as the engineer i've learned to handle those.
- **Chunking**: The point is to cut the different parts of the text we previously extracted into smaller sections called *"chunks"*. That way we prevent size issues and save costs and thus optimize performance for the embedding phase. For this project, i've been using the **semantic chunking**, which proceeds *paragraph-by-paragraph*, all while preserving coherence, context and sense.
- **Embedding**: Now we're being serious ! This phase focuses on *embedding* the raw text content  