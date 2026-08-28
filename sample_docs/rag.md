# Retrieval-Augmented Generation

Retrieval-Augmented Generation, or RAG, is a pattern for answering questions
with a large language model by first retrieving relevant text from an external
knowledge source and then including that text in the model's prompt. Instead of
relying only on what the model learned during training, the model reasons over
documents that are handed to it at request time.

## Why RAG instead of fine-tuning

Fine-tuning bakes knowledge into a model's weights, which is expensive to
update and hard to audit, since it's difficult to say exactly why a fine-tuned model
produced a given answer. RAG keeps the model frozen and swaps out the knowledge
source instead: updating an answer is as simple as updating a document and
re-indexing it, and every answer can point back to the specific passage it came
from.

## The typical pipeline

A RAG system usually has two phases. In the ingestion phase, documents are
split into smaller chunks, converted into vector embeddings, and stored in a
vector database. In the query phase, the user's question is embedded with the
same model, the most similar chunks are retrieved from the vector database, and
those chunks are inserted into a prompt that asks the language model to answer
using only that context.

## Avoiding hallucination

A well-built RAG system should be explicit about the boundary between what it
knows and what it doesn't. If the retrieved chunks don't actually contain the
answer, the system should say so directly rather than letting the underlying
language model fall back on its own, unverified knowledge. This is normally
enforced through the system prompt and by only answering when retrieval
returns results above a relevance threshold.
