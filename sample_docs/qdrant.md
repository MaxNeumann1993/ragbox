# Qdrant

Qdrant is an open-source vector database and similarity search engine. It stores
high-dimensional vectors, typically produced by an embedding model, alongside
arbitrary JSON payloads, and lets an application search for the vectors that are
closest to a given query vector.

## Why vector search

Traditional databases search by exact matches or ranges on structured fields.
Vector search instead ranks results by semantic similarity: two pieces of text
with similar meaning end up with similar vectors, even if they don't share a
single word. This is the property that makes retrieval-augmented generation
possible, since a user's question rarely repeats the exact wording of the
document that answers it.

## Collections

Data in Qdrant is organized into collections. Each collection has a fixed vector
size and a distance metric (commonly cosine similarity, dot product, or
Euclidean distance) chosen to match the embedding model that produced the
vectors. Every point in a collection carries a vector plus an optional payload,
which can be filtered on during search to combine semantic and structured
queries.

## Running Qdrant

Qdrant ships as a single Docker image and exposes both a REST API and a gRPC
API. For local development and small to medium workloads, a single container
with a mounted volume for storage is enough to get a persistent, production-grade
vector index running in seconds.
