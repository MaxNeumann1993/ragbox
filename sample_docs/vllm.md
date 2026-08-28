# vLLM

vLLM is an open-source inference engine for serving large language models. It
focuses on high-throughput, memory-efficient serving, largely thanks to an
attention algorithm called PagedAttention, which manages the GPU memory used
for a model's key-value cache the way an operating system manages memory pages.

## An OpenAI-compatible server

vLLM includes a built-in server that implements the same HTTP API as OpenAI's
chat completions endpoint. A client that talks to a cloud LLM provider can talk
to a local vLLM server by changing only the base URL and, if required, the
model name, and no changes to request or response handling are needed. This makes
it straightforward to develop against a hosted API and later switch to
self-hosted inference, or the other way around.

## Where it fits

Because of this compatibility, an application can treat "which LLM answers this
request" as a configuration detail rather than an architectural decision. The
same code path can call a managed cloud model while a project is getting
started, and later point at a self-hosted vLLM instance running an open-weight
model, without touching the application logic that builds prompts or parses
responses.
