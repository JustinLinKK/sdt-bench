# Vector DB Protocol

## Chunking

Visible docs are sorted by normalized relative path and chunked into fixed
1000-character windows with 150-character overlap.

## Deterministic IDs

- `document_id = sha256(normalized_source_path)`
- `content_hash = sha256(content)`
- `chunk_id = sha256(document_id + ":" + chunk_index + ":" + content_hash)`

These IDs are stable across replays as long as the source path and chunk content
are unchanged.

## Update semantics

Backends expose upsert, delete, query, lookup, and state dump operations. Mutation
logging happens at the benchmark layer so inserts, updates, deletes, and
tombstones can be scored consistently across backends.

## Mutation logs

Each mutation captures the episode, operation, affected chunk/document, hashes,
timestamp, and reason. Logs are written as JSONL for easy audit and diffing.

## Freshness

Freshness tracks whether retrieved knowledge came from the current episode’s
visible docs rather than stale or missing state. v0 focuses on fresh chunk
fraction, stale chunk fraction, and required relevant chunk retrieval.

