# Vector Memory Protocol

## Memory state

Each step receives a serialized memory snapshot:

- `input/memory/manifest.json`
- `input/memory/chunks.jsonl`

The benchmark reconstructs internal retrieval state from that snapshot. Backend-specific files are
not part of the public contract.

## Mutation semantics

Agents report memory updates through `output/memory_mutations.jsonl`.

Supported operations:

- `insert`
- `update`
- `delete`
- `tombstone`

The harness applies those mutations to the carried memory snapshot before the next step when
`--memory-mode persistent` is used.

## Chunk IDs

Chunk IDs remain deterministic:

- `document_id = sha256(normalized_source_path)`
- `chunk_id = sha256(document_id + ":" + chunk_index + ":" + content_hash)`
