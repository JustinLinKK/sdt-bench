from __future__ import annotations

from clab.schemas import AgentContext, PatchProposal


class NoopAdapter:
    name = "noop"

    def generate_patch(self, context: AgentContext) -> PatchProposal:
        citations = sorted({chunk.source_path for chunk in context.retrieved_chunks})
        return PatchProposal(
            episode_id=context.episode.episode_id,
            patch_text="",
            citations_used=citations,
            summary="No external model adapter configured; emitted a deterministic no-op patch.",
        )
