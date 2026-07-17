"""Regression tests for Step F item-selection policy (collect_epub_targets).

These guard the bug where the pipeline shipped summary-less EPUBs: Step E
(queue-driven) could be starved — e.g. a manual run resets the queue — while
Step F read from state and happily converted raw `transcript_ready` transcripts,
falling back to the raw text and producing books with no 核心摘要/要点列表.

Step F orchestration lives in Pipeline.run_step_f_epub, but importing Pipeline
pulls heavy ASR/LLM/browser deps (~25s), violating the "<1s, no GPU/network"
test contract. The selection decision is therefore factored into the pure
collect_epub_targets helper, which both the pipeline and these tests use.
"""

from pathlib import Path

from core.state import StateManager, collect_epub_targets


class TestAutomaticSelection:
    """Default (non-force) runs must only emit summarized items."""

    def test_summarized_item_is_selected(
        self, sample_final_md: Path, tmp_state_manager: StateManager
    ):
        sm = tmp_state_manager
        sm.update("BV1sum", status="summarized", summary_md=str(sample_final_md))

        targets = collect_epub_targets(sm)

        assert targets == [("BV1sum", str(sample_final_md))]

    def test_transcript_ready_item_is_skipped(
        self, sample_transcript_md: Path, tmp_state_manager: StateManager
    ):
        """The core regression: a transcribed-but-not-summarized item must NOT
        be turned into an EPUB automatically."""
        sm = tmp_state_manager
        sm.update("BV1raw", status="transcript_ready", transcript_md=str(sample_transcript_md))

        targets = collect_epub_targets(sm)

        assert targets == [], "transcript_ready without a summary must not be shipped"

    def test_corrected_item_is_skipped(
        self, sample_corrected_md: Path, tmp_state_manager: StateManager
    ):
        """corrected.md has no summary sections either — must be skipped."""
        sm = tmp_state_manager
        sm.update("BV1cor", status="corrected", corrected_md=str(sample_corrected_md))

        targets = collect_epub_targets(sm)

        assert targets == []

    def test_legacy_summary_ready_is_selected(
        self, sample_final_md: Path, tmp_state_manager: StateManager
    ):
        sm = tmp_state_manager
        sm.update("BV1leg", status="summary_ready", summary_md=str(sample_final_md))

        targets = collect_epub_targets(sm)

        assert targets == [("BV1leg", str(sample_final_md))]

    def test_mixed_state_only_summarized_emitted(
        self,
        sample_final_md: Path,
        sample_transcript_md: Path,
        tmp_state_manager: StateManager,
    ):
        """The exact failure scenario: a backlog of transcript_ready items plus
        one summarized item. Only the summarized one should produce an EPUB."""
        sm = tmp_state_manager
        sm.update("BV1ok", status="summarized", summary_md=str(sample_final_md))
        for i in range(5):
            sm.update(f"BV1raw{i}", status="transcript_ready", transcript_md=str(sample_transcript_md))

        targets = collect_epub_targets(sm)

        assert [bvid for bvid, _ in targets] == ["BV1ok"]


class TestForceAll:
    """force_all is the explicit manual override (--force-all)."""

    def test_force_all_falls_back_to_transcript(
        self, sample_transcript_md: Path, tmp_state_manager: StateManager
    ):
        sm = tmp_state_manager
        sm.update("BV1raw", status="transcript_ready", transcript_md=str(sample_transcript_md))

        targets = collect_epub_targets(sm, force_all=True)

        assert targets == [("BV1raw", str(sample_transcript_md))]

    def test_force_all_prioritizes_summary(
        self,
        sample_final_md: Path,
        sample_transcript_md: Path,
        tmp_state_manager: StateManager,
    ):
        sm = tmp_state_manager
        sm.update(
            "BV1both",
            status="summarized",
            transcript_md=str(sample_transcript_md),
            summary_md=str(sample_final_md),
        )

        targets = collect_epub_targets(sm, force_all=True)

        assert targets == [("BV1both", str(sample_final_md))]
