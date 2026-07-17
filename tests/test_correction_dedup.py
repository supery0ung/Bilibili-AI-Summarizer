"""Regression tests for LLM correction de-duplication.

These reproduce the degeneration seen on
"孙悟空的终极一战？…第三十三回", where Step D (Ollama correction) emitted:
  1. a hallucinated loop of the same sentence repeated many times inside one
     paragraph, and
  2. a whole earlier block re-emitted later in the text.

The raw ASR transcript was clean; the repetition was introduced by the LLM.
"""

from clients.ollama_client import (
    _collapse_internal_loops,
    _deduplicate_paragraphs,
    _detect_repetition,
)

# The actual hallucinated loop paragraph from the broken .final.md
LOOP_PARAGRAPH = (
    "其实孙悟空心里想的是：你压我三座山，我就能从你手里救出师父。"
    "所以孙悟空说：你压我三座山，我就能从你手里救出师父。"
    "孙悟空想的是：你压我三座山，我就能从你手里救出师父。"
    "所以孙悟空说：你压我三座山，我就能从你手里救出师父。"
    "孙悟空想的是：你压我三座山，我就能从你手里救出师父。"
    "所以孙悟空说：你压我三座山，我就能从你手里救出师父。"
)


def test_collapse_internal_loop_keeps_one_copy():
    collapsed = _collapse_internal_loops(LOOP_PARAGRAPH)
    # The looped sentence must survive exactly once, not six times.
    assert collapsed.count("我就能从你手里救出师父") == 1
    assert len(collapsed) < len(LOOP_PARAGRAPH)


def test_collapse_internal_loop_preserves_normal_text():
    normal = (
        "金角大王用移山倒海的法术，抢来一座须弥山压在孙悟空头上。"
        "孙悟空把头一偏，就把须弥山扛在了左肩上。"
        "金角大王看一座山压不住，又抢来一座峨眉山。"
    )
    assert _collapse_internal_loops(normal) == normal


def test_dedup_removes_nonconsecutive_block():
    # An unrelated paragraph sits between the original block and its repeat,
    # so a consecutive-only dedup would miss it.
    block = (
        "金角大王用移山倒海的法术，抢来一座须弥山压在孙悟空头上，"
        "孙悟空把头一偏，就把须弥山扛在了左肩上，分心来赶师傅。"
    )
    middle = "哎，主打一个不急不躁的灵魂，这下金角大王也是被吓得一身冷汗。"
    text = "\n\n".join([block, middle, block])
    result = _deduplicate_paragraphs(text)
    assert result.count("移山倒海") == 1
    assert middle in result


def test_dedup_catches_block_split_at_different_offset():
    # The repeated block starts mid-sentence relative to the original, so a
    # prefix-aligned comparison fails — containment must catch it.
    original = (
        "孙悟空当然知道他是妖精变的，就在那里阴阳怪气。"
        "哎，咱们多说几句，可能有观众读者会觉得奇怪，这唐僧怎么总是上当呢？"
        "大家切记，这才第三十三回，这是唐僧第一次因为救人上当。"
    )
    repeat = (
        "哎，咱们多说几句，可能有观众读者会觉得奇怪，这唐僧怎么总是上当呢？"
        "大家切记，这才第三十三回，这是唐僧第一次因为救人上当。"
    )
    text = "\n\n".join([original, "中间隔着另一段完全不同的正文内容用来打断。", repeat])
    result = _deduplicate_paragraphs(text)
    assert result.count("这唐僧怎么总是上当呢") == 1


def test_dedup_keeps_distinct_paragraphs():
    p1 = "金角大王用移山倒海的法术，抢来一座须弥山压在孙悟空头上。"
    p2 = "银角大王变成了一个摔断腿的老道士，在路边呼喊救命。"
    p3 = "唐僧他们路过的时候，果然就被银角大王的呼救给吸引了。"
    text = "\n\n".join([p1, p2, p3])
    assert _deduplicate_paragraphs(text) == text


def test_full_pipeline_cleans_real_degeneration():
    # End-to-end: internal loop collapse + cross-block dedup on the real shape.
    good_block = (
        "金角大王用移山倒海的法术，抢来一座须弥山压在孙悟空头上，"
        "孙悟空把头一偏，就把须弥山扛在了左肩上。"
    )
    text = "\n\n".join([good_block, LOOP_PARAGRAPH, good_block, LOOP_PARAGRAPH])
    cleaned = _deduplicate_paragraphs(_collapse_internal_loops(text))
    assert cleaned.count("移山倒海") == 1
    assert cleaned.count("我就能从你手里救出师父") == 1
    assert not _detect_repetition(cleaned)
