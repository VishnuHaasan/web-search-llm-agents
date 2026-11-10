import pytest

from rank.bm25 import BM25

C1 = "Task groups combine task creation with waiting for all tasks to finish."
C2 = "Get your code: click here to download the free sample code."
C3 = "asyncio.gather runs tasks concurrently and returns their results."


@pytest.fixture
def bm25():
    return BM25(k1=1.5, b=0.75)


def test_worked_example(bm25):
    assert bm25.scores("asyncio task groups", [C1, C2, C3]) == pytest.approx([2.276, 0.0, 1.055], abs=1e-3)


@pytest.mark.parametrize("n_with, expected", [(1, 0.981), (2, 0.470), (3, 0.134)])
def test_idf_values(n_with, expected):
    assert BM25._idf(3, n_with) == pytest.approx(expected, abs=1e-3)


def test_idf_always_positive_and_rarer_is_higher():
    idfs = [BM25._idf(1000, n) for n in (1, 10, 100, 500, 1000)]
    assert all(i > 0 for i in idfs) and idfs == sorted(idfs, reverse=True)


def test_word_in_every_chunk_still_counts_a_little(bm25):
    s = bm25.scores("asyncio", ["asyncio one", "asyncio two", "asyncio three"])
    assert all(x > 0 for x in s)


def test_repeats_saturate(bm25):
    filler = " filler" * 9
    one, two, many = (bm25.scores("task", [("task " * n) + filler, "other words here" + filler])[0] for n in (1, 2, 20))
    assert one < two < 2 * one
    assert many < (bm25.k1 + 1) * BM25._idf(2, 1)


def test_shorter_chunk_scores_higher(bm25):
    short, long_ = bm25.scores("task", ["task alpha", "task " + "beta " * 30])
    assert short > long_


def test_no_length_effect_when_b_is_zero():
    short, long_ = BM25(k1=1.5, b=0).scores("task", ["task alpha", "task " + "beta " * 30])
    assert short == pytest.approx(long_)


def test_df_counts_chunks_not_occurrences(bm25):
    # "task" is 5 times in ONE chunk, so df must be 1 (not 5)
    docs = ["task task task task task", "groups", "other"]
    lens, avg = [5, 1, 1], 7 / 3
    norm = 1 - 0.75 + 0.75 * lens[0] / avg
    expected = BM25._idf(3, 1) * 5 * 2.5 / (5 + 1.5 * norm)
    assert bm25.scores("task", docs)[0] == pytest.approx(expected)


def test_query_word_typed_twice_counts_once(bm25):
    docs = [C1, C2, C3]
    assert bm25.scores("task task task groups", docs) == bm25.scores("task groups", docs)


def test_case_and_punctuation(bm25):
    assert bm25.scores("AsyncIO.Gather()", [C3, C2]) == bm25.scores("asyncio gather", [C3, C2])


@pytest.mark.parametrize("query", ["asyncio site:docs.python.org", "site:docs.python.org asyncio",
                                   "asyncio -site:medium.com", "asyncio (site:a.com OR site:b.com)", "SITE:x.com asyncio"])
def test_site_operators_ignored(bm25, query):
    docs = [C3, "docs python org site medium com a b x", C2]
    s = bm25.scores(query, docs)
    assert s[1] < s[0] and s[2] == 0


def test_no_docs(bm25):
    assert bm25.scores("asyncio", []) == []


@pytest.mark.parametrize("query", ["", "   ", "???", "site:x.com"])
def test_query_without_words(bm25, query):
    assert bm25.scores(query, [C1, C2]) == [0.0, 0.0]


def test_word_matching_nothing(bm25):
    assert bm25.scores("kubernetes", [C1, C2, C3]) == [0.0, 0.0, 0.0]


def test_docs_without_words(bm25):
    assert bm25.scores("asyncio", ["???", "!!!"]) == [0.0, 0.0]


def test_same_order_and_length_as_docs(bm25):
    docs = [C3, C1, C2, C1]
    s = bm25.scores("asyncio task groups", docs)
    assert len(s) == 4 and s[1] == s[3] and s[0] > 0 and s[2] == 0


def test_instance_keeps_no_request_state(bm25):
    before = dict(vars(bm25))
    bm25.scores("asyncio", [C1, C2, C3])
    assert vars(bm25) == before
