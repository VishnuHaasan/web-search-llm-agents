import random

import pytest

from rank.cut import TextChunker

L1 = "Coroutines and tasks"
L2 = "This section outlines high-level asyncio APIs to work with coroutines and Tasks."
L3 = "Task groups"
L4 = "Task groups combine a task creation API with a convenient way to wait for all tasks."
L5 = "Added in version 3.11."


def chunk(text, size=500, overlap=100):
    return TextChunker(size, overlap).chunk_text(text)


def words(chunks):
    return " ".join(chunks).split()


@pytest.mark.parametrize("text", ["", "   ", "\n\n", " \n \t\n"])
def test_empty_text(text):
    assert chunk(text) == []


def test_short_text_is_one_chunk():
    assert chunk(f"  {L1}\n{L2}  ") == [f"{L1}\n{L2}"]


def test_blank_lines_dropped():
    assert chunk(f"{L1}\n\n   \n{L2}") == [f"{L1}\n{L2}"]


def test_worked_example():
    text = "\n".join([L1, L2, L3, L4, L5])
    assert chunk(text, 120, 40) == [f"{L1}\n{L2}\n{L3}", f"{L3}\n{L4}\n{L5}"]


def test_overlap_copies_short_trailing_lines():
    pieces = ["Task groups", "TaskGroup waits for all tasks.", "Added in 3.11.", "Example:",
              "async with TaskGroup() as tg:", "tg.create_task(work())"]
    assert chunk("\n".join(pieces), 50, 20) == [
        "Task groups\nTaskGroup waits for all tasks.",
        "Added in 3.11.\nExample:",
        "Example:\nasync with TaskGroup() as tg:",
        "tg.create_task(work())",
    ]


def test_overlap_dropped_when_it_does_not_fit():
    out = chunk("x" * 30 + "\nTask groups\n" + "y" * 45, 50, 20)
    assert out == ["x" * 30 + "\nTask groups", "y" * 45]


def test_long_line_split_at_sentences_not_inside_them():
    line = "Task groups were added in 3.11. They wait for asyncio.run() tasks. " * 20
    out = chunk(line.strip(), 100, 0)
    assert all(len(c) <= 100 for c in out)
    assert all(c.endswith(".") for c in out)                 # every cut is at a sentence end
    assert "3.11" in out[0] and "asyncio.run()" in out[0]    # no cut after a dot inside a word


def test_long_sentence_split_at_spaces_with_spaces():
    sentence = " ".join(f"word{n}" for n in range(200))       # one sentence, ~1300 chars
    out = chunk(sentence, 100, 0)
    assert all(len(c) <= 100 for c in out)
    assert words(out) == sentence.split()                     # nothing lost, nothing glued
    assert all("\n" not in c for c in out)                    # words joined by spaces, not newlines


@pytest.mark.parametrize("word_len", [101, 250, 2000])
def test_one_huge_word_is_hard_cut(word_len):
    out = chunk("a" * word_len, 100, 0)
    assert all(0 < len(c) <= 100 for c in out)
    assert "".join(out) == "a" * word_len


def test_huge_word_inside_a_sentence():
    out = chunk("start " + "b" * 250 + " end", 100, 0)
    assert all(0 < len(c) <= 100 for c in out)
    assert out[0] == "start" and out[-1].endswith("end")


@pytest.mark.parametrize("seed", range(20))
def test_random_text_respects_size_and_keeps_order(seed):
    rnd = random.Random(seed)
    vocab = ["asyncio", "task", "group", "3.11.", "cancel!", "run()?", "x" * 150, "a.b.c", ""]
    text = "\n".join(" ".join(rnd.choice(vocab) for _ in range(rnd.randint(0, 80))) for _ in range(30))
    out = chunk(text, 200, 50)
    assert all(0 < len(c) <= 200 for c in out)
    # without the overlap, the chunks read back as the original words, in order
    no_overlap = chunk(text, 200, 0)
    assert "".join(words(no_overlap)) == "".join(text.split())


def test_overlap_bigger_than_size_still_terminates():
    out = chunk("\n".join(["abc"] * 50), 10, 1000)
    assert all(len(c) <= 10 for c in out)


def test_line_exactly_size():
    assert chunk("a" * 100, 100, 20) == ["a" * 100]
