from app.chunking import chunk_text


def test_short_paragraphs_stay_in_one_chunk():
    text = "First paragraph with some content.\n\nSecond paragraph with more content."
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=0)

    assert len(chunks) == 1
    assert "First paragraph" in chunks[0].text
    assert "Second paragraph" in chunks[0].text


def test_respects_heading_structure():
    text = (
        "# Title\n\n"
        "Intro paragraph.\n\n"
        "## Section A\n\n"
        "Content of section A.\n\n"
        "## Section B\n\n"
        "Content of section B."
    )
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=0)

    headings = [c.heading_path for c in chunks]
    assert ["Title"] in headings
    assert ["Title", "Section A"] in headings
    assert ["Title", "Section B"] in headings


def test_never_splits_a_paragraph_that_fits_in_the_budget():
    paragraph_a = "Alpha content that stays intact and is not split mid-sentence."
    paragraph_b = "Beta content that also stays intact across the chunk boundary."
    text = f"{paragraph_a}\n\n{paragraph_b}"

    chunks = chunk_text(text, chunk_size=len(paragraph_a) + 5, chunk_overlap=0)

    assert len(chunks) == 2
    assert chunks[0].text == paragraph_a
    assert paragraph_b in chunks[1].text


def test_hard_splits_a_single_oversized_paragraph_on_sentence_boundaries():
    paragraph = "Sentence one is here. Sentence two is here. Sentence three is here."
    chunks = chunk_text(paragraph, chunk_size=30, chunk_overlap=0)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 30 or " " not in c.text.strip(".")


def test_overlap_carries_tail_into_next_chunk():
    paragraph_a = "Alpha content here that is reasonably long for testing overlap behavior."
    paragraph_b = "Beta content here that is reasonably long for testing overlap behavior."
    text = f"{paragraph_a}\n\n{paragraph_b}"

    chunks = chunk_text(text, chunk_size=len(paragraph_a) + 5, chunk_overlap=20)

    assert len(chunks) == 2
    assert paragraph_a[-20:] in chunks[1].text
