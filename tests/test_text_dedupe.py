from core.text_dedupe import TextDeduper, clean_text, is_useful_text


def test_text_dedupe_skips_highly_similar_text():
    deduper = TextDeduper(threshold=0.90)

    assert deduper.should_store("FastAPI tutorial with middleware and ORM examples")
    assert not deduper.should_store("FastAPI tutorial with middleware and ORM example")


def test_text_dedupe_allows_different_text():
    deduper = TextDeduper(threshold=0.90)

    assert deduper.should_store("FastAPI tutorial with middleware and ORM examples")
    assert deduper.should_store("A separate screen about Notion planning and notes")


def test_clean_and_useful_text_preserves_unicode():
    text = clean_text(" hello \n\n 世界 \t FastAPI ")

    assert text == "hello 世界 FastAPI"
    assert is_useful_text(text, min_length=5)
