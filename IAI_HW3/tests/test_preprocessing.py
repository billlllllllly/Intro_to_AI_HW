import pytest

from sentiment_preprocessing import preprocess


@pytest.mark.parametrize(("tweet", "expected"), [
    ("@united #AspenBaggageFail pls fix", "Aspen Baggage Fail"),
    ("flight cancelled :(", "sad face"),
    ("I love it <3", "heart"),
    ("I love it &lt;3", "heart"),
])
def test_clean_preprocessing_keeps_sentiment_cues(tweet, expected):
    assert expected in preprocess(tweet, clean=True)


@pytest.mark.parametrize(("tweet", "expected"), [
    ("soooo late", "soo late"),
    ("1000 miles", "1000 miles"),
    ("http://x.co/a:/", ""),
])
def test_clean_preprocessing_normalizes_text(tweet, expected):
    assert preprocess(tweet, clean=True) == expected


def test_elongation_is_preserved_without_cleaning():
    assert preprocess("soooo late") == "soooo late"
