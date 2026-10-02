"""Extracted citations are shown to the user, so they must be the citation only.

The case-caption branch grabbed a greedy run of capitalised words to the left of
" v. ". Bluebook signals and sentence openers are capitalised too, so they were
swallowed: "See Roe v. Wade", "Accord Smith v. Jones". That also broke
deduplication -- the same case reached the citation rail twice under different
prefixes. Meanwhile a party name containing a lowercase connector was cut short,
turning "Brown v. Board of Education" into "Brown v. Board".
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vector_database import _extract_citations  # noqa: E402


class TestSignalsAreNotPartOfTheCaption:
    @pytest.mark.parametrize(
        "text",
        [
            "See Roe v. Wade.",
            "Accord Roe v. Wade.",
            "Compare Roe v. Wade.",
            "But Roe v. Wade controls.",
            "Cf Roe v. Wade.",
            "Under Roe v. Wade the analysis differs.",
            "The Roe v. Wade decision.",
            "Citing Roe v. Wade.",
        ],
    )
    def test_signal_is_dropped(self, text):
        assert _extract_citations(text) == ["Roe v. Wade"], text

    def test_same_case_dedupes_across_signals(self):
        text = "Plaintiff relies on Roe v. Wade. See Roe v. Wade again."
        assert _extract_citations(text) == ["Roe v. Wade"]


class TestPartyNamesSurviveWhole:
    def test_connector_of_is_kept(self):
        got = _extract_citations("In Brown v. Board of Education the Court held.")
        assert got == ["Brown v. Board of Education"]

    def test_connector_and_is_kept(self):
        got = _extract_citations("Smith and Sons v. Jones.")
        assert got == ["Smith and Sons v. Jones"]

    def test_following_clause_is_not_absorbed(self):
        """"the" must not pull the next clause into the party name."""
        got = _extract_citations("Roe v. Wade the Court reversed.")
        assert got == ["Roe v. Wade"]


class TestOtherCitationFormsUnaffected:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Under 42 U.S.C. § 1983 relief is available.", "42 U.S.C. § 1983"),
            ("Reported at 410 U.S. 113 today.", "410 U.S. 113"),
            ("See 113 S. Ct. 2786 for the holding.", "113 S. Ct. 2786"),
            ("Article III vests the judicial power.", "Article III"),
        ],
    )
    def test_form_still_extracted(self, text, expected):
        assert expected in _extract_citations(text)

    def test_mixed_text(self):
        got = _extract_citations(
            "Under 42 U.S.C. § 1983 and Article III, see Terry v. Ohio."
        )
        assert got == sorted(["42 U.S.C. § 1983", "Article III", "Terry v. Ohio"])


class TestNoFalsePositives:
    @pytest.mark.parametrize(
        "text",
        [
            "The parties met in November and agreed to the terms.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "The Court held that the statute was valid.",
        ],
    )
    def test_ordinary_prose_yields_nothing(self, text):
        assert _extract_citations(text) == []

    def test_cap_is_respected(self):
        from vector_database import MAX_CITATIONS_PER_CHUNK

        # Letters only: the party pattern is [A-Z][a-z]+, so names carrying a
        # digit ("Party1") never matched -- before this change or after.
        names = ["Able", "Baker", "Cain", "Dover", "Eaton",
                 "Finch", "Grant", "Hobbs", "Ivers", "Jonas", "Kemp", "Lowe"]
        text = " ".join(f"{n} v. Smith." for n in names)
        assert len(_extract_citations(text)) == MAX_CITATIONS_PER_CHUNK
