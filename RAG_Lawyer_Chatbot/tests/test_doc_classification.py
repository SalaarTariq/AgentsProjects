"""doc_type is shown to the user, so it must not mislabel ordinary documents.

_classify_doc_type matched "v." as a bare substring, which also appears inside
Nov., Rev., Gov. and Univ. Because the case test runs ahead of statute and
memo, any document carrying a date or a common abbreviation was labelled a
court case — and that label is surfaced in the citation rail of both the
FastAPI UI and the Streamlit one.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vector_database import _classify_doc_type  # noqa: E402

MEMO = (
    "MEMORANDUM\nTo: General Counsel\nFrom: Associate\nRe: Lease renewal\n"
    "Dated Nov. 3, 2024. Please review before the board meeting."
)
STATUTE = (
    "Be it enacted by the Senate. Public Law 117-58. See Rev. Proc. 2021-45 "
    "for the applicable schedule under subsection (b)."
)
POLICY = (
    "This policy applies to all Univ. departments and takes effect "
    "at the start of the academic year."
)
CASE = (
    "Roe v. Wade, 410 U.S. 113. The plaintiff filed a complaint against "
    "the defendant seeking declaratory relief."
)
CONTRACT = (
    "THIS AGREEMENT is made this day, WITNESSETH, that the party of the "
    "first part, hereinafter the Seller, agrees as follows."
)


class TestAbbreviationsAreNotCaseCaptions:
    @pytest.mark.parametrize(
        "label,text,expected",
        [
            ("Nov.", MEMO, "memo"),
            ("Rev.", STATUTE, "statute"),
            ("Univ.", POLICY, "document"),
            ("Gov.", "MEMORANDUM\nTo: Staff\nFrom: Gov. Affairs\nRe: Deadline", "memo"),
        ],
    )
    def test_abbreviation_does_not_force_case(self, label, text, expected):
        got = _classify_doc_type(text)
        assert got == expected, f"{label} pushed the document to {got!r}"

    def test_therefore_is_not_a_memo_header(self):
        """Bare "re:" also matches inside "therefore:"."""
        text = "The parties have performed. Therefore: the balance is due on delivery."
        assert _classify_doc_type(text) == "document"


class TestRealDocumentsStillClassify:
    """Tightening the markers must not cost true positives."""

    def test_case_caption(self):
        assert _classify_doc_type(CASE) == "case"

    def test_case_caption_alone(self):
        assert _classify_doc_type("Brown v. Board of Education of Topeka.") == "case"

    def test_contract(self):
        assert _classify_doc_type(CONTRACT) == "contract"

    def test_statute(self):
        assert _classify_doc_type("Be it enacted. 42 U.S.C. 1983 applies.") == "statute"

    def test_memo(self):
        assert _classify_doc_type("MEMORANDUM\nTo: All\nRe: Policy") == "memo"

    def test_unknown_falls_through(self):
        assert _classify_doc_type("Lorem ipsum dolor sit amet.") == "document"
