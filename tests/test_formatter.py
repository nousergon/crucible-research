"""
Tests for emailer/formatter.py.
All functions are pure — no mocking needed.
"""

from emailer.formatter import _inline_md, _md_to_html, format_email

# ── _inline_md ────────────────────────────────────────────────────────────────

class TestInlineMd:
    def test_bold(self):
        assert _inline_md("**hello**") == "<strong>hello</strong>"

    def test_italic(self):
        assert _inline_md("*hello*") == "<em>hello</em>"

    def test_bold_italic(self):
        result = _inline_md("***hello***")
        assert "<strong>" in result and "<em>" in result

    def test_code(self):
        assert _inline_md("`code`") == "<code>code</code>"

    def test_buy_colored(self):
        result = _inline_md("Rating: BUY")
        assert '<span class="buy">BUY</span>' in result

    def test_sell_colored(self):
        result = _inline_md("Rating: SELL")
        assert '<span class="sell">SELL</span>' in result

    def test_hold_colored(self):
        result = _inline_md("Rating: HOLD")
        assert '<span class="hold">HOLD</span>' in result

    def test_stale_colored(self):
        result = _inline_md("⚠stale")
        assert '<span class="stale">⚠stale</span>' in result

    def test_no_change_plain_text(self):
        assert _inline_md("plain text") == "plain text"

    def test_buy_word_boundary(self):
        # "BUYOUT" should NOT be colored
        result = _inline_md("BUYOUT is happening")
        assert '<span class="buy">' not in result


# ── _md_to_html ───────────────────────────────────────────────────────────────

class TestMdToHtml:
    def test_h1_becomes_h2(self):
        result = _md_to_html("# Title")
        assert "<h2>" in result and "Title" in result

    def test_h2_becomes_h3(self):
        result = _md_to_html("## Section")
        assert "<h3>" in result and "Section" in result

    def test_h3_becomes_h4(self):
        result = _md_to_html("### Subsection")
        assert "<h4>" in result

    def test_bullet_dash(self):
        result = _md_to_html("- item one")
        assert "<ul>" in result
        assert "<li>" in result
        assert "item one" in result

    def test_bullet_star(self):
        result = _md_to_html("* item")
        assert "<li>" in result

    def test_bullet_dot(self):
        result = _md_to_html("• item")
        assert "<li>" in result

    def test_horizontal_rule_dashes(self):
        result = _md_to_html("---")
        assert "<hr>" in result

    def test_horizontal_rule_equals(self):
        result = _md_to_html("===")
        assert "<hr>" in result

    def test_empty_line_becomes_br(self):
        result = _md_to_html("")
        assert "<br>" in result

    def test_plain_paragraph(self):
        result = _md_to_html("Some plain text here.")
        assert "<p>" in result and "Some plain text here." in result

    def test_table_parsed(self):
        md = "| Col1 | Col2 |\n|---|---|\n| A | B |"
        result = _md_to_html(md)
        assert "<table>" in result
        assert "A" in result and "B" in result

    def test_list_closed_before_header(self):
        md = "- item\n# Header"
        result = _md_to_html(md)
        assert "</ul>" in result

    def test_list_closed_at_end(self):
        md = "- item one\n- item two"
        result = _md_to_html(md)
        assert result.count("</ul>") == 1

    def test_table_closed_at_end(self):
        md = "| A | B |\n| C | D |"
        result = _md_to_html(md)
        assert "</table>" in result

    def test_table_closed_before_hr(self):
        md = "| A | B |\n---"
        result = _md_to_html(md)
        assert "</table>" in result or "</tbody>" in result
        assert "<hr>" in result

    def test_inline_bold_in_paragraph(self):
        result = _md_to_html("**strong** word")
        assert "<strong>" in result


# ── format_email ──────────────────────────────────────────────────────────────

class TestFormatEmail:
    def test_returns_tuple(self):
        html, plain = format_email("# Report\n\nSome text.", "2026-03-05")
        assert isinstance(html, str)
        assert isinstance(plain, str)

    def test_html_contains_doctype(self):
        html, _ = format_email("Hello", "2026-03-05")
        assert "<!DOCTYPE html>" in html

    def test_html_contains_timestamp(self):
        html, _ = format_email("Hello", "2026-03-05")
        assert "2026-03-05" in html

    def test_plain_strips_headers(self):
        _, plain = format_email("# My Header\n\nBody text.", "2026-03-05")
        assert "#" not in plain
        assert "My Header" in plain

    def test_plain_strips_bold(self):
        _, plain = format_email("**bold text**", "2026-03-05")
        assert "**" not in plain
        assert "bold text" in plain

    def test_plain_strips_italic(self):
        _, plain = format_email("*italic*", "2026-03-05")
        assert "*italic*" not in plain
        assert "italic" in plain

    def test_plain_strips_code(self):
        _, plain = format_email("`some code`", "2026-03-05")
        assert "`" not in plain
        assert "some code" in plain

    def test_plain_has_footer(self):
        _, plain = format_email("text", "2026-03-05")
        assert "alpha-engine-research" in plain
        assert "2026-03-05" in plain

    def test_html_body_inner_content(self):
        html, _ = format_email("# Hello\n\nWorld.", "2026-03-05")
        assert "Hello" in html
        assert "World" in html

    def test_empty_report(self):
        html, plain = format_email("", "2026-03-05")
        assert isinstance(html, str)
        assert isinstance(plain, str)
