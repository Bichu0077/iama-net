"""Convert Markdown research report to publication-styled PDF using Chrome Headless.

Embeds all images as base64 data URLs to avoid headless browser local file blocking.
Applies clean academic styling, responsive tables, callout blocks, and page layout.
"""

import base64
import os
import re
import subprocess
from pathlib import Path
import markdown

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def embed_images_as_base64(md_content: str, base_dir: Path) -> str:
    """Find all image paths in markdown and replace them with base64 data URIs."""
    def replacer(match):
        alt = match.group(1)
        src = match.group(2).strip()

        # Handle file:/// or relative or absolute Windows paths
        clean_src = src.replace("file:///", "").replace("file://", "")
        # Remove leading slash on windows if format is /C:/...
        if len(clean_src) > 2 and clean_src[0] == "/" and clean_src[2] == ":":
            clean_src = clean_src[1:]

        path = Path(clean_src)
        if not path.is_absolute():
            path = (base_dir / clean_src).resolve()

        if path.exists() and path.is_file():
            suffix = path.suffix.lower()
            mime = "image/png" if suffix in (".png",) else "image/jpeg"
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                return f"![{alt}](data:{mime};base64,{b64})"
            except Exception as e:
                print(f"Warning: Could not embed {path}: {e}")
        else:
            print(f"Warning: Image file not found: {path}")
        return match.group(0)

    # Pattern for ![alt](path)
    pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
    return re.sub(pattern, replacer, md_content)


def generate_styled_html(body_html: str, title: str = "IAMA-Net Research Report") -> str:
    """Wrap body HTML with professional academic print styling."""
    css = """
    @page {
        size: A4;
        margin: 20mm 15mm 20mm 15mm;
        @bottom-right {
            content: counter(page);
            font-size: 9pt;
            color: #666;
        }
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 10pt;
        line-height: 1.55;
        color: #222;
        background-color: #fff;
        margin: 0;
        padding: 0;
    }
    h1 {
        font-size: 20pt;
        font-weight: 700;
        color: #1a2a44;
        border-bottom: 2.5px solid #2b5797;
        padding-bottom: 6px;
        margin-top: 10px;
        margin-bottom: 8px;
    }
    h2 {
        font-size: 14pt;
        font-weight: 600;
        color: #2b5797;
        border-bottom: 1px solid #dcdcdc;
        padding-bottom: 4px;
        margin-top: 22px;
        margin-bottom: 10px;
        page-break-after: avoid;
    }
    h3 {
        font-size: 11.5pt;
        font-weight: 600;
        color: #333;
        margin-top: 16px;
        margin-bottom: 6px;
        page-break-after: avoid;
    }
    h4 {
        font-size: 10pt;
        font-weight: 600;
        color: #444;
        margin-top: 12px;
        margin-bottom: 4px;
    }
    p {
        margin: 0 0 10px 0;
        text-align: justify;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0;
        font-size: 8.8pt;
        page-break-inside: avoid;
    }
    th, td {
        padding: 6px 8px;
        border: 1px solid #d8d8d8;
        text-align: left;
    }
    th {
        background-color: #f2f5f9;
        font-weight: 600;
        color: #1a2a44;
        border-bottom: 2px solid #b4c6e7;
    }
    tr:nth-child(even) {
        background-color: #fafbfc;
    }
    blockquote {
        margin: 14px 0;
        padding: 10px 14px;
        background-color: #f4f7fb;
        border-left: 4px solid #2b5797;
        border-radius: 0 4px 4px 0;
        font-size: 9.5pt;
        color: #2c3e50;
    }
    blockquote p {
        margin: 0;
    }
    code {
        font-family: Consolas, "Courier New", monospace;
        font-size: 8.5pt;
        background-color: #f5f6f8;
        padding: 2px 4px;
        border-radius: 3px;
        border: 1px solid #e1e4e8;
    }
    pre {
        background-color: #f8f9fa;
        border: 1px solid #e1e4e8;
        border-radius: 4px;
        padding: 10px;
        overflow-x: auto;
        font-family: Consolas, "Courier New", monospace;
        font-size: 8pt;
        line-height: 1.4;
        page-break-inside: avoid;
    }
    pre code {
        background: none;
        padding: 0;
        border: none;
    }
    img {
        max-width: 100%;
        height: auto;
        border-radius: 4px;
        border: 1px solid #e1e4e8;
        margin: 6px auto;
        display: block;
    }
    em {
        font-size: 8.5pt;
        color: #555;
        text-align: center;
        display: block;
        margin-top: 2px;
        margin-bottom: 12px;
    }
    hr {
        border: none;
        border-top: 1px solid #e1e4e8;
        margin: 18px 0;
    }
    .badge {
        display: inline-block;
        padding: 2px 6px;
        font-size: 8pt;
        font-weight: 600;
        border-radius: 3px;
        background: #2b5797;
        color: #fff;
    }
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>{css}</style>
</head>
<body>
{body_html}
</body>
</html>
"""


def convert_md_to_pdf(md_path: Path, output_pdf_path: Path) -> Path:
    """Convert Markdown file to high-quality PDF using Chrome headless."""
    print(f"Reading markdown from: {md_path}")
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    print("Embedding images as base64...")
    md_text_embedded = embed_images_as_base64(md_text, md_path.parent)

    print("Converting Markdown to HTML...")
    html_body = markdown.markdown(
        md_text_embedded,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
    )

    full_html = generate_styled_html(html_body, title="IAMA-Net Research Report")

    temp_html_path = output_pdf_path.parent / "temp_report.html"
    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Saved styled HTML to: {temp_html_path}")

    # Chrome executable location
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not os.path.exists(chrome_path):
        chrome_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

    print(f"Rendering PDF with browser: {chrome_path}")
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={output_pdf_path.resolve()}",
        str(temp_html_path.resolve()),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error rendering PDF: {res.stderr}")
    else:
        print(f"PDF rendered successfully to: {output_pdf_path}")

    # Clean up temp html
    if temp_html_path.exists():
        temp_html_path.unlink()

    return output_pdf_path


if __name__ == "__main__":
    import sys
    src_md = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "results" / "project_progress_report.md"
    dst_pdf = Path(sys.argv[2]) if len(sys.argv) > 2 else PROJECT_ROOT / "results" / "IAMA_Net_Project_Report.pdf"
    convert_md_to_pdf(src_md, dst_pdf)
