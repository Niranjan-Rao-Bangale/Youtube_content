# pdf_server.py  ▸  run with:  fastmcp run pdf_server.py  (or spawn via MCPServerStdio)

from fastmcp import FastMCP
from pathlib import Path
import pypdf  # pip install pypdf

mcp = FastMCP("PDF Explorer")

PDF_ROOT = Path(r"<Your_PDF Folder>")   # limit access to this folder for safety


def _open(path: str) -> pypdf.PdfReader:
    full = (PDF_ROOT / Path(path)).resolve()
    if not full.is_file() or not full.is_relative_to(PDF_ROOT):
        raise FileNotFoundError("PDF not found or outside allowed folder.")
    return pypdf.PdfReader(str(full))


@mcp.tool()
def list_pdfs() -> list[str]:
    """Return all PDF filenames in the shared folder."""
    return sorted(p.name for p in PDF_ROOT.glob("*.pdf"))


@mcp.tool()
def extract_text(file: str, max_chars: int = 20_000) -> str:
    """Extract raw text (first `max_chars` chars) from a PDF file."""
    reader = _open(file)
    out = []
    for page in reader.pages:
        out.append(page.extract_text() or "")
        if sum(map(len, out)) >= max_chars:
            break
    return "".join(out)[:max_chars]


@mcp.tool()
def summarise_pdf(file: str, hint: str = "") -> str:
    """
    Ask the MODEL to summarise the PDF.
    `hint` lets the user specify what to focus on.
    """
    text = extract_text(file, max_chars=30_000)
    return mcp.models.current_model.chat(
        f"Summarise this PDF. Focus on: {hint}\n\n{text}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")        # so an Agent can spawn it
