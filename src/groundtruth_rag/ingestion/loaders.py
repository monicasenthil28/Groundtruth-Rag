"""Load supported documents without discarding PDF page boundaries."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class LoadedPage:
    document_id: str
    source: str
    page: int | None
    text: str


def discover_documents(input_path: Path) -> tuple[list[tuple[Path, str]], list[Path]]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"unsupported document type: {input_path.suffix}")
        return [(input_path, input_path.name)], []
    if not input_path.is_dir():
        raise ValueError(f"input does not exist or is not a file/directory: {input_path}")
    paths = sorted((path for path in input_path.rglob("*") if path.is_file()),
                   key=lambda path: path.relative_to(input_path).as_posix())
    supported = [(path, path.relative_to(input_path).as_posix()) for path in paths
                 if path.suffix.lower() in SUPPORTED_SUFFIXES]
    ignored = [path for path in paths if path.suffix.lower() not in SUPPORTED_SUFFIXES]
    if not supported:
        raise ValueError(f"no supported documents found in {input_path}")
    return supported, ignored


def load_document(path: Path, source: str) -> list[LoadedPage]:
    document_id = sha256(source.encode("utf-8")).hexdigest()[:24]
    if path.suffix.lower() == ".pdf":
        try:
            reader = PdfReader(path)
            if reader.is_encrypted:
                raise ValueError(f"encrypted PDF is not supported: {path}")
            return [LoadedPage(document_id, source, page_number, page.extract_text() or "")
                    for page_number, page in enumerate(reader.pages, 1)]
        except PdfReadError as exc:
            raise ValueError(f"cannot read PDF {path}: {exc}") from exc
    if path.suffix.lower() in {".txt", ".md"}:
        return [LoadedPage(document_id, source, None, path.read_text(encoding="utf-8-sig"))]
    raise ValueError(f"unsupported document type: {path.suffix}")
