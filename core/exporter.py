from pathlib import Path
from textwrap import wrap


class Exporter:
    def __init__(self, app_dir: Path) -> None:
        self.exports_dir = app_dir / "exports"
        self.exports_dir.mkdir(exist_ok=True)

    def export_markdown(self, session_id: str, title: str, content: str) -> Path:
        path = self.exports_dir / f"{session_id}_{title.lower().replace(' ', '_')}.md"
        path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
        return path

    def export_pdf(self, session_id: str, title: str, content: str) -> Path:
        path = self.exports_dir / f"{session_id}_{title.lower().replace(' ', '_')}.pdf"
        lines = [title, ""] + self._wrap_text(content)
        self._write_simple_pdf(path, lines)
        return path

    def _wrap_text(self, content: str) -> list[str]:
        lines: list[str] = []
        for paragraph in content.splitlines():
            if not paragraph.strip():
                lines.append("")
                continue
            lines.extend(wrap(paragraph, width=92) or [""])
        return lines

    def _write_simple_pdf(self, path: Path, lines: list[str]) -> None:
        escaped_lines = [self._escape_pdf_text(line) for line in lines[:55]]
        text_ops = ["BT", "/F1 10 Tf", "50 790 Td", "14 TL"]
        for line in escaped_lines:
            text_ops.append(f"({line}) Tj")
            text_ops.append("T*")
        text_ops.append("ET")
        stream = "\n".join(text_ops).encode("latin-1", errors="replace")

        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        ]

        chunks = [b"%PDF-1.4\n"]
        offsets = [0]
        for index, obj in enumerate(objects, start=1):
            offsets.append(sum(len(chunk) for chunk in chunks))
            chunks.append(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
        xref_offset = sum(len(chunk) for chunk in chunks)
        chunks.append(f"xref\n0 {len(objects) + 1}\n".encode())
        chunks.append(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            chunks.append(f"{offset:010d} 00000 n \n".encode())
        chunks.append(
            f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
        )
        path.write_bytes(b"".join(chunks))

    def _escape_pdf_text(self, text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
