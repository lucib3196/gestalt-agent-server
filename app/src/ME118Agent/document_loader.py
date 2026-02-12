from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pathlib import Path
from typing import List
from langchain_core.documents import Document
import json


class LectureDocumentLoader(BaseLoader):
    def __init__(
        self,
        root: str | Path,
        recursive: bool = True,
        metadata: dict[str, str] | None = None,
    ):
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise ValueError(f"[Document Loader] Failed to resolve path {self.root}.")

        self.recursive = recursive
        self.base_metadata = metadata or {}
        self.cwd = Path().resolve()

    def _relpath(self, path: Path | None) -> str | None:
        """Return path relative to CWD as POSIX string, or None."""
        if path is None:
            return None
        return path.resolve().relative_to(self.cwd).as_posix()

    def load(self) -> List[Document]:
        docs = []
        for lecture_dir in self.root.iterdir():
            if not lecture_dir.is_dir():
                continue

            # -------------------------
            # Markdown content
            # -------------------------
            content = ""
            md_path = next(lecture_dir.glob("*.md"), None)
            if md_path:
                content = md_path.read_text(encoding="utf-8")

            # -------------------------
            # Source PDF
            # -------------------------
            pdf_path = next(lecture_dir.glob("*.pdf"), None)

            # -------------------------
            # Lecture metadata
            # -------------------------
            lecture_metadata: dict = {}
            metadata_path = lecture_dir / "output.json"
            if metadata_path.exists():
                raw = json.loads(metadata_path.read_text(encoding="utf-8"))
                lecture_metadata = raw.get("lecture_analysis", {})
            # -------------------------
            # Yield document
            # -------------------------
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        **self.base_metadata,
                        **lecture_metadata,
                        "source_pdf": self._relpath(pdf_path),
                        "source_markdown": self._relpath(md_path),
                    },
                )
            )
        return docs


if __name__ == "__main__":
    print("Running")
    loader = LectureDocumentLoader(
        root=r"assets/ME118Lecture",
        metadata={
            "course": "ME118 Mechanical Engineering Modeling and Analysis ",
            "professor": "Sundar",
        },
    )
    print(loader.load_and_split()[0])