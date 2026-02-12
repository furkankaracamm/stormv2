from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class FigureExtractionResult:
    page: int
    figure_id: str
    caption: str | None = None


class DeepFiguresExtractor:
    def is_available(self) -> bool:
        return False

    def extract_figures_from_pdf(self, pdf_path: str) -> List[FigureExtractionResult]:
        if not self.is_available():
            # [GRACEFUL FAIL] Log warning but do not crash the pipeline
            print("[FigureExtractor] deepfigures-open not installed. Skipping.")
            return []
        raise RuntimeError("deepfigures-open integration is not wired yet")

    def dry_run(self) -> List[FigureExtractionResult]:
        return [FigureExtractionResult(page=2, figure_id="fig-2")]
