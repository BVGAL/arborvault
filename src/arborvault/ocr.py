"""Local PDF rasterization and Tesseract, called only from an isolated parser worker."""

import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


class PDFOCR:
    def __init__(self, data, limits, languages="eng+ita"):
        import pypdfium2 as pdfium

        if shutil.which("tesseract") is None:
            raise ValueError("OCR needs Tesseract and the ocr package extra")
        if not re.fullmatch(r"[a-z]{3}(\+[a-z]{3})*", languages):
            raise ValueError("Invalid OCR languages")
        self.document = pdfium.PdfDocument(data)
        self.limits, self.languages = limits, languages

    def page_text(self, number):
        page = self.document[number]
        bitmap = None
        try:
            width, height = page.get_size()
            if width <= 0 or height <= 0:
                raise ValueError("Invalid PDF page dimensions")
            scale = min(3.0, math.sqrt(self.limits.ocr_max_pixels / (width * height)))
            if scale < 0.1:
                raise ValueError("PDF page exceeds the OCR raster budget")
            bitmap = page.render(scale=scale)
            with tempfile.TemporaryDirectory(prefix="ocr-page-") as temporary:
                source = Path(temporary) / "page.png"
                with bitmap.to_pil() as image:
                    image.save(source)
                result = subprocess.run(
                    ["tesseract", str(source), "stdout", "-l", self.languages, "--psm", "3"],
                    capture_output=True,
                    timeout=self.limits.ocr_page_seconds,
                    check=False,
                )
                if result.returncode:
                    raise ValueError("OCR failed; check installed language data")
                if len(result.stdout) > self.limits.max_text * 4:
                    raise ValueError("OCR output exceeds text budget")
                return result.stdout.decode("utf-8")
        finally:
            if bitmap:
                bitmap.close()
            page.close()

    def close(self):
        self.document.close()
