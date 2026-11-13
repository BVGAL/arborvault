"""Deployment-controlled parsing budgets; request data cannot raise these limits."""

import os
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class ParsingLimits:
    max_bytes: int = 25_000_000
    max_text: int = 20_000_000
    max_nodes: int = 50_000
    max_pdf_pages: int = 2000
    wall_seconds: int = 300
    cpu_seconds: int = 300
    memory_mb: int = 1024
    ocr_page_seconds: int = 30
    ocr_max_pixels: int = 12_000_000

    @classmethod
    def from_env(cls):
        values = {
            field.name: int(os.getenv("PARSER_" + field.name.upper(), field.default))
            for field in fields(cls)
        }
        if any(value < 1 for value in values.values()):
            raise ValueError("Parser limits must be positive integers")
        return cls(**values)

    def environment(self):
        return {
            "PARSER_" + field.name.upper(): str(getattr(self, field.name)) for field in fields(self)
        }

    @property
    def output_bytes(self):
        return self.max_text * 6 + self.max_nodes * 2000
