"""LaTeX compilation utilities with centralized dependency management."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from crystalfig.exceptions import LatexCompilationError
from crystalfig.renderers.tikz_renderer import TikzRenderer


@dataclass
class CompilationResult:
    """Result of a LaTeX compilation."""

    success: bool
    output_path: str | None
    engine: str
    return_code: int
    stdout: str
    stderr: str
    log_path: str | None
    warnings: list[str]


class LatexCompiler:
    """Compile LaTeX snippets or standalone documents to PDF."""

    ENGINES = ["pdflatex", "xelatex", "lualatex"]

    def __init__(self, engine: str = "pdflatex", runs: int = 1, keep_log_on_failure: bool = True):
        if engine not in self.ENGINES:
            raise ValueError(f"Unknown LaTeX engine '{engine}'. Use one of {self.ENGINES}")
        self.engine = engine
        self.runs = runs
        self.keep_log_on_failure = keep_log_on_failure

    @classmethod
    def detect_engine(cls) -> str | None:
        """Return the first available LaTeX engine or None."""
        for engine in cls.ENGINES:
            if shutil.which(engine):
                return engine
        return None

    def compile(
        self,
        source: str,
        output_path: str,
        source_name: str = "figure.tex",
        build_dir: str | None = None,
    ) -> CompilationResult:
        """Compile LaTeX source to PDF."""
        build_path = Path(build_dir) if build_dir else Path(tempfile.mkdtemp(prefix="crystalfig_latex_"))
        build_path.mkdir(parents=True, exist_ok=True)
        tex_file = build_path / source_name
        tex_file.write_text(source, encoding="utf-8")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stdout_acc = []
        stderr_acc = []
        warnings = []
        return_code = -1

        for _ in range(self.runs):
            result = subprocess.run(
                [self.engine, "-interaction=nonstopmode", "-halt-on-error", str(tex_file.name)],
                cwd=str(build_path),
                capture_output=True,
                text=True,
                shell=False,
            )
            stdout_acc.append(result.stdout)
            stderr_acc.append(result.stderr)
            return_code = result.returncode
            if return_code != 0:
                break

        log_file = build_path / (source_name.replace(".tex", ".log"))
        log_path = str(log_file) if log_file.exists() else None

        # Extract warnings from log
        if log_path:
            log_text = log_file.read_text(errors="ignore")
            warnings = [line.strip() for line in log_text.splitlines() if "Warning" in line]

        # Move PDF to requested output path
        pdf_file = build_path / (source_name.replace(".tex", ".pdf"))
        success = return_code == 0 and pdf_file.exists()
        if success:
            shutil.copy(str(pdf_file), str(output_path))
            final_path = str(output_path)
        else:
            final_path = None
            if self.keep_log_on_failure and log_path:
                shutil.copy(str(log_file), str(output_path.with_suffix(".log")))

        if not self.keep_log_on_failure or success:
            shutil.rmtree(str(build_path), ignore_errors=True)

        if not success:
            raise LatexCompilationError(
                f"LaTeX compilation failed with return code {return_code}.\n"
                f"stdout: {''.join(stdout_acc)[-2000:]}\n"
                f"stderr: {''.join(stderr_acc)[-2000:]}"
            )

        return CompilationResult(
            success=success,
            output_path=final_path,
            engine=self.engine,
            return_code=return_code,
            stdout="".join(stdout_acc),
            stderr="".join(stderr_acc),
            log_path=log_path,
            warnings=warnings,
        )


def compile_tikz_to_pdf(tikz_code: str, output_pdf: str = "figure.pdf", engine: str = "pdflatex") -> CompilationResult:
    """Compile a TikZ snippet or standalone document to PDF."""
    compiler = LatexCompiler(engine=engine)
    if r"\documentclass" not in tikz_code:
        # Wrap snippet in standalone
        libs = ",".join(TikzRenderer.REQUIRED_LIBRARIES) if "TikzRenderer" in globals() else "arrows.meta,calc,positioning,shadings,backgrounds,fit,shapes.geometric"
        source = (
            r"\documentclass[tikz,border=10pt]{standalone}"
            f"\n\\usepackage{{tikz}}\n\\usepackage{{amsmath,amssymb}}\n\\usepackage{{xcolor}}\n"
            f"\\usetikzlibrary{{{libs}}}\n\\begin{{document}}\n{tikz_code}\n\\end{{document}}"
        )
    else:
        source = tikz_code
    return compiler.compile(source, output_pdf)
