# arXiv source bundle

Self-contained LaTeX source for the camera-ready paper (CEURART one-column
class vendored as `ceurart.cls`). Build with `pdflatex main.tex` (twice) or
`tectonic main.tex`. Regenerate this directory from `paper/latex/` after any
manuscript change:

    cp latex/main.tex latex/ceurart.cls latex/flowchart.pdf latex/transfer_contrast.pdf arxiv/

Before uploading to arXiv: (1) add the CEUR-WS volume/URL to the title
footnote once the proceedings appear; (2) upload as a .tar.gz of this
directory (`tar czf evorank_arxiv.tar.gz -C arxiv .`); (3) license: CC BY 4.0
(matches the CEUR copyright clause in `main.tex`).
