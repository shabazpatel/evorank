# arXiv source bundle

Self-contained LaTeX source of the camera-ready paper (CEURART one-column class
vendored as `ceurart.cls`; bibliography is inline, no .bib). The first line of
`main.tex` sets `\pdfoutput=1` under pdfLaTeX so arXiv picks the pdfLaTeX path
(the figures are PDF). Locally it also builds with `tectonic main.tex`.

Regenerate this directory after any manuscript change (keep the first line):

    cp latex/ceurart.cls latex/flowchart.pdf latex/transfer_contrast.pdf arxiv/
    (sed '1i\' ... ) or re-apply the \pdfoutput line to a fresh copy of latex/main.tex

Build the upload archive:

    tar czf evorank_arxiv.tar.gz -C arxiv main.tex ceurart.cls flowchart.pdf transfer_contrast.pdf

Submission metadata is in `ARXIV_METADATA.md`. Before uploading, do one
pdfLaTeX test build (arXiv uses pdfLaTeX, TeX Live): upload the archive to an
Overleaf project with the compiler set to pdfLaTeX, or install BasicTeX
(`brew install --cask basictex`) and run `pdflatex main.tex` twice.

After the CEUR-WS volume is published, add the volume URL to the title footnote
in `latex/main.tex`, rebuild, regenerate this bundle, and update the arXiv
record's journal reference (Comments/Journal-ref fields), not a new version of
the PDF unless the text changes.
