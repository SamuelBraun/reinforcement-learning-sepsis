# Report

Source for the RL Sepsis project report. Targets max 15 pages excluding references.

## Structure

```
report/
  main.tex                       # top-level document
  sections/
    01_introduction.tex
    02_methodology.tex
    03_evaluation.tex
    04_creative_extension.tex
    05_conclusion.tex
  references.bib                 # bibliography (BibTeX)
  figures/                       # PNGs copied from ../plots/
  Makefile                       # local build
```

## Build (local, requires LaTeX toolchain)

If BasicTeX / MacTeX / TeX Live is installed on `$PATH`:

```bash
cd report
make pdf            # produces main.pdf
make clean          # removes auxiliary files
```

If you do not have LaTeX installed, get BasicTeX:

```bash
brew install --cask basictex     # requires sudo, ~100MB
eval "$(/usr/libexec/path_helper)"  # refresh PATH
sudo tlmgr update --self
sudo tlmgr install booktabs cleveref enumitem natbib
```

## Build (Overleaf, no local install)

1. Zip the entire `report/` directory.
2. Sign in at <https://www.overleaf.com>.
3. New Project → Upload Project → drop the zip.
4. Set the compiler to `pdfLaTeX` and the main document to `main.tex`.
5. Click **Recompile**.

## Before submission checklist

- Replace the four `[Author N -- fill in before submission]` placeholders in `main.tex` with the actual group member names and student IDs.
- Replace `[Group number]` with the assigned group number.
- Update the date if June 2026 is wrong.
- Verify the PDF is under 15 pages (excluding bibliography).
- Confirm all figures render correctly (they should, since they are PNGs copied from the notebook's `plots/` directory).
