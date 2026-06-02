MAIN = main
LATEX = pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: pdf clean watch

pdf:
	$(LATEX) $(MAIN).tex
	bibtex $(MAIN) || true
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).fls $(MAIN).fdb_latexmk

watch:
	@command -v latexmk >/dev/null 2>&1 || { echo "latexmk required for watch mode" >&2; exit 1; }
	latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex
