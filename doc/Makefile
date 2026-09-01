ELEGANT ?= elegant

all:: elegant.ps elegant.pdf overview.ps overview.pdf Pelegant.ps Pelegant.pdf elegantOverview.pdf html

elegant.ps:
	rm -f elegant.aux elegant.dvi elegant.log elegant.toc dictionary.tex
	$(ELEGANT) print_dict.ele
	examples/collectDescriptions 
	latex elegant
	bibtex elegant
	latex elegant
	latex elegant
	dvips elegant -o elegant.ps
	rm -f dictionary.tex elegant.dvi elegant.log dictionary.aux elegant.aux elegant.out dictionary.sdds elegant.bbl elegant.blg

elegant.pdf: elegant.ps
	ps2pdf13 elegant.ps elegant.pdf

overview.ps: overview.tex
	rm -f overview.aux overview.dvi overview.log overview.toc 
	latex overview
	latex overview
	dvips overview -o overview.ps
	rm -f overview.dvi overview.log overview.aux overview.out

overview.pdf: overview.ps
	ps2pdf13 overview.ps overview.pdf

Pelegant.ps: Pelegant.tex
	rm -f Pelegant.aux Pelegant.dvi Pelegant.log Pelegant.toc 
	latex Pelegant
	latex Pelegant
	dvips Pelegant -o Pelegant.ps
	rm -f Pelegant.dvi Pelegant.log Pelegant.aux Pelegant.out

Pelegant.pdf: Pelegant.ps
	ps2pdf13 Pelegant.ps Pelegant.pdf

elegantOverview.pdf: elegantOverview.tex references.bib
	rm -f elegantOverview.aux elegantOverview.bbl elegantOverview.blg elegantOverview.log elegantOverview.out elegantOverview.toc
	pdflatex elegantOverview
	bibtex elegantOverview
	pdflatex elegantOverview
	pdflatex elegantOverview
	rm -f elegantOverview.aux elegantOverview.bbl elegantOverview.blg elegantOverview.log elegantOverview.out elegantOverview.toc

html:
	mkdir -p elegant
	$(ELEGANT) print_dict.ele
	htlatex elegant "html,3" "" "-delegant/"
	bibtex elegant
	htlatex elegant "html,3" "" "-delegant/"
	rm -f *.html *.png *.css *.4ct *.tmp *.dvi *.idv *.4tc *.lg *.xref *.log *.aux dictionary.tex elegant.bbl elegant.blg
	mkdir -p Pelegant
	htlatex Pelegant "html,3" "" "-dPelegant/"
	rm -f *.html *.png *.css *.4ct *.tmp *.dvi *.idv *.4tc *.lg *.xref *.log *.aux


clean::
	rm -f elegant.aux elegant.dvi elegant.log elegant.toc elegant.ps elegant.pdf dictionary.tex
	rm -rf elegant
	rm -f overview.aux overview.dvi overview.log overview.toc overview.ps overview.pdf
	rm -rf overview
	rm -f Pelegant.aux Pelegant.dvi Pelegant.log Pelegant.toc Pelegant.ps Pelegant.pdf
	rm -f elegantOverview.aux elegantOverview.bbl elegantOverview.blg elegantOverview.log elegantOverview.out elegantOverview.toc elegantOverview.pdf
	rm -rf Pelegant
