# Full pipeline. `make dev` is safe to re-run; `make test` only works after `make freeze`.
PY ?= python3

data:
	$(PY) code/01_fetch_hpd.py
	$(PY) code/02_order_categories.py

panel:
	$(PY) code/03_build_panel.py
	$(PY) code/07_descriptives.py

dev: panel
	$(PY) code/04_models.py dev
	$(PY) code/05_evaluate.py dev

link:
	$(PY) code/06_rentcast_link.py

freeze:
	$(PY) code/08_freeze.py

test:
	$(PY) code/04_models.py test
	$(PY) code/05_evaluate.py test

secondary:
	$(PY) code/09_fetch_acs.py
	$(PY) code/10_acs_context.py
	$(PY) code/11_recalibration.py
	$(PY) code/12_partial_recal.py
	$(PY) code/01_fetch_hpd.py violations_ab
	$(PY) code/13_sensitivity.py

paper:
	bash code/20_build_paper.sh

.PHONY: data panel dev link freeze test secondary paper
