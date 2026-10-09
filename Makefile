PYTHON ?= .venv/bin/python

.PHONY: setup doctor check test smoke
setup:
	python3 -m venv .venv

doctor:
	$(PYTHON) scripts/jev.py doctor

check:
	$(PYTHON) scripts/jev.py check-config --config configs/week01/protocol.json

test:
	$(PYTHON) -m unittest discover -s tests -v

smoke:
	$(PYTHON) scripts/jev.py smoke
