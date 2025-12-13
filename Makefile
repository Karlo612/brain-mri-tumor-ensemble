PYTHON ?= python
CFG ?= config.yaml
BACKBONE ?= xception

.PHONY: install train eval gradcam check_personal_ids

install:
	$(PYTHON) -m pip install -e .

train:
	$(PYTHON) train.py --cfg $(CFG) --backbone $(BACKBONE)

eval:
	$(PYTHON) eval.py --cfg $(CFG)

gradcam:
	$(PYTHON) -m brain_mri_tumor_ensemble.gradcam --cfg $(CFG) --model_path $(MODEL)

check_personal_ids:
	@if rg -n "19003070" . --glob '!.git' --glob '!Makefile'; then \
		echo "Prohibited university ID found in repository. Please remove it."; \
		exit 1; \
	else \
		echo "No occurrences of the prohibited university ID detected."; \
	fi
