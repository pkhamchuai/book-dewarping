.PHONY: install-python-package setup clean

PYTHON := python3

install-python-package:
	@echo "Installing requirements python package"
	# install python3.10
	@sudo apt-get update
	@sudo apt-get install -y software-properties-common tesseract-ocr libgl1
	@sudo add-apt-repository -y ppa:deadsnakes/ppa
	@sudo apt-get update
	@sudo apt-get install -y python3.10 python3.10-venv
	@python3.10 -m venv .venv
	@.venv/bin/pip install --upgrade pip
	@.venv/bin/pip install -r requirements.txt
	# @.venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	# @.venv/bin/pip install torchsummary pytorch_model_summary

clean:
	@echo "Cleaning up"
	@rm -rf .venv

# Add a target for setup that depends on installing Python packages
setup: install-python-package create-directories
	@echo "ALL requirements are installed"

create-directories:
	@mkdir -p data

# To use setup and cleanup, you can run:
# make setup
# make clean

# sudo apt-get install python3-tk
# source .venv/bin/activate