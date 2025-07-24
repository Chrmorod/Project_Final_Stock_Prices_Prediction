PROJECT_DIR := Project_Final_Stock_Prices_Prediction
#INFRA_DIR := $(PROJECT_DIR)/infra
INFRA_DIR := infra
.DEFAULT_GOAL := help
STOCK ?= MSFT
YEAR_BACK ?= 3

install:
	pip install -r requirements.txt

infra-up:
#docker-compose -f $(INFRA_DIR)/docker-compose.yml up -d
	PROJECT_NAME=mlops \
		MAGE_CODE_PATH=/home/src \
		STOCK=$(STOCK) \
		YEAR_BACK=$(YEAR_BACK) \
		docker-compose -f $(INFRA_DIR)/docker-compose.yml up -d

infra-down:
	docker-compose -f $(INFRA_DIR)/docker-compose.yml down

export-mlflow:
	python mlflow_data_export.py

clean:
	rm -f *.csv
	rm -rf __pycache__

eda:
	python -m jupyter notebook EDA.ipynb

test:
	pytest

help:
	@echo "Comands availables:"
	@echo "  install        - Install dependencies of the project"
	@echo "  mlflow-up      - Run MLflow with docker-compose"
	@echo "  mlflow-down    - Stop containers MLflow"
	@echo "  export-mlflow  - Export data from MLflow to CSV"
	@echo "  clean          - Delete temp files"
	@echo "  eda            - Exec EDA Notebook"
