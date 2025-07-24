# 📈 Stock Prices Prediction Platform
![Stock Prediction Banner](./images/stockprices.jpg)
## 1. 🧩 Problem Description

This project aims to forecast future stock prices using a variety of statistical and machine learning models such as ARIMA, LSTM, and Monte Carlo simulations.

Time series data in financial markets is complex, non-linear, and highly volatile—posing a significant challenge.  
The goal is to provide a **reproducible**, **scalable**, and **production-ready** platform for managing the complete machine learning lifecycle: from data ingestion and transformation, to training, experiment tracking, and deployment.

---

## 2. ☁️ Cloud Infrastructure

The infrastructure is defined using **Infrastructure as Code (IaC)** principles. A `Makefile` handles environment setup and orchestration.

While the project runs locally, it is **cloud-ready** and compatible with **Docker** and **Kubernetes** for deployment.

It integrates with **MLflow** for experiment tracking and model registry, and could be easily adapted to cloud storage or cloud-native workflows.

---

## 3. 📊 Experiment Tracking & Model Registry

The project integrates **MLflow** for:

- Logging experiments and their metrics
- Tracking parameters and performance
- Saving serialized model artifacts (e.g., `.pkl` models, charts)

Both **experiment tracking** and **model registry** are actively used and embedded in the training pipeline.

---

## 4. 🔁 Workflow Orchestration

The project uses **[Mage](https://www.mage.ai/)** as a workflow orchestration tool to automate and schedule ML pipelines. Mage orchestrates:

- Data extraction  
- Model training  
- Evaluation  
- Export processes  

Each step is modularized into Mage pipelines, enabling:

- 🔄 Reproducible, automated workflows  
- 🛠 Monitoring and retrying failed steps  
- 🧩 A visual interface for debugging and versioning  

This orchestration enables robust and maintainable MLOps and can be deployed both locally and in the cloud.

---

## 5. 🚀 Model Deployment

- Trained models are serialized and saved automatically.
- A **Jupyter Notebook interface** can be launched via `make infra-up` for interactive testing.
- The project supports **containerization** and **cloud deployment** with minimal configuration.

---

## 6. 📈 Model Monitoring

Basic model monitoring is provided via:

- Tracking prediction error metrics
- Logging outputs from Monte Carlo simulations
- Comparing performance across models

![Monitoring Banner](./images/dashboard.jpg)
All metrics are stored and visualized through **MLflow**, allowing historical performance comparison.
---

## 7. 🧪 Reproducibility

- ✅ All dependencies listed in `requirements.txt`
- ✅ Environment setup is automated using `make infra-up`
- ✅ Jupyter notebooks launchable via CLI
- ✅ Modular, well-documented pipelines
- ✅ Full end-to-end execution with a few commands

---

## 8. 🧠 Best Practices

- ✅ **Unit tests**: Written using `pytest` for all major modules (data loaders, transformers, exporters)
- ✅ **Integration test**: Modular pipeline tested end-to-end
- ✅ **Linter & Formatter**: Uses `black` and `flake8`, integrated into pre-commit
- ✅ **Makefile**: Targets include infrastructure setup, testing, linting, notebook startup, and more
- ✅ **Pre-commit Hooks**: Configured via `.pre-commit-config.yaml`
- ✅ **CI/CD Pipeline**: GitHub Actions workflow for:
  - Installing dependencies
  - Running unit tests
  - Validating code format

---
## 9. 📁 Project Structure
```plaintext
Project_Final_Stock_Prices_Prediction/
├── images/                         # Project assets (e.g., README banners, plots)
├── infra/                          # Infrastructure & MLOps
│   ├── grafana/                    # Grafana dashboards & configs
│   ├── mage_data/                  # Mage workflow data
│   ├── mlflow_data/                # MLflow tracking data
│   ├── mlops/                      # Core ML logic
│   │   ├── data_exporters/         # Export processed results
│   │   ├── data_loaders/           # Load and ingest raw data
│   │   ├── pipelines/              # ML pipelines (e.g., prediction)
│   │   ├── presenters/             # Present output (dashboards, reports)
│   │   └── transformers/           # Feature engineering & transformation
│   └── scripts/                    # Auxiliary scripts (e.g., DB init)
├── mlruns/                         # MLflow experiments and model registry
├── tests/                          # Unit and integration tests
│   └── test_montecarlo_t.py        # Monte Carlo method test
```
---
## 10. 🚀 Quick Start Examples
-  Clone the repo
```bash
git clone https://github.com/your-username/Project_Final_Stock_Prices_Prediction.git
cd Project_Final_Stock_Prices_Prediction
```
- Set up environment and install dependencies
```bash
make infra-up
```
- Launch Jupyter Notebook for EDA or simulations
```bash
make eda
```
- Run all tests
```bash
make test
```
