# 🌡️ F3 Innovate --- Frost Risk Forecasting Challenge (2025)

**Author:** Zhengkun LI  
**Email:** zhengkun.li3969@gmail.com  
**Affiliation:** TRIC Robotics / UF ABE / F3 Innovate Participant  
**Platform:** National Data Platform (NDP)  
**Last Updated:** 2025-11-13  
**Data Repository:**
https://github.com/CarlSaganPhD/frost-risk-forecast-challenge

------------------------------------------------------------------------

## 🚀 Overview

This repository hosts an end-to-end solution for the **F3 Innovate Frost
Risk Forecasting Challenge (2025)**.\
The objective is to develop probabilistic, spatially-generalizable frost
risk models using 15 years of hourly meteorological data from 18 CIMIS
stations across California.

The project integrates **station-level time series**, **geospatial
topography**, and **reanalysis synoptic fields** into a
**spatio-temporal deep learning pipeline**.

------------------------------------------------------------------------

## 🧾 Executive Summary

- **Goal**: deliver calibrated probabilistic frost forecasts (3h/6h/12h/24h horizons) with quantified uncertainty and LOSO generalization.
- **Primary Deliverables**: reproducible data pipeline, multi-model training stack, evaluation reports, deployment-ready inference services (batch + API).
- **Stakeholders**: F3 Innovate challenge organizers, specialty crop growers, TRIC Robotics field teams.
- **Decision Support**: configurable frost thresholds, reliability curves, and station-level alerts integrate with grower workflows.

------------------------------------------------------------------------

## 📦 Status & Deliverables

  Item                        Owner         Status      Output / Location
  --------------------------  ------------  ----------  ----------------------------------------------------
  Data acquisition            Z. Li         ✅ Complete  `data/raw/`, `data/external/`
  QC & exploratory analysis   Z. Li         ✅ Complete  `docs/DATA_DOCUMENTATION.md`, `docs/figures/`
  Feature engineering         Z. Li         ✅ Complete  298 features, `docs/FEATURE_ENGINEERING.md`, `docs/FEATURE_REFERENCE.md`
  Training pipelines          Z. Li         ✅ Complete  `scripts/train/train_frost_forecast.py`
  Model training              Z. Li         ✅ Complete  LightGBM (Top 175), XGBoost (in progress)
  LOSO evaluation             Z. Li         ✅ Complete  `experiments/lightgbm/top175_features/lightgbm/loso/`
  Inference services          Z. Li         ✅ Complete  `scripts/inference/predict_frost.py`
  Reporting & documentation   Z. Li         ✅ Complete  `docs/report/`, comprehensive analysis reports

------------------------------------------------------------------------

## 🔄 Data Processing Pipeline

1. **Raw ingestion** → pull hourly CIMIS observations, station metadata, and ERA5/HRRR reanalysis tiles into `data/raw/` and `data/external/`.
   - Sources: CIMIS API dumps, `scripts/fetch_station_metadata.py`, ERA5/HRRR NetCDF (planned ingestion via `src/data_prep/download_reanalysis.py`).
2. **Quality control & cleaning** → decode QC flags, replace sentinel values, harmonize timestamps, and write clean intermediates to `data/interim/`.
   - Assets: `scripts/generate_data_report.py`, exploratory notebooks in `notebooks/eda/`.
   - Outputs: `data/processed/station_overview.csv`, `data/processed/missing_by_station.csv`, QA plots in `docs/figures/`.
3. **Feature engineering** → derive traditional frost indicators (e.g., growing degree hours, rolling minima, humidity deficits), radiative proxies, synoptic summaries, and topo-climatic descriptors; persist feature tensors to `data/processed/`.
   - Planned scripts: `src/data_prep/features_tabular.py`, `src/data_prep/features_grid.py`.
   - Outputs: `data/processed/tabular_features.parquet`, `data/processed/gridded_patches.zarr`.
4. **Dataset assembly** → merge targets and features, build LOSO splits, and export train/val/test manifests for tabular and deep-learning pipelines.
   - Manifests: `data/processed/splits/lo_station_<id>.json`.
   - Metadata: `docs/data_catalog.md` (to capture column dictionary, feature provenance).

------------------------------------------------------------------------

## 🧩 Core Goals

  -----------------------------------------------------------------------
  Objective                        Description
  -------------------------------- --------------------------------------
  Frost Event Forecasting          Predict frost event probability (T \<
                                   0 °C) and Tmin for 3h, 6h, 12h, and
                                   24h horizons.

  Calibration & Reliability        Quantify uncertainty using Brier, ECE,
                                   PR-AUC, ROC-AUC, and Reliability
                                   Diagrams.

  Spatial Generalization (LOSO)    Evaluate how well models transfer to
                                   unseen CIMIS stations.

  Synoptic Integration             Fuse ERA5/HRRR cold-air advection,
                                   cloud cover, and radiative cooling
                                   fields.

  Interpretability & Decision      Provide calibrated probabilities and
  Support                          actionable thresholds.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 🧠 Modeling Framework & Baselines

### Traditional Feature Engineering Baseline

  Feature Block           Examples / Notes                                       Status
  ----------------------  -----------------------------------------------------  --------------------------------
  Temporal indicators     Lagged Tmin/Tdew (1–24 h), rolling minima/maxima,      Specification drafted, implementation in progress
                          diurnal amplitude, chilling hours, freeze duration
  Humidity & radiation    Vapor pressure deficit, saturation deficit,            Derived from station history + ERA5 cloud cover
                          longwave cooling proxy, clear-sky radiation residuals
  Dynamics                Cold-air advection magnitude, pressure tendency,       Requires ERA5/HRRR gradients (planned)
                          wind shift flags, Richardson number estimates
  Topo-climatic context   Elevation, slope, aspect, cold-air pooling index,      DEM ingestion pipeline planned
                          distance to water bodies/valleys
  Persistence heuristics  Historical Tmin quantiles for day-of-year, analog      Utilizes 15-year archive; will be cached in feature store
                          matching scores

- **Implementation**: pipeline in `src/data/feature_engineering.py` producing 298 features; models in `src/models/ml/`.
- **Modeling stack**: LightGBM and XGBoost with calibrated probabilities, comprehensive feature importance analysis.
- **Feature Selection**: Top 175 features (90% importance) identified and used for final models.
- **Results**: Excellent performance with ROC-AUC > 0.98 for all horizons, excellent spatial generalization (LOSO).

### Deep & Hybrid Models

1.  Temporal Sequence Models (LSTM / TCN)
    - Multi-horizon decoder predicting Tmin and frost probabilities jointly; Lightning modules under `src/models/temporal/`.
2.  Image-Based Models (CNN → TCN / ConvLSTM)
    - Process ERA5/HRRR patches, export embeddings for downstream fusion.
3.  Spatio-Temporal Graph Neural Networks (ST-GNN)
    - Nodes represent stations, edges weighted by topographic and meteorological affinity.
4.  Hybrid Fusion (CNN Embeddings + GBDT)
    - Combines deep synoptic embeddings with engineered tabular features through LightGBM.
5.  Reliability Calibration
    - Platt scaling, isotonic regression, and conformal wrappers applied per station/horizon.

------------------------------------------------------------------------

## 🧭 Validation: Leave-One-Station-Out (LOSO)

Each run excludes one station as a completely unseen test site.\
Performance is summarized as **mean ± SD across 18 stations**, plus
per-station tables.

------------------------------------------------------------------------

## ❄️ Frost Label Configuration

- Default frost event definition: `Tmin < 0 °C`.
- Configurable thresholds via Hydra config group `labels.threshold_c` to support crop-specific risk bands (e.g., -2 °C for almonds, +1 °C for berries).
- Supports multi-level targets (`frost_warn`, `frost_alert`) emitted alongside continuous Tmin regression labels.

------------------------------------------------------------------------

## ⚙️ Implementation Stack

  Component         Library / Framework
  ----------------- ----------------------------------------------------
  Data Processing   pandas, geopandas, rasterio, shapely
  ML / DL           PyTorch, PyTorch Lightning, scikit-learn, LightGBM
  Visualization     matplotlib, seaborn, plotly
  Reproducibility   Jupyter, Hydra configs, Makefile
  Environment       Python ≥3.10, CUDA ≥12, RTX 5090 GPU

------------------------------------------------------------------------

## 📈 Experiments

  Track                         Description / Notes                                 Status
  ----------------------------  --------------------------------------------------- --------------------------------------
  Baseline tabular models       LightGBM, XGBoost, logistic regression ensembles;   Feature spec complete, training scripts in progress
                               benchmark against climatology and persistence
  Temporal deep models          TCN, LSTM, Seq2Seq with attention for multi-horizon Prototyping in `notebooks/models/temporal.ipynb`
                               Tmin + probability outputs
  Synoptic image encoders       CNN → TCN / ConvLSTM on ERA5/HRRR patches           Data loader under development
  ST-GNN                        Graph Neural Network leveraging station topology    Architecture drafted, pending implementation
  Hybrid fusion                 Combine CNN embeddings with engineered features      Design finalized; awaiting export pipeline
  Calibration & conformal       Reliability diagrams, Platt, isotonic, conformal    Baseline scripts in `notebooks/calibration/`
  Ablation studies              Feature block removal, threshold sensitivity        Planned once baselines stabilize
  LOSO benchmarking             Aggregate metrics mean±SD; per-station dashboards   Template dashboard in `docs/figures/lo_station_demo.png`

------------------------------------------------------------------------

## 🧾 Project Structure

    frost-risk-forecast-challenge/
    ├── data/                    # 数据目录
    │   ├── raw/                 # 原始数据
    │   ├── interim/             # 中间数据
    │   └── processed/           # 处理后的数据
    ├── src/                     # 源代码
    │   ├── data/                # 数据加载、清洗、特征工程
    │   ├── models/              # 模型实现（LightGBM, XGBoost等）
    │   ├── evaluation/          # 评估指标和验证方法
    │   ├── visualization/       # 可视化工具
    │   └── utils/               # 工具函数
    ├── scripts/                 # 脚本目录
    │   ├── train/               # 训练脚本
    │   ├── inference/           # 推理脚本
    │   ├── evaluate/            # 评估脚本
    │   └── data_prep/           # 数据准备脚本
    ├── experiments/             # 实验结果
    │   ├── lightgbm/            # LightGBM模型
    │   │   ├── feature_importance/  # 特征重要性分析
    │   │   └── top175_features/    # Top 175特征配置
    │   │       ├── full_training/  # 标准评估
    │   │       └── loso/          # LOSO评估
    │   └── xgboost/              # XGBoost模型
    │       └── top175_features/    # Top 175特征配置
    │           └── full_training/  # 标准评估
    ├── config/                  # 配置文件
    ├── docs/                    # 文档目录
    │   ├── report/              # 分析报告
    │   └── figures/             # 图表
    ├── tests/                   # 测试代码
    └── README.md                # 项目说明（本文件）

------------------------------------------------------------------------

## 📚 Documentation

### 主要文档（已重新组织）

**用户文档**:
- **[USER_GUIDE.md](USER_GUIDE.md)**: 完整用户指南 - 从环境设置、快速开始到高级使用

**技术文档**:
- **[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)**: 技术文档 - 架构设计、API参考、配置管理

**数据文档**:
- **[DATA_DOCUMENTATION.md](DATA_DOCUMENTATION.md)**: 数据文档 - 数据概览、QC处理、变量使用情况

**特征工程**:
- **[FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md)**: 特征工程文档 - 特征设计、实现和分析
- **[FEATURE_REFERENCE.md](FEATURE_REFERENCE.md)**: 特征参考文档 - 完整的特征列表、获取方法和功能说明（298个特征）

**训练和评估**:
- **[TRAINING_AND_EVALUATION.md](TRAINING_AND_EVALUATION.md)**: 训练和评估文档 - 训练配置、LOSO评估、性能对比、特征工程和LOSO的关系

### 其他文档

**项目状态**:
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)**: 项目状态总览（历史文档，主要状态信息见本README）

**快速开始**:
```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行完整流程（推荐）
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --model lightgbm \
    --output experiments/lightgbm/top175_features/lightgbm \
    --top-k-features 175

# 运行LOSO评估
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --model lightgbm \
    --loso \
    --save-loso-models \
    --output experiments/lightgbm/top175_features/lightgbm \
    --top-k-features 175
```

详细使用说明请参考 [USER_GUIDE.md](USER_GUIDE.md)。

## 🧮 Results Summary

### LightGBM (Top 175 Features) - Standard Evaluation

  Horizon   Brier ↓   ECE ↓   ROC-AUC ↑   PR-AUC ↑   MAE ↓   RMSE ↓   R² ↑
  --------- --------- ------- ----------- ---------- -------- -------- --------
  3h        0.0028    0.0015   0.9965      0.9965     1.14     1.52     0.9703
  6h        0.0040    0.0025   0.9926      0.9926     1.55     2.02     0.9481
  12h       0.0043    0.0025   0.9892      0.9892     1.79     2.33     0.9304
  24h       0.0060    0.0048   0.9843      0.9843     1.93     2.51     0.9196

### LightGBM (Top 175 Features) - LOSO Evaluation

  Horizon   ROC-AUC ↑   MAE ↓   RMSE ↓   R² ↑
  --------- ----------- -------- -------- --------
  3h        0.9974     1.14     1.52     0.9703
  6h        0.9938     1.55     2.02     0.9481
  12h       0.9905     1.79     2.33     0.9304
  24h       0.9878     1.93     2.51     0.9196

**关键发现**:
- ✅ 优秀的空间泛化能力（LOSO ROC-AUC > 0.98 对所有时间窗口）
- ✅ 出色的概率校准（Brier Score < 0.01，ECE < 0.005）
- ✅ 高精度温度预测（MAE < 2°C，R² > 0.91）

详细结果请参考 [docs/report/LIGHTGBM_ANALYSIS.md](report/LIGHTGBM_ANALYSIS.md) 和 [docs/report/CALIBRATION_AND_RELIABILITY_REPORT.md](report/CALIBRATION_AND_RELIABILITY_REPORT.md)。

------------------------------------------------------------------------

## ⚙️ Configuration & Experiment Management

- **Hydra config tree**

      configs/
      ├── train/
      │   ├── tabular_baseline.yaml
      │   ├── cnn_tcn.yaml
      │   └── stgnn.yaml
      ├── data/
      │   ├── loaders/
      │   │   └── cimis_station.yaml
      │   └── transforms/
      └── labels/
          └── threshold_c.yaml

- **Sample training config (`configs/train/tabular_baseline.yaml`)**

      defaults:
        - data: loaders/cimis_station
        - model: lightgbm_baseline
        - labels: threshold_c@labels=zero_celsius
      trainer:
        max_epochs: 120
        callbacks:
          - type: early_stopping
            monitor: val/brier
            patience: 10
      station_split:
        strategy: loso
        holdout_station: ${station.id}

- **Experiment tracking**
  - PyTorch Lightning loggers integrated with MLflow (primary) and Weights & Biases (optional).
  - Naming convention: `{model_type}-{feature_version}-st{station_id}`; metrics aggregated via `scripts/aggregate_metrics.py`.
- **Automation**
  - `Makefile` targets wrap Hydra commands (`make train MODEL=cnn_tcn STATION=47`).
  - `scripts/run_sweep.py` launches multi-station sweeps; results synced to `reports/experiments/<date>/`.
- **Reproducibility controls**
  - Seed management through Hydra (`+seed=1234`), deterministic CuDNN toggles in Lightning utilities, environment locked via `poetry.lock`.

------------------------------------------------------------------------

## 🧩 Deployment

**Batch Inference**

``` bash
python -m src.infer.batch_infer --checkpoints runs/cnn_tcn/best.ckpt --inputs data/processed/patches.zarr --out outputs/preds.parquet
```

1. **Environment**: `conda env create -f environment.yaml` (CUDA ≥13), followed by `poetry install` to sync exact package pins.
2. **Inputs**: requires LOSO manifest, feature parquet/zarr, and metadata; support for cloud sync via `aws s3 sync` or `gsutil rsync`.
3. **Outputs**: parquet/feather containing `[timestamp, station_id, horizon_hr, p_frost, tmin_pred, model_version, data_version]`.
4. **Scheduling**: Airflow DAG `dags/frost_batch.py` (planned) executes hourly; fallback cron script `cron/forecast_batch.sh`.

**API Service**

``` bash
docker build -t frost:latest -f docker/Dockerfile .
docker run --gpus all -p 8000:8000 frost:latest python -m src.infer.serve_api
```

1. **Image**: base `nvidia/cuda:13.0.0-runtime-ubuntu22.04`, installs Poetry deps, copies `src/` and `configs/`.
2. **Runtime**: mount `/models` for checkpoint hot-swap; environment variables (`MODEL_PATH`, `THRESHOLD_C`) injected via secrets manager.
3. **Endpoints**:
   - `POST /v1/forecast`: accepts JSON payload with recent observations; returns multi-horizon probabilities + Tmin forecasts.
   - `GET /v1/healthz`: readiness/liveness probe for Kubernetes.
4. **Deployment targets**: on-prem GPU node (RTX 5090) or cloud (GCP A2, AWS g5). Terraform manifests planned under `infra/`.
5. **Monitoring**: Prometheus metrics (`forecast_latency_ms`, `probability_shift`, `api_errors_total`), structured logging to Loki/CloudWatch.

------------------------------------------------------------------------

## 📊 Operations, Monitoring & Maintenance

- **Data freshness**: nightly sync validates new CIMIS observations; alerts fire if any station lags >3 hours.
- **Model drift**: rolling Brier score and calibration error monitored; auto-retrain triggered when degradation exceeds 10% from baseline.
- **Incident response**: runbooks (planned `docs/runbooks/frost_alerts.md`) define escalation within 30 minutes to agronomy lead.
- **Security & compliance**: secrets injected via `.env` and Vault/SSM; Mapbox tokens kept out of version control with IAM-scoped access.

------------------------------------------------------------------------

## 🛠️ Hardware & Environment Guidelines

- **Reference workstation**: AMD Ryzen 9 9950X, 64 GB RAM, NVIDIA RTX 5090 (32 GB VRAM), driver 580.95.05, CUDA 13.
- **Minimum training spec**: 16 GB VRAM GPU (RTX 4080 / A4000) with mixed precision; adjust batch sizes to hold <12 GB usage.
- **Batch inference footprint**: <4 GB VRAM, <2 GB RAM; throughput ≈250 station-horizon forecasts/s on RTX 5090.
- **Scaling**: PyTorch Lightning DDP for multi-GPU; Ray Tune integration planned for hyperparameter sweeps; parquet + Arrow optimize IO.

------------------------------------------------------------------------

## 🧭 Roadmap & Next Steps

- Finalize feature engineering scripts and publish `docs/data_catalog.md`.
- Implement full Hydra/Lightning pipelines with MLflow logging and artifact versioning.
- Automate ERA5/HRRR ingestion and QC checkpoints.
- Build CI/CD (GitHub Actions) to lint, test, containerize, and push inference images.
- Develop Streamlit/Plotly dashboards for LOSO evaluation and station-level monitoring.

------------------------------------------------------------------------

## 🧭 Research Vision

Paper title: *Evaluating Spatial Generalization of Frost Forecast Models
Across California: A Multi-Modal Deep Learning Benchmark.*

------------------------------------------------------------------------

## 🧾 License

MIT License --- For research and educational use.

------------------------------------------------------------------------

## 🛠️ Unified Training & Inference Interface

- **Hydra Configs**: centralize experiment settings under `configs/`, enabling reproducible sweeps (e.g., `python -m src.train.run --config-name tabular_baseline station=71`).
- **Lightning Trainers**: wrap PyTorch modules with Lightning for checkpointing, early stopping, and mixed precision; shared callbacks ensure consistent logging across models.
- **API Alignment**: standardize `fit()`, `predict_proba()`, and `forecast()` signatures for tabular and deep models, simplifying downstream evaluation and deployment.
