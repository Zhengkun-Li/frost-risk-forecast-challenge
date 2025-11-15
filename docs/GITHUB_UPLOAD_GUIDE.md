# GitHub 上传指南

本指南说明如何将项目上传到 GitHub，以及哪些文件应该包含或排除。

## 📁 项目结构说明

### ✅ 应该包含的文件和文件夹

1. **源代码** (`src/`)
   - ✅ 所有 Python 源代码
   - ✅ 模型实现
   - ✅ 数据处理工具
   - ✅ 评估工具

2. **脚本** (`scripts/`)
   - ✅ 所有训练和工具脚本
   - ✅ 分析脚本

3. **配置文件**
   - ✅ `requirements.txt`
   - ✅ `config/` 目录下的配置文件
   - ✅ `README.md` 和所有文档

4. **实验结果的摘要和日志** (`experiments/`)
   - ✅ `summary.json` - 训练摘要
   - ✅ `metrics.json` - 评估指标
   - ✅ `*.log` - 训练日志
   - ✅ `training_history.json` - 训练历史
   - ✅ `training_curves.png` - 训练曲线图
   - ✅ `reliability_diagram.png` - 可靠性图
   - ✅ `calibration_curve.png` - 校准曲线
   - ✅ `predictions.json` - 预测结果（如果文件不大）

### ❌ 不应该包含的文件

1. **大型数据文件**
   - ❌ `data/raw/` - 原始数据文件
   - ❌ `*.csv`, `*.parquet`, `*.h5` - 大型数据文件

2. **模型权重文件**
   - ❌ `*.pkl`, `*.pth`, `*.cbm`, `*.model` - 模型权重文件
   - ❌ `checkpoints/` 目录 - Checkpoint 文件

3. **临时文件**
   - ❌ `__pycache__/` - Python 缓存
   - ❌ `*.pyc`, `*.pyo` - 编译的 Python 文件
   - ❌ `.DS_Store`, `Thumbs.db` - 系统文件

4. **虚拟环境**
   - ❌ `venv/`, `env/` - 虚拟环境目录

## 📋 .gitignore 说明

项目已包含 `.gitignore` 文件，它会自动排除：

- 所有 Python 缓存和编译文件
- 虚拟环境目录
- IDE 配置文件
- 大型数据文件（`.csv`, `.parquet`, `.h5` 等）
- 模型权重文件（`.pkl`, `.pth`, `.cbm`, `.model` 等）
- Checkpoint 文件
- 临时文件

但会保留：
- 实验摘要和日志（`.json`, `.log`）
- 训练曲线图（`.png`）
- 评估指标文件

## 🚀 上传到 GitHub 的步骤

### 1. 检查 Git 状态

```bash
# 检查当前状态
git status

# 查看会被忽略的文件
git status --ignored
```

### 2. 初始化 Git 仓库（如果还没有）

```bash
# 如果还没有初始化
git init

# 添加远程仓库
git remote add origin https://github.com/yourusername/frost-risk-forecast-challenge.git
```

### 3. 添加文件并提交

```bash
# 添加所有应该包含的文件
git add .

# 检查将要提交的文件
git status

# 提交
git commit -m "Initial commit: Frost Risk Forecast Challenge project"

# 推送到 GitHub
git push -u origin main
```

### 4. 验证上传的文件

上传后，检查 GitHub 仓库确保：
- ✅ 源代码都在
- ✅ 配置文件都在
- ✅ 实验摘要和日志都在
- ❌ 没有大型数据文件
- ❌ 没有模型权重文件

## 📊 实验结果的保留策略

### 保留的内容

实验目录 (`experiments/`) 中会保留：

```
experiments/
├── lightgbm/
│   └── top175_features/
│       ├── full_training/
│       │   ├── summary.json          ✅ 保留
│       │   ├── lightgbm_training.log ✅ 保留
│       │   ├── training_history.json ✅ 保留
│       │   └── training_curves.png   ✅ 保留
│       └── loso/
│           ├── summary.json          ✅ 保留
│           └── loso_training.log     ✅ 保留
```

### 排除的内容

- ❌ `labeled_data.parquet` - 大型数据文件
- ❌ `model.pkl`, `model.pth` - 模型权重
- ❌ `checkpoints/` - Checkpoint 目录
- ❌ `feature_importance.csv` - 大型特征重要性文件

## 💡 建议

1. **README.md**: 确保 README 包含：
   - 项目描述
   - 安装说明
   - 使用示例
   - 数据获取说明（如果数据不在仓库中）

2. **LICENSE**: 添加适当的许可证文件

3. **数据说明**: 在 README 中说明如何获取原始数据

4. **模型权重**: 如果需要分享训练好的模型，考虑使用：
   - GitHub Releases（适合小文件）
   - Google Drive / Dropbox（适合大文件）
   - Hugging Face Model Hub（适合机器学习模型）

## 🔍 检查清单

上传前检查：

- [ ] `.gitignore` 文件已创建并正确配置
- [ ] 没有大型数据文件被添加
- [ ] 没有模型权重文件被添加
- [ ] 实验摘要和日志已包含
- [ ] `requirements.txt` 已更新
- [ ] `README.md` 已更新
- [ ] 所有源代码都已包含

## 📝 示例命令

```bash
# 查看将被忽略的文件
git status --ignored | head -20

# 查看将要提交的文件大小
git ls-files | xargs ls -lh | sort -k5 -hr | head -20

# 检查是否有大文件
find . -type f -size +10M -not -path "./.git/*" | grep -v ".gitignore"
```

## ⚠️ 注意事项

1. **如果已经提交了大文件**：
   ```bash
   # 从 Git 历史中移除大文件（需要谨慎操作）
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch path/to/large/file" \
     --prune-empty --tag-name-filter cat -- --all
   ```

2. **GitHub 文件大小限制**：
   - 单个文件最大 100 MB
   - 仓库总大小建议不超过 1 GB
   - 如果超过，考虑使用 Git LFS

3. **如果需要分享模型权重**：
   - 使用 GitHub Releases
   - 或使用外部存储（Google Drive, Dropbox）
   - 在 README 中提供下载链接

