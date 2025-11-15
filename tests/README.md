# 测试代码组织说明

**最后更新**: 2025-11-12

## 目录结构

```
tests/
├── __init__.py
├── conftest.py              # 共享 fixtures 和配置
├── README.md               # 本文档
├── data/                   # 数据模块测试
│   ├── __init__.py
│   ├── test_loaders.py     # 数据加载器测试
│   ├── test_cleaners.py    # 数据清洗器测试
│   └── test_feature_engineering.py  # 特征工程测试
├── models/                 # 模型模块测试（Phase 2）
│   ├── __init__.py
│   ├── test_base.py
│   └── test_lightgbm.py
└── evaluation/             # 评估模块测试（Phase 2）
    ├── __init__.py
    ├── test_metrics.py
    └── test_validators.py
```

## 运行测试

### 运行所有测试
```bash
pytest tests/
```

### 运行特定模块的测试
```bash
# 数据模块测试
pytest tests/data/

# 模型模块测试
pytest tests/models/

# 评估模块测试
pytest tests/evaluation/
```

### 运行特定测试文件
```bash
pytest tests/data/test_loaders.py
```

### 运行特定测试类或方法
```bash
pytest tests/data/test_loaders.py::TestDataLoader::test_load_raw_data_csv
```

### 带覆盖率的测试
```bash
pytest --cov=src tests/
pytest --cov=src --cov-report=html tests/
```

### 详细输出
```bash
pytest -v tests/
pytest -vv tests/  # 更详细
```

## 共享 Fixtures

在 `conftest.py` 中定义的 fixtures 可以在所有测试中使用：

- `sample_dataframe`: 基础样本数据
- `sample_dataframe_with_qc`: 带 QC 标记的数据
- `sample_dataframe_with_sentinels`: 带哨兵值的数据
- `sample_dataframe_with_missing`: 带缺失值的数据
- `temp_dir`: 临时目录
- `sample_feature_config`: 特征工程配置示例

## 测试编写规范

1. **测试类命名**: `Test<ClassName>`
2. **测试方法命名**: `test_<functionality>`
3. **使用 fixtures**: 尽量使用共享 fixtures 而不是重复创建数据
4. **断言清晰**: 每个测试应该有明确的断言
5. **独立性**: 每个测试应该独立运行，不依赖其他测试

## 示例

```python
import pytest
from src.data.loaders import DataLoader

class TestDataLoader:
    def test_load_raw_data_csv(self, temp_dir):
        """Test loading CSV file."""
        # 使用 temp_dir fixture
        csv_path = temp_dir / "test.csv"
        # ... 测试代码 ...
```

## 持续集成

测试应该在以下情况自动运行：
- 提交代码前（pre-commit hook）
- Pull Request 时
- 定期运行（nightly builds）

