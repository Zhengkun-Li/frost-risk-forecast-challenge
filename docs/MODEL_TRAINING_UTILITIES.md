# Model Training Utilities

## 概述

为了统一和复用训练功能，我们创建了一套通用的训练工具模块，位于 `src/models/utils/`。这些工具可以被所有模型类型（树模型、神经网络等）使用。

## 工具模块

### 1. TrainingHistory - 训练历史记录

统一的训练历史记录接口，支持任意指标的记录和保存。

**使用示例：**

```python
from src.models.utils import TrainingHistory

# 初始化
history = TrainingHistory(metrics=['train_loss', 'val_loss', 'learning_rate'])

# 开始训练
history.start_training()

# 记录每个 epoch
for epoch in range(1, num_epochs + 1):
    history.record_epoch(
        epoch=epoch,
        train_loss=0.5,
        val_loss=0.6,
        learning_rate=0.001,
        epoch_time=10.5
    )

# 保存历史
history.save(Path("training_history.json"))

# 加载历史
loaded_history = TrainingHistory.load(Path("training_history.json"))
```

**主要方法：**
- `start_training()`: 标记训练开始
- `record_epoch()`: 记录一个 epoch 的指标
- `get_history()`: 获取完整历史
- `get_latest(metric)`: 获取最新指标值
- `save(path)`: 保存到 JSON 文件
- `load(path)`: 从 JSON 文件加载

### 2. CheckpointManager - Checkpoint 管理

统一的 checkpoint 管理接口，支持定期保存和最佳模型保存。

**使用示例：**

```python
from src.models.utils import CheckpointManager
from pathlib import Path

# 初始化
checkpoint_mgr = CheckpointManager(
    checkpoint_dir=Path("checkpoints"),
    checkpoint_frequency=10,  # 每 10 个 epoch 保存一次
    save_best=True,
    best_metric="val_loss",
    best_mode="min"
)

# 训练循环中
for epoch in range(1, num_epochs + 1):
    # ... 训练代码 ...
    
    # 定期保存 checkpoint
    if checkpoint_mgr.should_save_checkpoint(epoch):
        checkpoint_mgr.save_checkpoint(
            epoch=epoch,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            metrics={'train_loss': 0.5, 'val_loss': 0.6},
            training_history=history
        )
    
    # 保存最佳模型
    checkpoint_mgr.save_best_checkpoint(
        epoch=epoch,
        model_state=model.state_dict(),
        metric_value=val_loss
    )
```

**主要方法：**
- `should_save_checkpoint(epoch)`: 检查是否应该保存 checkpoint
- `is_best(metric_value)`: 检查是否为最佳值
- `update_best(epoch, metric_value)`: 更新最佳值
- `save_checkpoint()`: 保存 checkpoint
- `save_best_checkpoint()`: 保存最佳模型
- `list_checkpoints()`: 列出所有 checkpoint
- `get_checkpoint_path(epoch)`: 获取 checkpoint 路径

### 3. TrainingCurvePlotter - 训练曲线绘制

统一的训练曲线绘制接口，支持单任务和多任务模型。

**使用示例：**

```python
from src.models.utils import TrainingCurvePlotter
from pathlib import Path

# 初始化
plotter = TrainingCurvePlotter(backend="matplotlib")

# 绘制单任务模型曲线
history_dict = {
    'epoch': [1, 2, 3, ...],
    'train_loss': [0.5, 0.4, 0.3, ...],
    'val_loss': [0.6, 0.5, 0.4, ...],
    'learning_rate': [0.001, 0.001, 0.0005, ...]
}
plotter.plot(
    history=history_dict,
    save_path=Path("training_curves.png"),
    title="Training Curves"
)

# 绘制多任务模型曲线
multitask_history = {
    'epoch': [1, 2, 3, ...],
    'train_loss_total': [0.5, 0.4, 0.3, ...],
    'val_loss_total': [0.6, 0.5, 0.4, ...],
    'train_loss_temp': [0.3, 0.2, 0.1, ...],
    'val_loss_temp': [0.4, 0.3, 0.2, ...],
    'train_loss_frost': [0.2, 0.2, 0.2, ...],
    'val_loss_frost': [0.2, 0.2, 0.2, ...],
    'learning_rate': [0.001, 0.001, 0.0005, ...]
}
plotter.plot_multitask(
    history=multitask_history,
    save_path=Path("multitask_curves.png"),
    title="Multi-task Training Curves"
)
```

**主要方法：**
- `plot()`: 绘制单任务模型曲线
- `plot_multitask()`: 绘制多任务模型曲线
- 自动检测可用后端 (matplotlib/plotly)

### 4. ProgressLogger - 进度日志

统一的进度日志接口，支持文件日志和 stdout。

**使用示例：**

```python
from src.models.utils import ProgressLogger
from pathlib import Path

# 初始化
logger = ProgressLogger(
    log_file=Path("training.log"),
    use_tqdm=True,
    flush_interval=1
)

# 记录训练开始
logger.log_training_start(
    model_name="LSTM",
    device="cuda",
    config={"batch_size": 32, "epochs": 100}
)

# 记录每个 epoch
for epoch in range(1, num_epochs + 1):
    logger.log_epoch(
        epoch=epoch,
        total_epochs=num_epochs,
        train_loss=0.5,
        val_loss=0.6,
        learning_rate=0.001,
        epoch_time=10.5,
        eta=100.0
    )

# 记录改进
logger.log_improvement("val_loss", 0.5, 0.6)

# 记录训练完成
logger.log_training_complete(total_time=1000.0, total_epochs=100)
```

**主要方法：**
- `log(message)`: 记录消息
- `log_training_start()`: 记录训练开始
- `log_epoch()`: 记录 epoch 信息
- `log_improvement()`: 记录指标改进
- `log_early_stopping()`: 记录早停
- `log_training_complete()`: 记录训练完成
- `get_tqdm()`: 获取 tqdm 进度条

## 集成到模型

### 在 BaseModel 中添加支持（可选）

可以在 `BaseModel` 中添加可选的训练工具支持：

```python
from src.models.base import BaseModel
from src.models.utils import TrainingHistory, CheckpointManager, TrainingCurvePlotter, ProgressLogger

class BaseModel(ABC):
    def __init__(self, config):
        # ... 现有代码 ...
        
        # 可选的训练工具
        self.training_history = None
        self.checkpoint_manager = None
        self.curve_plotter = None
        self.progress_logger = None
    
    def setup_training_tools(self, checkpoint_dir=None, log_file=None):
        """设置训练工具（可选）"""
        model_params = self.config.get("model_params", {})
        
        # 训练历史
        self.training_history = TrainingHistory()
        
        # Checkpoint 管理
        if checkpoint_dir:
            checkpoint_frequency = model_params.get("checkpoint_frequency", 10)
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                checkpoint_frequency=checkpoint_frequency
            )
        
        # 曲线绘制
        self.curve_plotter = TrainingCurvePlotter()
        
        # 进度日志
        self.progress_logger = ProgressLogger(log_file=log_file)
```

## 优势

1. **代码复用**: 避免在每个模型中重复实现相同功能
2. **统一接口**: 所有模型使用相同的接口，易于维护
3. **易于扩展**: 新模型可以轻松集成这些功能
4. **灵活性**: 工具可以独立使用，也可以组合使用
5. **可测试性**: 工具模块可以独立测试

## 未来改进

1. **为树模型添加支持**: 为 LightGBM、XGBoost 等添加 checkpoint 和训练曲线
2. **恢复训练功能**: 从 checkpoint 恢复训练
3. **更多后端支持**: 支持 plotly 等交互式绘图后端
4. **分布式训练支持**: 支持多 GPU/多节点训练的工具

