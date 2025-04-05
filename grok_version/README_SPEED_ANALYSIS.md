# 车速分析工具使用说明

## 简介

本工具用于分析SUMO训练过程中的平均车速数据。通过记录每个step的平均车速，可以分析训练过程中智能体的行为变化。

## 使用方法

### 1. 训练并收集速度数据

运行训练程序，它会自动记录平均车速数据:

```bash
cd grok_version
python train.py --mode train
```

训练过程中，系统会自动在 `speed_data` 目录下生成CSV文件，记录每10个step的平均车速数据。

### 2. 分析速度数据

训练完成后，可以使用 `analyze_speed.py` 脚本分析速度数据:

```bash
cd grok_version
python analyze_speed.py
```

#### 可选参数:

- `--file`: 指定速度数据文件路径，不提供则使用最新的
- `--save_dir`: 图表保存目录，默认为 `./speed_analysis`
- `--show`: 显示图表(默认只保存不显示)

示例:

```bash
# 分析最新的速度数据文件并显示图表
python analyze_speed.py --show

# 分析指定的速度数据文件
python analyze_speed.py --file speed_data/speed_data_20250401_123456.csv

# 指定保存目录
python analyze_speed.py --save_dir my_analysis
```

## 数据文件格式

速度数据CSV文件包含以下列:

- `step`: 训练步数
- `avg_speed`: 该步骤的平均车速(m/s)
- `vehicle_count`: 该步骤的车辆数量

## 生成的图表

分析脚本会生成以下图表:

1. 平均车速与训练步数的关系图
2. 车辆数量与训练步数的关系图
3. 平均车速分布直方图
4. 平均车速与车辆数量的关系散点图

此外，脚本还会输出速度数据的基本统计信息和分段分析结果。

## 注意事项

- 由于每10步记录一次数据，图表中的x轴步数会以10为间隔
- 如果训练中断，可能需要手动整理数据文件
- 对于大型训练，数据文件可能会很大，建议适当增加记录间隔 