@echo off
echo SUMO-RL 训练批处理脚本
echo ===================================

REM 首先生成新的随机车流
echo 正在生成随机车流...
python generate_random_traffic.py --main_prob 0.5 --ramp_prob 0.3 --cav_prob 0.4 --duration 100
echo 车流生成完成

REM 运行修复版训练脚本
echo 开始训练...
python test_training_fixed.py

echo 训练完成！
pause 