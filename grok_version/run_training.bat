@echo off
echo SUMO-RL 训练批处理脚本
echo ===================================

REM 首先生成新的随机车流
echo 正在生成随机车流...
python generate_random_traffic.py --main_prob 0.5 --ramp_prob 0.3 --cav_prob 0.5 --duration 1000 
echo 车流生成完成

REM 运行修复版训练脚本
echo 开始训练...
python train.py train --total_timesteps 300000 --learning_rate 0.0003 --episode_length 15000 --gui

echo 训练完成！
pause 