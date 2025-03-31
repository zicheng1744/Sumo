@echo off
echo SUMO-RL Training Batch Script
echo ===================================

REM Kill any existing SUMO and TRACI processes
echo Cleaning up existing SUMO processes...
taskkill /F /IM sumo.exe /T 2>nul
taskkill /F /IM sumo-gui.exe /T 2>nul
timeout /t 2 /nobreak >nul

REM Generate sustainable traffic flow
echo Generating traffic...
python generate_random_traffic.py --main_prob 0.3 --ramp_prob 0.2 --cav_prob 0.5 --duration 3600 --speed 30.0
echo Traffic generation completed

REM Wait a moment to ensure resources are freed
echo Waiting for resources to be freed...
timeout /t 3 /nobreak >nul

REM Run optimized training script
echo Starting training...
python train.py --mode train --total_timesteps 30000 --learning_rate 1e-3 --min_learning_rate 1e-5 --lr_schedule linear --episode_length 15000 --n_steps 2048 --batch_size 128 --n_epochs 8 --ent_coef 0.01 --max_grad_norm 0.5 --action_scale 10.0 --max_speed 30.0
echo Training completed

pause 