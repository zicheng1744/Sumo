#!/bin/bash
echo "SUMO-RL Training Batch Script"
echo "==================================="

# Kill any existing SUMO and TRACI processes
echo "Cleaning up existing SUMO processes..."
pkill -f sumo
pkill -f sumo-gui
sleep 2

# Generate sustainable traffic flow
echo "Generating traffic..."
cd grok_version
python generate_random_traffic.py --main_prob 0.7 --ramp_prob 0.3 --cav_prob 0.1 --duration 1000000 --speed 15.0
echo "Traffic generation completed"

# Wait a moment to ensure resources are freed
echo "Waiting for resources to be freed..."
sleep 3

# Run optimized training script
echo "Starting training..."
python train.py --mode train --total_timesteps 1000000 --learning_rate 1e-3 --min_learning_rate 1e-5 --lr_schedule linear --episode_length 100000 --n_steps 128 --batch_size 128 --n_epochs 8 --ent_coef 0.01 --max_grad_norm 0.5 --action_scale 1.0 --max_speed 15.0
echo "Training completed"

# auto plot
echo "Running speed analysis..."
python analyze_speed.py --show

read -p "Press Enter to continue..."