@echo off

echo Generating random traffic...

cd final_version
python generate_random_traffic.py --main_prob 0.7 --ramp_prob 0.3 --cav_prob 0.5 --duration 1000000 --speed 15.0

echo Random traffic generation completed

REM Wait for resources to be released
echo Waiting for resources to be released...
timeout /t 3 /nobreak >nul

REM Set pretest directory
set PRETEST_DIR=pretest_results\pretest_session_%timestamp%

REM Run pretest (without model control)
echo Running pretest (observing traffic flow without model control)...
python pretest.py --gui --episode_length 10000 --result_dir %PRETEST_DIR%

echo Pretest completed

REM Wait for resources to be released
echo Waiting for resources to be released...
timeout /t 3 /nobreak >nul


REM Run test (using model control)
echo Running test (controlling traffic flow with model)...
python test.py --gui --episode_length 10000 --test_episodes 1 --model_path "C:\Users\18576\Sumo\final_version\results\training_session_20250406_151319\models\final_model.zip" --pretest_dir "%PRETEST_DIR%"

echo Test completed
pause 