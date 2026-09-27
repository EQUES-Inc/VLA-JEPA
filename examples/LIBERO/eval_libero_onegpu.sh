#!/bin/bash

export PYTHONDONTWRITEBYTECODE=1

export LIBERO_HOME=/home/ubuntu/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}

# LIBERO用conda環境のPython
export sim_python=/home/ubuntu/miniconda3/envs/libero/bin/python

# VLA-JEPA checkpoint
your_ckpt=/home/ubuntu/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

# 4つのtask suiteを1つずつ順番に実行
items=("libero_10" "libero_goal" "libero_object" "libero_spatial")

host="127.0.0.1"
base_port=15083

num_trials_per_task=1
with_state="true"

# GPUは1枚なのでGPU 0固定
cuda_id=0


for task_suite_name in "${items[@]}"
do
    echo "========================================"
    echo "Starting task suite: ${task_suite_name}"
    echo "Using GPU: ${cuda_id}"
    echo "========================================"

    port=${base_port}

    video_out_path="results/${task_suite_name}/${folder_name}"

    LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
    mkdir -p "${LOG_DIR}"
    mkdir -p "${video_out_path}"


    # VLA-JEPA model serverを起動
    python ./deployment/model_server/server_policy.py \
        --ckpt_path "${your_ckpt}" \
        --port "${port}" \
        --use_bf16 \
        --cuda "${cuda_id}" &

    server_pid=$!

    echo "Model server PID: ${server_pid}"

    # サーバー起動待ち
    sleep 15


    # LIBERO evaluationを実行
    ${sim_python} ./examples/LIBERO/eval_libero.py \
        --args.pretrained-path "${your_ckpt}" \
        --args.host "${host}" \
        --args.port "${port}" \
        --args.task-suite-name "${task_suite_name}" \
        --args.num-trials-per-task "${num_trials_per_task}" \
        --args.video-out-path "${video_out_path}" \
        --args.with_state "${with_state}" \
        > "${video_out_path}/eval.log" 2>&1

    eval_status=$?


    # 評価終了後、モデルサーバーを終了
    echo "Stopping model server PID: ${server_pid}"
    kill "${server_pid}" 2>/dev/null
    wait "${server_pid}" 2>/dev/null


    if [ "${eval_status}" -ne 0 ]; then
        echo "ERROR: Evaluation failed for ${task_suite_name}"
        echo "Check log:"
        echo "${video_out_path}/eval.log"
        exit "${eval_status}"
    fi

    echo "Completed: ${task_suite_name}"
    echo

done


echo "========================================"
echo "All LIBERO task suites completed."
echo "========================================"

