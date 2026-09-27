#!/bin/bash

export PYTHONDONTWRITEBYTECODE=1

export PYTHONPATH=$(pwd):${PYTHONPATH}

# Picku your VLA-JEPA checkpoint
# your_ckpt=/home/ubuntu/checkpoints/VLA-JEPA/Pretrain/checkpoints/VLA-JEPA-pretrain.pt
# your_ckpt=/home/ubuntu/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
# your_ckpt=/home/ubuntu/checkpoints/VLA-JEPA/Real-world/checkpoints/VLA-JEPA-Real-World.pt
your_ckpt=/home/ubuntu/VLA-JEPA/checkpoints/MATTERIX_ACTION_HEAD/checkpoints/steps_10000_pytorch_model.pt



host="127.0.0.1"
base_port=15083

num_trials=1
max_steps=400

cuda_id=0

echo "========================================"
echo "Starting VLA-JEPA model server"
echo "Using GPU: ${cuda_id}"
echo "========================================"

# VLA-JEPA model serverを起動
python ./deployment/model_server/server_policy.py \
    --ckpt_path "${your_ckpt}" \
    --port "${base_port}" \
    --use_bf16 \
    --cuda "${cuda_id}" &

server_pid=$!

echo "Model server PID: ${server_pid}"

# 終了時に必ずmodel serverを停止
cleanup() {
    echo
    echo "Stopping model server PID: ${server_pid}"
    kill "${server_pid}" 2>/dev/null
    wait "${server_pid}" 2>/dev/null
}

trap cleanup EXIT INT TERM

# サーバー起動待ち
sleep 15

echo "========================================"
echo "Starting MATTERIX evaluation"
echo "========================================"

python ./examples/MATTERIX/eval_matterix.py \
    --host "${host}" \
    --port 5555 \
    --pretrained-path "${your_ckpt}" \
    --model-host "${host}" \
    --model-port "${base_port}" \
    --num-trials "${num_trials}" \
    --max-steps "${max_steps}" \
    --task-description "Pick up the beaker" \
    --with-state \
    --save-video  
    

eval_status=$?

if [ "${eval_status}" -ne 0 ]; then
    echo "ERROR: MATTERIX evaluation failed."
    exit "${eval_status}"
fi

echo "========================================"
echo "MATTERIX evaluation completed."
echo "========================================"