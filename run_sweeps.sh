#!/usr/bin/env bash
set -euo pipefail

# Register sweeps, launch both agents on the same GPU with memory split 45/45
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45

echo "[*] Registering PPO sweep..."
PPO_OUT=$(wandb sweep sweep.yaml 2>&1 | tee /dev/stderr)
PPO_ID=$(echo "$PPO_OUT" | grep -oE 'wandb agent [^ ]+' | awk '{print $3}')

echo "[*] Registering DQN sweep..."
DQN_OUT=$(wandb sweep sweep_dqn.yaml 2>&1 | tee /dev/stderr)
DQN_ID=$(echo "$DQN_OUT" | grep -oE 'wandb agent [^ ]+' | awk '{print $3}')

echo ""
echo "[*] PPO sweep ID: $PPO_ID"
echo "[*] DQN sweep ID: $DQN_ID"
echo ""

mkdir -p logs

echo "[*] Launching agents..."
wandb agent "$PPO_ID" 2>&1 | tee logs/ppo_sweep.log &
PPO_PID=$!

wandb agent "$DQN_ID" 2>&1 | tee logs/dqn_sweep.log &
DQN_PID=$!

echo "[*] PPO agent PID: $PPO_PID  (logs/ppo_sweep.log)"
echo "[*] DQN agent PID: $DQN_PID  (logs/dqn_sweep.log)"
echo "[*] Both running. Ctrl+C to stop both."
echo ""

trap "kill $PPO_PID $DQN_PID 2>/dev/null; echo 'Stopped.'" SIGINT SIGTERM

wait $PPO_PID $DQN_PID
echo "[*] Both sweeps finished."
