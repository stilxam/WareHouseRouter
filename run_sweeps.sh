#!/usr/bin/env bash
set -euo pipefail

# Each world gets its own independent Bayesian sweep so the optimizer cannot
# conflate world difficulty with hyperparameter quality.
# Runs: 3 worlds × 2 algos = 6 sweeps; PPO+DQN agents run in parallel per world,
# worlds run sequentially to stay within single-GPU memory (2 × 0.45 fraction).

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45

WORLDS=(6 7 34)
mkdir -p logs sweeps

# ── Register all 6 sweeps up-front ──────────────────────────────────────────

declare -A PPO_IDS DQN_IDS

for WORLD in "${WORLDS[@]}"; do
    # Patch world_seed into temp configs using Python (avoids fragile sed)
    python3 - <<PYEOF
import yaml, pathlib
for src, dst in [("sweep.yaml", "sweeps/ppo_world_${WORLD}.yaml"),
                 ("sweep_dqn.yaml", "sweeps/dqn_world_${WORLD}.yaml")]:
    cfg = yaml.safe_load(pathlib.Path(src).read_text())
    cfg["parameters"]["world_seed"] = {"value": ${WORLD}}
    pathlib.Path(dst).write_text(yaml.dump(cfg))
PYEOF

    echo "[*] Registering PPO sweep  (world_seed=${WORLD})..."
    PPO_OUT=$(wandb sweep "sweeps/ppo_world_${WORLD}.yaml" 2>&1 | tee /dev/stderr)
    PPO_IDS[$WORLD]=$(echo "$PPO_OUT" | grep -oE 'wandb agent [^ ]+' | awk '{print $3}')

    echo "[*] Registering DQN sweep  (world_seed=${WORLD})..."
    DQN_OUT=$(wandb sweep "sweeps/dqn_world_${WORLD}.yaml" 2>&1 | tee /dev/stderr)
    DQN_IDS[$WORLD]=$(echo "$DQN_OUT" | grep -oE 'wandb agent [^ ]+' | awk '{print $3}')
done

echo ""
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│  Registered sweeps                                          │"
echo "├──────────┬──────────────────────────────┬───────────────────┤"
echo "│  World   │  PPO sweep ID                │  DQN sweep ID     │"
echo "├──────────┼──────────────────────────────┼───────────────────┤"
for WORLD in "${WORLDS[@]}"; do
    printf "│  %-8s│  %-28s│  %-17s│\n" \
        "$WORLD" "${PPO_IDS[$WORLD]}" "${DQN_IDS[$WORLD]}"
done
echo "└──────────┴──────────────────────────────┴───────────────────┘"
echo ""

# ── Run agents world-by-world ────────────────────────────────────────────────

for WORLD in "${WORLDS[@]}"; do
    echo "[*] ── World seed ${WORLD} ────────────────────────────────────"

    wandb agent "${PPO_IDS[$WORLD]}" \
        2>&1 | tee "logs/ppo_world_${WORLD}.log" &
    PPO_PID=$!

    wandb agent "${DQN_IDS[$WORLD]}" \
        2>&1 | tee "logs/dqn_world_${WORLD}.log" &
    DQN_PID=$!

    echo "[*] PPO PID $PPO_PID  →  logs/ppo_world_${WORLD}.log"
    echo "[*] DQN PID $DQN_PID  →  logs/dqn_world_${WORLD}.log"

    trap "kill $PPO_PID $DQN_PID 2>/dev/null; echo 'Stopped.'; exit 1" SIGINT SIGTERM

    wait $PPO_PID $DQN_PID
    echo "[*] World ${WORLD} done."
    echo ""
done

echo "[*] All sweeps finished."
