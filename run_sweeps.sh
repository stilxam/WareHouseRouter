#!/usr/bin/env bash
set -euo pipefail

# Each world gets its own independent Bayesian sweep so the optimizer cannot
# conflate world difficulty with hyperparameter quality.
# Runs: 3 worlds × 2 algos = 6 sweeps; PPO+DQN run in parallel per world,
# worlds run sequentially (2 processes at a time) to avoid OOM.

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45

WORLDS=(5 6 7)
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

# ── Run agents via pool of 2 — next job starts as soon as any slot frees ─────

JOBS=()
for WORLD in "${WORLDS[@]}"; do
    JOBS+=("ppo:${WORLD}:${PPO_IDS[$WORLD]}:logs/ppo_world_${WORLD}.log")
    JOBS+=("dqn:${WORLD}:${DQN_IDS[$WORLD]}:logs/dqn_world_${WORLD}.log")
done

MAX_PARALLEL=2
active=0

trap 'echo "Interrupted — killing all jobs."; kill 0; wait 2>/dev/null; exit 1' SIGINT SIGTERM

for job in "${JOBS[@]}"; do
    algo=$(cut -d: -f1 <<<"$job")
    world=$(cut -d: -f2 <<<"$job")
    sweep=$(cut -d: -f3 <<<"$job")
    log=$(cut -d: -f4 <<<"$job")

    if (( active >= MAX_PARALLEL )); then
        wait -n
        (( active-- ))
    fi

    wandb agent "$sweep" 2>&1 | tee "$log" &
    (( active++ ))
    echo "[*] Started ${algo} world=${world} (PID $!)  →  ${log}"
done

wait
echo "[*] All sweeps finished."
