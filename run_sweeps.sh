#!/usr/bin/env bash
set -euo pipefail

# Sequential Bayesian sweeps — one wandb agent at a time.
# Backend-agnostic: JAX picks whatever device is available (CPU on Mac,
# CUDA on 4090). No XLA env vars set here; export them in your shell if
# you need them on a specific machine.
#
# Each world × algo runs as its own sweep so the optimizer cannot conflate
# world difficulty with hyperparameter quality. 3 worlds × 2 algos = 6 sweeps.

# WORLDS=(5 7)
WORLDS=(14 27)
RUNS_PER_SWEEP=10 # must match `run_cap` in sweep*.yaml
mkdir -p logs sweeps

# Parallel indexed arrays — macOS ships bash 3.2 which lacks `declare -A`.
PPO_IDS=()
DQN_IDS=()

# ── Register all 6 sweeps up-front ──────────────────────────────────────────

for WORLD in "${WORLDS[@]}"; do
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
  PPO_IDS+=("$(echo "$PPO_OUT" | grep -oE 'wandb agent [^ ]+' | awk '{print $3}')")

  echo "[*] Registering DQN sweep  (world_seed=${WORLD})..."
  DQN_OUT=$(wandb sweep "sweeps/dqn_world_${WORLD}.yaml" 2>&1 | tee /dev/stderr)
  DQN_IDS+=("$(echo "$DQN_OUT" | grep -oE 'wandb agent [^ ]+' | awk '{print $3}')")
done

echo ""
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│  Registered sweeps                                          │"
echo "├──────────┬──────────────────────────────┬───────────────────┤"
echo "│  World   │  PPO sweep ID                │  DQN sweep ID     │"
echo "├──────────┼──────────────────────────────┼───────────────────┤"
for i in "${!WORLDS[@]}"; do
  printf "│  %-8s│  %-28s│  %-17s│\n" \
    "${WORLDS[$i]}" "${PPO_IDS[$i]}" "${DQN_IDS[$i]}"
done
echo "└──────────┴──────────────────────────────┴───────────────────┘"
echo ""

# ── Run agents sequentially ─────────────────────────────────────────────────
# One sweep at a time. Each agent runs in a fresh process; JAX/XLA state is
# fully reset between sweeps. Pass --count explicitly so the agent exits
# after RUNS_PER_SWEEP instead of waiting on the server for more work.

JOBS=()
for i in "${!WORLDS[@]}"; do
  WORLD="${WORLDS[$i]}"
  JOBS+=("dqn:${WORLD}:${DQN_IDS[$i]}:logs/dqn_world_${WORLD}.log")
done
for i in "${!WORLDS[@]}"; do
  WORLD="${WORLDS[$i]}"
  JOBS+=("ppo:${WORLD}:${PPO_IDS[$i]}:logs/ppo_world_${WORLD}.log")
done

trap 'echo "Interrupted."; exit 130' SIGINT SIGTERM

for job in "${JOBS[@]}"; do
  IFS=: read -r algo world sweep log <<<"$job"
  echo "[*] Starting ${algo} world=${world} sweep=${sweep}  →  ${log}"
  wandb agent --count "$RUNS_PER_SWEEP" "$sweep" 2>&1 | tee "$log"
  echo "[*] Finished ${algo} world=${world}"
done

echo "[*] All sweeps finished."
