#!/usr/bin/env bash
set -euo pipefail

WORLDS=(5 14 27)
RUNS_PER_SWEEP=10 # must match `run_cap` in sweep*.yaml
mkdir -p logs sweeps

PPO_IDS=()
DQN_IDS=()


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
