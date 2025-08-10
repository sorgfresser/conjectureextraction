#!/bin/bash
# entrypoint for the tactics workers

module purge
module load rhel8/default-ccl
module load sqlite
module load gcc/11

trap "echo \"$(date): Received SIGTERM/SIGINT, exiting.\"; exit 0" SIGTERM SIGINT

export LEAN_PATH=<path-to-lean-project>
export LEAN_HEADER="import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"

PYTHON=<python-executable>
NOTIFY_SCRIPT=<path-to-notify-script>
export RABBIT_USER=rabbitmq
export RABBIT_PASSWORD=<your_rabbitmq_password>

while true; do
  echo "$(date): Starting environment worker (PID $$)"
  $PYTHON <path_to_environment_script>

  exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo "$(date): environment.py exited with code $exit_code — sending Telegram alert"
    $PYTHON $NOTIFY_SCRIPT --fail \
        --error "environment.py on $(hostname) exited with code $exit_code"
  fi

  echo "$(date): environment.py exited with code $exit_code — restarting in 5 seconds..."
  sleep 5
done
