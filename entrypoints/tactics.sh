#!/bin/bash
# entrypoint for the tactics workers

module purge
module load rhel8/default-amp
module load sqlite
module load gcc/11

trap "echo \"$(date): Received SIGTERM/SIGINT, exiting.\"; exit 0" SIGTERM SIGINT

export VLLM_BIN=<path-to-vllm-bin>
export RABBIT_USER=rabbitmq
export RABBIT_PASSWORD=<your_rabbitmq_password>

PYTHON=<python-executable>
NOTIFY_SCRIPT=<path-to-notify-script>

while true; do
  echo "$(date): Starting tactics worker (PID $$)"
  $PYTHON <path-to-repository>/diophantineequations/distributed/workers/tactics/tactics.py

  exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo "$(date): tactics.py exited with code $exit_code — sending Telegram alert"
    $PYTHON $NOTIFY_SCRIPT --fail \
        --error "tactics.py on $(hostname) exited with code $exit_code"
  fi

  echo "$(date): tactics.py exited with code $exit_code — restarting in 5 seconds..."
  sleep 5
done
