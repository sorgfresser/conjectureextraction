#!/bin/bash
# Entrypoint for the retrieval worker

module purge
module load rhel8/default-ccl
module load sqlite
module load gcc/11

trap "echo \"$(date): Received SIGTERM/SIGINT, exiting.\"; exit 0" SIGTERM SIGINT

export VECTOR_STORE=<path-to-persistent-vector-store>
export RABBIT_USER=rabbitmq
export RABBIT_PASSWORD=<your_rabbitmq_password>

cd <entrypoint-directory>

PYTHON=<python-executable>
NOTIFY_SCRIPT=<path-to-notify-script>

while true; do
  echo "$(date): Starting retrieval (PID $$)"
  $PYTHON <path-to-repository>/diophantineequations/distributed/workers/retrieval/retrieval.py

  exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo "$(date): retrieval.py exited with code $exit_code — sending Telegram alert"
    $PYTHON $NOTIFY_SCRIPT --fail \
        --error "retrieval.py on $(hostname) exited with code $exit_code"
  fi

  echo "$(date): retrieval.py exited with code $exit_code — restarting in 10 seconds..."
  sleep 10
done
