#!/bin/bash
# Entrypoint for the retrieval worker

module purge
module load rhel8/default-ccl
module load sqlite
module load gcc/11

trap "echo \"$(date): Received SIGTERM/SIGINT, exiting.\"; exit 0" SIGTERM SIGINT

if [[ -z ${LEAN_ROOT} ]]; then
  export LEAN_ROOT=<path-to-a-lean-project>
fi
export LEAN_SOURCE=$LEAN_ROOT/Putnamproject/proven
export VECTOR_STORE=<path-to-persistent-vector-store>
export LEAN_FILE=$LEAN_ROOT/Putnamproject.lean
export RABBIT_USER=rabbitmq
export RABBIT_PASSWORD=<your_rabbitmq_password>
export TWILIO_ACCOUNT_SID=<your_twilio_account_sid>
export TWILIO_AUTH_TOKEN=<your_twilio_auth_token>
export TWILIO_FROM_NUMBER=<number-on-twilio>
export ALERT_PHONE_NUMBER=<number_to_call>

cd <entrypoint-directory>

PYTHON=<python-executable>
NOTIFY_SCRIPT=<path-to-notify-script>

$PYTHON $NOTIFY_SCRIPT --call

while true; do
  echo "$(date): Starting embedding (PID $$)"
  $PYTHON <path_to_embed_script>

  exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo "$(date): embed.py exited with code $exit_code — sending Telegram alert"
    $PYTHON $NOTIFY_SCRIPT --fail \
        --error "embed.py on $(hostname) exited with code $exit_code"
  fi

  echo "$(date): embed.py exited with code $exit_code — restarting in 10 seconds..."
  sleep 10
done
