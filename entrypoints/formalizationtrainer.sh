#!/bin/bash
# Entrypoint for the trainer

module purge
module load rhel8/default-amp
module load gcc/9

trap "echo \"$(date): Received SIGTERM/SIGINT, exiting.\"; exit 0" SIGTERM SIGINT

# Ensure the variables are defined
: "${RABBIT_SEARCH:?Need to set RABBIT_SEARCH}"
: "${RABBIT_FORMALIZATION:?Need to set RABBIT_FORMALIZATION}"
: "${RABBIT_TACTICS:?Need to set RABBIT_TACTICS}"

# Decide on POSTGRES_DB
if [[ "$RABBIT_SEARCH" == "nosearch" ]] \
  && [[ "$RABBIT_FORMALIZATION" == "gptformalization" ]] \
  && [[ "$RABBIT_TACTICS" == "tactics" ]]; then
  export POSTGRES_DB="rabbitnosearchdeepseekgptformalizer"
elif [[ "$RABBIT_SEARCH" == "searchnosearchgptdeepseekvalid" ]]; then
  export POSTGRES_DB="rabbitnosearchdeepseekgptformalizervalid"
elif [[ "$RABBIT_SEARCH" == "searchnodeepseekkiminaformalize" ]]; then
  export POSTGRES_DB="rabbitnosearchdeepseekkiminaformalize"
elif [[ "$RABBIT_SEARCH" == "searchnodeepseekkiminaformalizevalid" ]]; then
  export POSTGRES_DB="rabbitnosearchdeepseekkiminaformalizevalid"
elif [[ "$RABBIT_SEARCH" == "hammernosearch" ]] \
  && [[ "$RABBIT_FORMALIZATION" == "kiminaformalization" ]]; then
  export POSTGRES_DB="rabbitnosearchhammerkiminaformalize"
elif [[ "$RABBIT_SEARCH" == "searchdeepseekkiminaformalizeputnam" ]]; then
  export POSTGRES_DB="rabbitsearchdeepseekkiminaformalize"
elif [[ "$RABBIT_SEARCH" == "searchdeepseekkiminaformalizeputnamonline" ]]; then
  export POSTGRES_DB="rabbitsearchonlinedeepseekkiminaformalize"
elif [[ -z ${POSTGRES_URL} ]]; then
  echo "Error: no matching rule for:
  RABBIT_SEARCH='$RABBIT_SEARCH'
  RABBIT_FORMALIZATION='$RABBIT_FORMALIZATION'
  RABBIT_TACTICS='$RABBIT_TACTICS'"
  exit 1
fi
echo "POSTGRES_DB set to: $POSTGRES_DB"

if [[ -z ${POSTGRES_URL} ]]; then
  export POSTGRES_URL=postgresql+psycopg2://postgres:<postgres-password>@<postgres-ip>:5432/$POSTGRES_DB
fi
if [[ -z ${MODEL_DIR} ]]; then
  export MODEL_DIR=<some-model-directory-to-store-models>
fi
if [[ -z ${RABBIT_FORMALIZATION_TRAINER} ]]; then
  export RABBIT_FORMALIZATION_TRAINER=formalizationtrainer
fi

export DEEPSPEED_BIN=<path-to-deepspeed-bin>
export DEEPSPEED_CONFIG=<path-to-deepspeed-config-json>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=<path-to-venv/bin>:$PATH
export RABBIT_USER=rabbitmq
export RABBIT_PASSWORD=<your_rabbitmq_password>

PYTHON=<python-executable>
NOTIFY_SCRIPT=<path-to-notify-script>


while true; do
  echo "$(date): Starting formalization trainer (PID $$)"
  $PYTHON <path-to-repo>/diophantineequations/distributed/workers/formalizationtrain/formalizationtrainer.py

  exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo "$(date): formalizer.py exited with code $exit_code — sending Telegram alert"
    $PYTHON $NOTIFY_SCRIPT --fail \
        --error "formalizer.py on $(hostname) exited with code $exit_code"
  fi

  echo "$(date): formalizer.py exited with code $exit_code — restarting in 10 seconds..."
  sleep 10
done
