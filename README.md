# Conjecture extraction

This repository contains the source code for conjecture extraction, an algorithm combining conjecture generation through informal reasoning models, online autoformalization, and online automated theorem proving to generate and prove conjectures (and ultimately the original statements) in the Lean theorem prover.

## Navigating

The main entrypoint is `./diophantineequations/distributed/main.py`, which runs the distributed experiments.
Experiments use multiple simultaneous workers to process and generate conjectures, which communicate via RabbitMQ.

You can find the entrypoints for workers in `./entrypoints/`, where some things need to be set for your custom setup.

I'm also happy to provide slurm files privately, though not publicly in this repository.

This repo uses `poetry` for dependency management. To install dependencies, run:

```bash
poetry install
```


Make sure to have Lean's elan installed, and a valid Lean project setup somewhere, which has to be provided to the workers / main script.

## How does it work?

The overall idea is this:

![Overall overview](https://raw.githubusercontent.com/sorgfresser/conjectureextraction/master/assets/overall.png "Overall overview")

## Precise architecture
There are a lot of services, all communicating via RabbitMQ.
To reproduce, make sure to set-up RabbitMQ, ChromaDB, and Postgres. You can use docker-compose for this, following `chromadb.yml` and `rabbitmq.yml` in the root directory.

![Precise architecture](https://raw.githubusercontent.com/sorgfresser/conjectureextraction/master/assets/architecture.png "Precise architecture")
