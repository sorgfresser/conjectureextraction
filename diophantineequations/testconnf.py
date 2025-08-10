from pathlib import Path
import json
from random import randint, shuffle
from tqdm import tqdm

library = Path(__file__).parent.parent / "connf" / "library.jsonl"
benchmark = Path(__file__).parent.parent / "connf" / "benchmark.jsonl"

benchmark_objects = []
with benchmark.open("r") as file:
    for line in file:
        benchmark_objects.append(json.loads(line))
keys = benchmark_objects[0].keys()
print(keys)

library_objects = []
with library.open("r") as file:
    for line in file:
        library_objects.append(json.loads(line))

library_keys = library_objects[0].keys()
library_by_key = {obj["full_name"]: obj for obj in library_objects}
print(library_keys)

connf_distractors = [obj for obj in library_objects if obj["full_name"].startswith("ConNF")]

data = []

for bench in benchmark_objects:
    deps = bench["mathlib_dependencies"]
    context_objects = []
    if bench["formal_stmt"].strip().startswith("@[simp]"):
        continue
    if bench["informal_stmt"].strip().startswith("The theorem") or bench["informal_stmt"].strip().startswith("In the context of "):
        continue
    for dep in deps:
        if dep.startswith("ConNF"):
            context_objects.append(library_by_key[dep]["code"])
    # Sample a few random objects from the library to use as distractors
    for _ in range(randint(1, 7)):
        random_obj = connf_distractors[randint(0, len(connf_distractors) - 1)]
        if random_obj["full_name"] not in deps:
            context_objects.append(random_obj["code"])
    # Shuffle the context
    shuffle(context_objects)
    context = "\n".join(context_objects)
    print(context)
    print(bench["formal_stmt"])
    print(bench["informal_stmt"])
    append_bool = input("Append this to the dataset? (y/n): ")
    if append_bool.lower() != 'y':
        continue
    data.append({"context": context, "informal": bench["informal_stmt"], "formal": bench["formal_stmt"]})

from datasets import Dataset

dataset = Dataset.from_list(data)
dataset.save_to_disk("connf_dataset")
