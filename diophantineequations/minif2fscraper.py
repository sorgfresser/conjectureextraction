from pathlib import Path
import json


def formal_to_individual_files(test_path: Path, result_dir: Path):
    result_dir.mkdir(exist_ok=True)
    with test_path.open("r") as f:
        data = f.read()
    data = data.removeprefix("import Mathlib")
    delim = ":= by sorry"
    theorems = data.split(delim)
    theorems = [substr + delim for substr in theorems[:-1]] + [theorems[-1]]
    theorems.pop(-1)
    assert all(theorem.strip().startswith("theorem") or theorem.strip().startswith("set_option") for theorem in
               theorems), next(theorem for theorem in theorems if not theorem.strip().startswith("theorem"))
    assert all(
        (" " + theorem.strip()).count(" theorem ") == 1 or theorem.strip().startswith("set_option") for theorem in
        theorems), next(theorem for theorem in theorems if (" " + theorem.strip()).count(" theorem ") != 1)
    print(theorems)
    names = []
    for theorem in theorems:
        if theorem.strip().startswith("theorem"):
            names.append(theorem.strip().removeprefix("theorem").strip().split(" ")[0].strip())
        elif theorem.strip().startswith("set_option"):
            names.append(
                theorem.strip().split("\n", maxsplit=1)[1].strip().removeprefix("theorem").strip().split(" ")[0].strip())
        else:
            raise RuntimeError("Should not happen!")
    for name, theorem in zip(names, theorems, strict=True):
        with (result_dir / (name + "_sol.lean")).open("w") as f:
            f.write("import Mathlib\n")
            f.write(theorem)


def informal_to_individual_files(jsonl_path: Path, result_dir: Path, valid_path: Path, test_path: Path):
    valid_informal = result_dir.with_stem(result_dir.stem + "_valid")
    valid_informal.mkdir(exist_ok=True)
    test_informal = result_dir.with_stem(result_dir.stem + "_test")
    test_informal.mkdir(exist_ok=True)

    with jsonl_path.open("r") as f:
        data = f.readlines()
    data = [json.loads(d) for d in data]

    for value in data:
        name = value["formal_statement"].strip().removeprefix("theorem").strip().split("\n")[0].split(" ")[0]
        assert value["split"] in ["valid", "test"]
        if value["split"] == "valid":
            assert (valid_path / (name + "_sol.lean")).exists()
        else:
            assert (test_path / (name + "_sol.lean")).exists()
        result = {"title": name, "problem": value["id"], "problem_text": value["informal_stmt"], "solution_text": value["informal_proof"]}
        if value["split"] == "valid":
            with (valid_informal / (name + ".json")).open("w") as f:
                json.dump(result, f)
        else:
            with (test_informal / (name + ".json")).open("w") as f:
                json.dump(result, f)

    # Check other way
    for filepath in valid_path.glob("*.lean"):
        assert (valid_informal / (filepath.stem.removesuffix("_sol") + ".json")).exists()
    for filepath in test_path.glob("*.lean"):
        assert (test_informal / (filepath.stem.removesuffix("_sol") + ".json")).exists()
