from lean_repl_py import LeanREPLHandler
from pathlib import Path
import json

def main():
    root_path = Path("/home/simon/PycharmProjects/diophantineequations/FLT")
    handler = LeanREPLHandler(root_path)
    file_to_read = "/home/simon/PycharmProjects/diophantineequations/FLT/FLT/AutomorphicForm/QuaternionAlgebra/Defs.lean"
    json_str = json.dumps(
        {"path": file_to_read, "allTactics": True}
    )
    handler.send_json_str(json_str)
    result, env = handler.receive_json()
    print(result)


if __name__ == '__main__':
    main()
