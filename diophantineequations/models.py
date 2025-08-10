from dataclasses import dataclass, asdict
from typing import List, Optional
from pathlib import Path
from logging import getLogger
from abc import ABC, abstractmethod
from diophantineequations.utils import replace_sorry
import json

logger = getLogger(__name__)


@dataclass
class LeanFile:
    filepath: Path
    content: str
    full_proof: Optional[str] = None

    def import_string(self, relative_to: Path) -> str:
        logger.debug("Import string called with relative_to %s and filepath %s", relative_to, self.filepath)
        if not relative_to.is_dir():
            logger.error("Relative to path is not a directory, cannot import")
            raise ValueError("Relative to path is not a directory")
        if self.filepath.is_absolute():
            logger.debug("self.filepath is absolute")
            # Need to replace symlinks
            filepath = self.filepath.resolve()
            relative = filepath.relative_to(relative_to.resolve()).with_suffix("")
        elif not relative_to.is_absolute():
            logger.debug("self.filepath and relative_to are both relative")
            relative = self.filepath.relative_to(relative_to).with_suffix("")
        else:
            logger.debug("relative_to is absolute, simply using filepath")
            relative = self.filepath.with_suffix("")
        logger.info("Relative path: %s", relative)
        replaced = str(relative).replace("/", ".")
        # Remove leading lake section
        if ".lake." in replaced:
            split = replaced.split(".lake.")
            replaced = split[0].removesuffix(".")
            for split_part in split[1:]:
                if len(split_part.split(".")) > 2:
                    replaced = replaced + "." if replaced else replaced
                    replaced += ".".join(split_part.split(".")[2:])
        return f"import {replaced}\n"


@dataclass
class BaseConjecture(ABC):
    nl_conjecture: str
    formalized_conjecture: str
    imports: List[str]

    @property
    @abstractmethod
    def is_proven(self) -> bool:
        """Check if the conjecture is proven"""
        pass

    def to_file(self, path: Path, namespace: Optional[str] = None) -> LeanFile:
        """Write the conjecture to a file.

        :param path: Path to write the conjecture to
        :return: LeanFile object representing the written conjecture"""
        logger.info("Writing conjecture to %s", path)
        content = self._file_content()
        logger.debug("Conjecture content: %s", content)
        if namespace:
            imports = [line for line in content.splitlines(keepends=False) if line.strip().startswith("import")]
            rest = [line for line in content.splitlines(keepends=False) if not line.strip().startswith("import")]
            namespace = namespace.strip()
            namespace_line = f"namespace {namespace}"
            end_namespace_line = f"end {namespace}"
            content = "\n".join(imports + [namespace_line] + rest + [end_namespace_line])
        with open(path, "w") as f:
            f.write(content)
        logger.info("Written conjecture to %s", path)
        return LeanFile(path, content)

    @abstractmethod
    def _file_content(self) -> str:
        """Get the content of the file to write"""
        pass

    def json(self) -> dict:
        """Get the JSON representation of the conjecture"""
        return asdict(self)

    def save_json(self, json_path: Path):
        """Save the conjecture to a JSON file

        :param json_path: Path to save the JSON file to"""
        logger.info("Saving conjecture to JSON file %s", json_path)
        json_obj = self.json()
        logger.debug("Conjecture JSON: %s", json_obj)
        with open(json_path, "w") as f:
            json.dump(json_obj, f)


@dataclass
class FormalizedConjecture(BaseConjecture):
    def _file_content(self) -> str:
        return "\n".join(self.imports) + "\n" + self.formalized_conjecture

    @property
    def is_proven(self) -> bool:
        return False


@dataclass
class ProvenTheorem(BaseConjecture):
    proof: str

    def _file_content(self) -> str:
        final_theorem = replace_sorry(self.formalized_conjecture, self.proof)
        return "\n".join(self.imports) + "\n" + final_theorem

    @classmethod
    def from_formalized_conjecture(cls, conjecture: FormalizedConjecture, proof: str):
        """Create a ProvenTheorem from a FormalizedConjecture and a proof

        :param conjecture: The conjecture to prove
        :param proof: The proof of the conjecture

        :return: A ProvenTheorem object"""
        return cls(conjecture.nl_conjecture, conjecture.formalized_conjecture, conjecture.imports, proof)

    @property
    def is_proven(self) -> bool:
        return True
