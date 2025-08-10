from typing import Optional

from diophantineequations.distributed.workers.formalizer.abstract import AbstractFormalizationWorker, Conjecture
from diophantineequations.lemma_form import generate_formalizations, LeanFile
from diophantineequations.notify import send_notification
import logging

logging.basicConfig(level=logging.INFO)

class FormalizerGPT(AbstractFormalizationWorker):
    def get_formalizations(self, conjecture: Conjecture, definitions: list[LeanFile]) -> list[Optional[str]]:
        formalized = generate_formalizations(conjecture.informal_problem, definitions, tries=self._max_attempts)
        formalized_lines = [
            [line.strip() for line in formalproblem.split("\n") if not line.startswith("import") and line.strip()]
            if formalproblem is not None else None for formalproblem in formalized]
        formalized = ["\n".join(formalproblem_lines) if formalproblem_lines else None for formalproblem_lines in
                      formalized_lines]
        return formalized


def main():
    worker = FormalizerGPT.from_env_vars()
    worker.start()


if __name__ == '__main__':
    import traceback

    try:
        main()
    except Exception as e:
        send_notification(True, traceback.format_exc(), "GPTFormalizer")
        raise e
    send_notification(False, job_name="GPTFormalizer")
