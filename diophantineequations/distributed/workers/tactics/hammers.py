from diophantineequations.distributed.messages import ActionTactics, ResponseTactics
from diophantineequations.notify import send_notification
from diophantineequations.distributed.workers.tactics.abstract import AbstractTacticWorker
from diophantineequations.utils import theorem_name_span
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HammerTacticWorker(AbstractTacticWorker):
    def reload_model(self, action: ActionTactics):
        return

    def do_work(self, action: ActionTactics, diff: int) -> ResponseTactics:
        logger.debug("Generating %s tactics for goal %s", action.k, action.goal)
        logger.debug("Premises: %s", action.premises)
        tactics = ["aesop", "simpa", "simp", "linarith", "decide"]
        spans = [theorem_name_span(premise) for premise in action.premises]
        names = [premise[span[0]: span[1]].removeprefix("theorem").strip() for premise, span in
                 zip(action.premises, spans, strict=True)]
        logger.debug("Names: %s", names)
        probs = [-0.4, -0.7, -0.3]
        for name in names:
            tactics.append(f"apply {name}")
            probs.append(-0.3)
        return ResponseTactics(action=action, strings=tactics, logprobs=probs, ms_between=diff)


def main():
    worker = HammerTacticWorker()
    worker.start()


if __name__ == '__main__':
    import traceback

    try:
        main()
    except Exception as e:
        send_notification(True, traceback.format_exc(), "Tactics")
        raise e
    send_notification(False, job_name="Tactics")
