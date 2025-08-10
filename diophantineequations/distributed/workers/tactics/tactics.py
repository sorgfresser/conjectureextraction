from diophantineequations.distributed.messages import ActionTactics, ResponseTactics, ModelType
from diophantineequations.deepseekprover import reload_model as ds_reload_model, generate_tactic as ds_generate_tactic, \
    clean_model as ds_clean_model
from diophantineequations.reprover import reload_model as reprover_reload_model, \
    generate_tactic as reprover_generate_tactic, clean_model as reprover_clean_model
from diophantineequations.notify import send_notification
from diophantineequations.distributed.workers.tactics.abstract import AbstractTacticWorker
from diophantineequations.utils import theorem_name_span
import logging
from os import getenv

BLACKLISTED = ["sorry", "admit"]
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TacticWorker(AbstractTacticWorker):
    def __init__(self, vllm_bin: str):
        super().__init__()
        self._vllm_bin = vllm_bin

    def reload_model(self, action: ActionTactics):
        logger.info("Cleaning up old model of type %s", self.previous_model_type)
        if self.previous_model_type == ModelType.deepseek:
            ds_clean_model()
        elif self.previous_model_type == ModelType.reprover:
            reprover_clean_model()
        logger.info("Loading new model from %s of type %s", action.model_path, action.model_type)
        if action.model_type == ModelType.reprover:
            reprover_reload_model(action.model_path)
        elif action.model_type == ModelType.deepseek:
            ds_reload_model(action.model_path, vllm_bin=self._vllm_bin)
        else:
            raise ValueError("Invalid model type")

    def do_work(self, action: ActionTactics, diff: int) -> ResponseTactics:
        logger.debug("Generating %s tactics for goal %s", action.k, action.goal)
        logger.debug("Premises: %s", action.premises)
        if action.model_type == ModelType.deepseek:
            tactics, probs = ds_generate_tactic(action.goal, action.premises, action.k)
        elif action.model_type == ModelType.reprover:
            tactics, probs = reprover_generate_tactic(action.goal, action.premises, action.k)
        else:
            raise ValueError("Invalid model type")
        tactics, prob_list = self.filter_tactics(tactics, probs.tolist(), action)
        logger.debug("Filtered tactics: %s", tactics)
        return ResponseTactics(action=action, strings=tactics, logprobs=prob_list, ms_between=diff)

    @staticmethod
    def filter_tactics(tactics: list[str], probs: list[float], action: ActionTactics):
        filtered_tactics, filtered_probs = [], []
        for idx, tactic in enumerate(tactics):
            tactic = tactic.split("--")[0]  # had the case 'sorry -- exact hx' before
            if tactic.strip() in BLACKLISTED:
                continue
            if "sorry" in tactic:
                logger.error("Sorry found in tactic %s", tactic)
                continue
            if "admit" in tactic:
                logger.error("Admit found in tactic %s", tactic)
                continue
            filtered_tactics.append(tactic)
            filtered_probs.append(probs[idx])
        return filtered_tactics, filtered_probs

    @classmethod
    def from_env_vars(cls):
        vllm_bin = getenv("VLLM_BIN")
        if vllm_bin is None:
            raise ValueError("VLLM_BIN not set!")
        return cls(vllm_bin)


def main():
    worker = TacticWorker.from_env_vars()
    worker.start()


if __name__ == '__main__':
    from diophantineequations.distributed.gpu_lock import reserve_one_gpu
    import traceback

    try:
        reserve_one_gpu()
        main()
    except Exception as e:
        send_notification(True, traceback.format_exc(), "Tactics")
        raise e
    send_notification(False, job_name="Tactics")
