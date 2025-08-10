"""
Script to train autoformalization models on syntactically correct autoformalization samples. Data is generated via distillformalizersft.py from the postgres database.
"""
import logging
import os

logger = logging.getLogger(__name__)


def main():
    from datasets import load_dataset
    import evaluate
    dataset = load_dataset("sorgfresser/formalizationsamples", split="train")

    from transformers import AutoTokenizer, EvalPrediction
    from trl import SFTConfig, SFTTrainer
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    output_dir = "qwentrain0.5b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bleu = evaluate.load("bleu")

    def to_chat_template(batch):
        prompts = []
        completions = []
        for prompt, completion in zip(batch["prompt"], batch["completion"]):
            p, c = tokenizer.apply_chat_template(prompt + completion, tokenize=False, add_generation_prompt=False).split("<|im_start|>assistant")
            prompts.append(p)
            c = "<|im_start|>assistant" + c
            completions.append(c)
        batch["prompt"] = prompts
        batch["completion"] = completions
        return batch

    def filter_length(elem, length: int = tokenizer.model_max_length):
        if len(tokenizer(elem["prompt"] + elem["completion"]).input_ids) > length:
            return False
        return True

    def compute_metrics(pred: EvalPrediction):
        max_preds = pred.predictions.argmax(-1)
        mask = pred.label_ids != -100
        acc = (pred.label_ids == max_preds)[mask].mean()
        labels = [string + tokenizer.eos_token for string in
                  tokenizer.decode(pred.label_ids[mask]).split(tokenizer.eos_token)[:-1]]
        for idx, b in enumerate(mask):
            if not b.nonzero() or len(b.nonzero()[0]) == 0:
                continue
            last_idx = b.nonzero()[0][-1]
            if max_preds[idx, last_idx] != tokenizer.eos_token_id:
                max_preds[idx, last_idx] = tokenizer.eos_token_id
        preds = [string + tokenizer.eos_token for string in
                 tokenizer.decode(max_preds[mask]).split(tokenizer.eos_token)[:-1]]
        preds = preds if len(preds) <= len(labels) else preds[:len(labels)]
        # If more labels than preds, simply increment preds as if it were labels many
        if len(labels) > len(preds):
            preds = preds + [""] * (len(labels) - len(preds))
        return {"accuracy": acc, "bleu": bleu.compute(predictions=preds, references=labels)["bleu"]}

    dataset = dataset.map(to_chat_template, batched=True, batch_size=1000)
    dataset = dataset.filter(lambda x: filter_length(x, tokenizer.model_max_length // 64))
    split_dataset = dataset.train_test_split(test_size=0.0007)
    training_args = SFTConfig(
        max_length=tokenizer.model_max_length,
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        logging_steps=2,
        eval_steps=30000,
        eval_strategy="steps",
        eval_on_start=False,
        num_train_epochs=3,
        padding_free=False,
        eval_accumulation_steps=16,
        save_steps=300,
        push_to_hub=True
    )

    trainer = SFTTrainer(model_name,
                         train_dataset=split_dataset["train"],
                         eval_dataset=split_dataset["test"],
                         args=training_args,
                         compute_metrics=compute_metrics,
                         )
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(output_dir)


if __name__ == '__main__':
    main()
