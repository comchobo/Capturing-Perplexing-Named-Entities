from transformers import M2M100ForConditionalGeneration, NllbTokenizerFast, set_seed
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import get_peft_config, get_peft_model, IA3Config, TaskType
from datasets import Dataset, load_from_disk, load_dataset, concatenate_datasets
import os, json, shutil, gc, copy
import numpy as np
import evaluate
os.environ["TQDM_DISABLE"] = "1"
import torch

def train_lora(args):
    tokenizer = NllbTokenizerFast.from_pretrained(
        "facebook/nllb-200-distilled-1.3B", src_lang="kor_Hang", tgt_lang="eng_Latn")

    model = M2M100ForConditionalGeneration.from_pretrained("facebook/nllb-200-distilled-1.3B")
    peft_config = IA3Config(target_modules=['k_proj', 'v_proj', 'out_proj'],
                            task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, feedforward_modules=[])
    model = get_peft_model(model, peft_config)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    def preprocess_function(examples):
        model_inputs = tokenizer(examples['ko'], text_target=examples['en'], max_length=72, truncation=True)
        return model_inputs

    if args.prune_method == 'random':
        train_dataset = load_from_disk(f'{args.data_path}/domain_datasets/{args.which_domain}')['train']
        prune_len = int(args.prune_len) * 1000
        if prune_len==2000:
            train_dataset = train_dataset.train_test_split(test_size=0.0025)
        elif prune_len==8000:
            train_dataset = train_dataset.train_test_split(test_size=0.01)
        else:
            train_dataset = train_dataset.train_test_split(test_size=0.02)
        pruned_train_dataset = Dataset.from_dict(train_dataset['train'][:prune_len])
        tokenized_train_dataset = pruned_train_dataset.map(preprocess_function, batched=True)
        tokenized_valid_dataset = train_dataset['test'].map(preprocess_function, batched=True, keep_in_memory=True)

    else:
        dataset_name = f'{args.data_path}/better_training_datasets/{args.which_domain}/' \
                       f'{args.which_domain}_{args.prune_method}_{args.prune_len}_{args.seed}'
        loaded_dataset = load_from_disk(dataset_name)
        tokenized_train_dataset = loaded_dataset[args.difficulty].map(preprocess_function, batched=True)
        tokenized_train_dataset = tokenized_train_dataset.remove_columns(['ko','en'])
        tokenized_valid_dataset = loaded_dataset['test'].map(preprocess_function, batched=True, keep_in_memory=True)
        tokenized_valid_dataset = tokenized_valid_dataset.remove_columns(['ko','en'])

    metric = evaluate.load("sacrebleu")
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    model_save_path = f'{args.output_path}/{args.which_domain}'
    model_save_name = f'Diff{args.difficulty}_{args.prune_method}_pruned_{args.prune_len}_seed{args.seed}_saved'
    HP_search_name = f'HP_Diff{args.difficulty}_{args.prune_method}_pruned_{args.prune_len}_seed{args.seed}'

    steps_standard = int(args.prune_len) * 20

    try:
        os.mkdir(f'{args.output_path}/{args.which_domain}')
    except FileExistsError:
        pass

    metric_accumul = []
    lr_list = {0:1e-2, 1:2e-2, 2:3e-2} #{0:1e-2, 1:2e-2, 2:3e-2}
    os.environ['WANDB_DISABLED'] = 'true'
    for trial_key in lr_list.keys():
        training_args = Seq2SeqTrainingArguments(
            report_to='none',
            generation_max_length=144,
            output_dir=f'{model_save_path}/{HP_search_name}_{trial_key}',
            evaluation_strategy="steps",
            save_strategy='steps',
            logging_steps=steps_standard,
            eval_steps=steps_standard * 4,
            save_steps=steps_standard * 4,
            max_steps=steps_standard * 16,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=2,
            per_device_eval_batch_size=24,
            warmup_steps=steps_standard,
            predict_with_generate=True,
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model='eval_bleu',
            learning_rate=lr_list[trial_key],
            tf32=True,
        )

        trainer = Seq2SeqTrainer(
            model=copy.deepcopy(model),
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_valid_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        trainer.model.save_pretrained(f'{model_save_path}/{HP_search_name}_{trial_key}_saved')
        metric_accumul.append(trainer.state.best_metric)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    save_target = metric_accumul.index(max(metric_accumul))
    print(metric_accumul)
    print(f'{model_save_path}/{HP_search_name}_{save_target}_saved',' will be saved')
    shutil.copytree(f'{model_save_path}/{HP_search_name}_{save_target}_saved', f'{model_save_path}/{model_save_name}')

    run_folders = [x for x in os.listdir(model_save_path)
                   if x.startswith(f'HP_Diff{args.difficulty}_{args.prune_method}_pruned') is True]
    for run_folder in run_folders:
        shutil.rmtree(f'{model_save_path}/{run_folder}', ignore_errors=True)

    try:
        os.mkdir(f'HPs')
        os.mkdir(f'HPs/{args.which_domain}')
    except FileExistsError:
        pass

    with open(f'HPs/{args.which_domain}/HP_found_Diff{args.difficulty}_{args.prune_method}_pruned_{args.prune_len}_{args.seed}.json','w') as f:

        json.dump(metric_accumul, f)
    del tokenized_train_dataset
    del tokenized_valid_dataset

if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prune_len", default='2') 
    parser.add_argument("--prune_method", default='random') # random, entropy_NE, entropy, el2n, ...
    parser.add_argument("--data_path", default='./sorryhyun_datasets') 
    parser.add_argument("--output_path", default= './server_weights') # /app/outputs
    parser.add_argument("--which_domain", default='law')
    parser.add_argument("--seed", default='40')
    # parser.add_argument("--difficulty", default='4')
    args = parser.parse_args()
    os.environ['PYTHONHASHSEED'] = str(42)

    for difficulty in ['0','1','2','3']:
        args.difficulty = difficulty
        set_seed(int(args.seed))
        train_lora(args)
        torch.cuda.empty_cache()