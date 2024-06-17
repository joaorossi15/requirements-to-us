import transformers
import peft
import pandas as pd
import datasets

def load_model(model: str):
    # load model
    m = transformers.AutoModelForCausalLM.from_pretrained(
        model,
        device_map='auto',
        trust_remote_code=False,
        revision='main'
    )
    
    return m

def load_and_prepare_dataset(model_name: str):
    df = pd.read_csv('/home/bielrossi/us-translator/data/data_openai_api_with_mask.csv')
    df = df[['text', 'ethical_us']]
    instruction = ''
    df['data'] = df.apply((lambda row: f'''<s>[INST] {instruction} \n{row['text']} \n[/INST] {row['ethical_us']}</s>'''), axis=1)
    
    t = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    dt = datasets.Dataset.from_pandas(df)
    
    t.pad_token = t.eos_token
    data_collator = transformers.DataCollatorForLanguageModeling(t, mlm=False)

    return dt, data_collator, t

def tokenize_function(examples):
    # extract text
    text = examples["data"]

    #tokenize and truncate text
    t.truncation_side = "left"
    tokenized_inputs = t(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs


def train_model(model, lr, batch_size, num_epochs, tokenized_data, collator):
    model.train() # training state
    model.gradient_checkpointing_enable()
    model = peft.prepare_model_for_kbit_training(model) # turn into qlora

    # lora config
    config = peft.LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj"],
        lora_dropout=.05,
        bias="none",
        task_type="CAUSAL_LLM"
    )

    model = peft.get_peft_model(model, config) # model in lora style
    

    training_args = transformers.TrainingArguments(
        output_dir= "../model",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        fp16=True,
        optim="paged_adamw_8bit",

    )

    trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_data,
    # eval_dataset=tokenized_data["test"],
    args=training_args,
    data_collator=collator
    )

    # train model
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    # renable warnings
    model.config.use_cache = True

    return model

def main():
    m = load_model("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")
    data, data_collator, t = load_and_prepare_dataset("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")
    tokenized_data = data.map(tokenize_function, batched=True)
    final_model = train_model(m, 1e-4, 4, 10, tokenized_data, data_collator)
    final_model.eval()