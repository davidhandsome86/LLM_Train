from transformers import BertConfig,BertLMHeadModel,BertTokenizer,LineByLineTextDataset,DataCollatorForLanguageModeling,Trainer, TrainingArguments

tokenizer = BertTokenizer.from_pretrained("./bert") #tokenizer_bert生成的分词器

config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    is_decoder=True
)
model = BertLMHeadModel(config)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./sanguo.txt", #你的训练数据 your training_data
    block_size=32
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_device_train_batch_size=16,
    save_steps=2000,
    save_total_limit=2
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()

model.save_pretrained("./bert")  #模型和分词器应放到一个文件夹下，要不模型无法推理

# 模型推理
from transformers import pipeline, set_seed

generator = pipeline("text-generation", model="./bert")
set_seed(42)
txt = generator("吕布", max_length=50)
print(txt)

txt = generator("接着奏乐", max_length=50)
print(txt)


