from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# # 사전 학습된 모델과 토크나이저 로드
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# 사전 학습된 모델과 토크나이저를 불러옵니다.
tokenizer = GPT2Tokenizer.from_pretrained('./result_storage')
# config.json 파일이 있는 경로로 가야 함
model = GPT2LMHeadModel.from_pretrained('./result_storage/checkpoint-2')
# 학습 데이터 준비
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="D:/김형찬/P6_disabled/P6_EmpCont/data/training_data.txt", # 학습 데이터 텍스트 파일 경로
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)
# 학습 설정
training_args = TrainingArguments(
    output_dir="./result_storage", # 학습 결과를 저장할 경로
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    # 스텝마다 체크포인트 저장
    save_steps=1,
    save_total_limit=2,
)
# 트레이너 생성
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
# 학습 시작
trainer.train()

# 학습 후 토크나이저를 저장
tokenizer.save_pretrained("./result_storage")
