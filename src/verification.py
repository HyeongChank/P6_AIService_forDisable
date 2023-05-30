from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 사전 학습된 모델과 토크나이저를 불러옵니다.
tokenizer = GPT2Tokenizer.from_pretrained('./result_storage')
# config.json 파일이 있는 경로로 가야 함
model = GPT2LMHeadModel.from_pretrained('./result_storage/checkpoint-27')

# 입력 텍스트를 준비합니다.
input_text = "민원 업무"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 모델을 사용하여 출력을 생성합니다.
output = model.generate(input_ids, max_length=50, num_return_sequences=1, temperature=0.7)

# 출력을 텍스트로 디코딩합니다.
generated_text = tokenizer.decode(output[0])
print(generated_text)
