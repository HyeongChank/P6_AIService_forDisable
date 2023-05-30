from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = 'gpt2'  # 모델의 이름 또는 경로
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 모델 사용 예시
input_text = "Hello, how are you?"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model.generate(input_ids)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
