from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 사전 학습된 모델 및 토크나이저 초기화
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
# gpt2 는 주어진 입력에 기반하여 예상되는 텍스트를 생성하는 능력 가짐
# 텍스트를 토큰으로 변환
input_ids = tokenizer.encode("Hello, my name is", return_tensors='pt')

# 모델에 토큰을 입력하고 출력을 얻음 max_length 는 생성될 텍스트의 최대 길이, sequences는 문장의 개수
output = model.generate(input_ids, max_length=100, num_return_sequences=5, do_sample=True)

# 출력을 텍스트로 변환
for i in range(5):
    # tokenizer.decode 는 사람이 이해할 수 있는 자연어로 변환
    print(tokenizer.decode(output[i], skip_special_tokens=True))
