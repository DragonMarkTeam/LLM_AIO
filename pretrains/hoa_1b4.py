from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("vlsp-2023-vllm/hoa-1b4")
model = AutoModelForCausalLM.from_pretrained("vlsp-2023-vllm/hoa-1b4", low_cpu_mem_usage=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model.to(device)

prompt = "Địa chỉ trường Đại học Tôn Đức Thắng nằm ở số"
input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)

gen_tokens = model.generate(input_ids, max_length=max_length, repetition_penalty=1.1)

print(tokenizer.batch_decode(gen_tokens)[0])