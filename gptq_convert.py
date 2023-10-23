from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
## Examples of model_id
#model_id = 'beomi/kollama-7b'
model_id = 'EleutherAI/polyglot-ko-1.3b'
save_id = 'polyglot-ko-1.3b_gptq'

tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)
model.save_pretrained(save_id)
tokenizer.save_pretrained(save_id)
