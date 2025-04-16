from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_t5_model():
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    return model, tokenizer

def generate_questions(context, num_questions=3):
    model, tokenizer = load_t5_model()
    input_text = f"generate questions: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)

    outputs = model.generate(
        input_ids=input_ids,
        max_length=64,
        num_return_sequences=num_questions,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )

    return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]