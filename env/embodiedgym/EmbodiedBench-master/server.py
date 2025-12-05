from flask import Flask, request, jsonify
import os
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig, pipeline, Gemma3ForConditionalGeneration
import torch
from PIL import Image

max_token = 1024
# model_path = "microsoft/Phi-4-multimodal-instruct"
# model_path = 'AIDC-AI/Ovis2-16B'
# model_path = 'AIDC-AI/Ovis2-34B'
model_path = 'google/gemma-3-12b-it'

# Load the custom model
class CustomModel:
    def __init__(self, model_path, language_only):
        self.model_path = model_path
        self.language_only = language_only
        self.model_type = 'custom'

        if 'Ovis' in model_path:
            self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=20000,
                                             trust_remote_code=True,
                                             attn_implementation="eager",
                                             device_map='auto')
            self.text_tokenizer = self.model.get_text_tokenizer()
            self.visual_tokenizer = self.model.get_visual_tokenizer()
        elif 'Phi-4' in model_path:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True, 
                device_map='auto',
                attn_implementation="flash_attention_2"
            )
            self.generation_config = GenerationConfig.from_pretrained(model_path)
        elif 'gemma' in model_path:
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_path, device_map="auto", torch_dtype=torch.bfloat16,
                attn_implementation="eager"
            )
            self.processor = AutoProcessor.from_pretrained(model_path)


    def respond(self, prompt, image_path=None):
        if 'microsoft/Phi-4' in self.model_path:
            user_prompt = '<|user|>'
            assistant_prompt = '<|assistant|>'
            prompt_suffix = '<|end|>'
            formatted_prompt = f'{user_prompt}<|image_1|>{prompt}{prompt_suffix}{assistant_prompt}'
            
            image = Image.open(image_path)
            inputs = self.processor(text=formatted_prompt, images=image, return_tensors='pt').to(self.model.device)
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_token,  # Adjust as needed
                    temperature=0.0,      # Adjust as needed
                    generation_config=self.generation_config,
                )
        
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = self.processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        elif 'Ovis' in self.model_path:
            images = [Image.open(image_path)]
            max_partition = 9
            query = f'<image>\n{prompt}'
            prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, images, max_partition=max_partition)
            attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
            attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)
            pixel_values = [pixel_values]
            # generate output
            with torch.inference_mode():
                gen_kwargs = dict(
                    max_new_tokens=max_token,
                    do_sample=False,
                    temperature=0.0,
                    repetition_penalty=None,
                    eos_token_id=self.model.generation_config.eos_token_id,
                    pad_token_id=self.text_tokenizer.pad_token_id,
                    use_cache=True
                )
                output_ids = self.model.generate(input_ids,  pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
                response = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
        else:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image_path},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            inputs = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                            return_dict=True, return_tensors="pt"
                        ).to(self.model.device)

            input_len = inputs["input_ids"].shape[-1]
            print(input_len)
            with torch.inference_mode():
                generation = self.model.generate(**inputs, max_new_tokens=max_token, do_sample=False, temperature=0.0, use_cache=True)
                generation = generation[0][input_len:]

            response = self.processor.decode(generation, skip_special_tokens=True)
        return response

# Initialize Flask app and model
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = CustomModel(model_path=model_path, language_only=False)

@app.route('/process', methods=['POST'])
def process_request():
    if 'image' not in request.files or 'sentence' not in request.form:
        return jsonify({'error': 'Missing image or sentence'}), 400

    image = request.files['image']
    sentence = request.form['sentence']

    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the image temporarily
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    # Generate response from the model
    model_response = model.respond(sentence, image_path=image_path)

    return jsonify({'response': model_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=23333)
