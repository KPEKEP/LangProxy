print("Importing modules...")
import re
import yaml
import torch
import httpx
import uvicorn
import traceback
from tqdm.auto import tqdm
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse
from transformers import SeamlessM4Tv2Model, AutoProcessor
print("Imported!")

# Load configuration
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

DEVICE = config['DEVICE']
ENDPOINT = config['ENDPOINT']
TIMEOUT = httpx.Timeout(float(config['CONNECT_TIMEOUT']), read=float(config['READ_TIMEOUT']))
SRC_LANGUAGE = config['SRC_LANGUAGE']
TARGET_LANGUAGE = config['TARGET_LANGUAGE']
TRANSLATIONS_DICT = dict()
TRANSLATIONS_LIST = list()
MAX_TRANSLATIONS_CACHE = int(config['MAX_TRANSLATIONS_CACHE'])

app = FastAPI()
    
print("Loading M4T...")
processor = AutoProcessor.from_pretrained(config['MODEL_NAME'])
model = SeamlessM4Tv2Model.from_pretrained(config['MODEL_NAME']).to(DEVICE)
print("Loaded!")

def get_cached_translation(key):
    global TRANSLATIONS_DICT
    return TRANSLATIONS_DICT.get(key, None)

def cache_translation(text, translation):
    global TRANSLATIONS_DICT, TRANSLATIONS_LIST
    TRANSLATIONS_DICT[text] = translation
    TRANSLATIONS_LIST.append(text)
    TRANSLATIONS_DICT[translation] = text
    TRANSLATIONS_LIST.append(translation)
    if len(TRANSLATIONS_LIST)>MAX_TRANSLATIONS_CACHE:
        print("Was", TRANSLATIONS_LIST) 
        for key in TRANSLATIONS_LIST[:-MAX_TRANSLATIONS_CACHE]:
            del TRANSLATIONS_DICT[key]
        TRANSLATIONS_LIST = TRANSLATIONS_LIST[-MAX_TRANSLATIONS_CACHE:]
        print("Became", TRANSLATIONS_LIST)
    
def translate(text, src_language, target_language):
    cached_translation = get_cached_translation(text)
    if cached_translation is not None:
        return cached_translation
        
    input_message = processor(text = text, src_lang=src_language, return_tensors="pt").to(DEVICE)
    output_tokens = model.generate(**input_message, tgt_lang=target_language, generate_speech=False)
    translated_text_from_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    cache_translation(text, translated_text_from_text)
    return translated_text_from_text

def process_message(text, src_language, target_language):
    triple_backtick_pattern = r"(```.*?```)"
    single_backtick_pattern = r"(?<!`)`(?!`)"

    output_parts = []
    last_index = 0

    # Process only triple backticks
    for match in re.finditer(triple_backtick_pattern, text, re.DOTALL):
        start, end = match.span()

        # Replace single backticks in the text before the matched block and translate
        text_before = text[last_index:start]
        text_before = re.sub(single_backtick_pattern, r'"', text_before)
        translated_text_before = translate(text_before, src_language, target_language) + "\n"
        if translated_text_before:
            output_parts.append(translated_text_before)

        # Append the matched text without translation
        matched_text = match.group() + "\n"
        output_parts.append(matched_text)

        last_index = end

    # Replace single backticks in any remaining text after the last match and translate
    if last_index < len(text):
        text_tail = text[last_index:]
        text_tail = re.sub(single_backtick_pattern, r'"', text_tail)
        translated_text_tail = translate(text_tail, src_language, target_language)
        output_parts.append(translated_text_tail)

    return ''.join(output_parts)

@app.post("/chat/completions")
async def chat_proxy(request: Request):
    try:
        incoming_data = await request.json()
        headers = dict(request.headers)
        del headers['content-length']
        for i, message in enumerate(tqdm(incoming_data["messages"], desc="Translating request messages")):
            message_translated = process_message(message['content'], SRC_LANGUAGE, TARGET_LANGUAGE)
            incoming_data["messages"][i]['content'] = message_translated
            
        print("Forwarding the request...")
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                ENDPOINT,
                json=incoming_data,
                headers=headers
            )
    
        if response.status_code == 200:
            response_data = response.json()
            for i, message in enumerate(tqdm(response_data['choices'], desc="Translating response messages")):
                response_data['choices'][i]['message']['content'] = process_message(response_data['choices'][i]['message']['content'], TARGET_LANGUAGE, SRC_LANGUAGE)
            return JSONResponse(status_code=response.status_code, content=response_data)
        else:
            return JSONResponse(status_code=response.status_code, content=response.text)
    except Exception as e:
       error_msg = "Error while translating: " + traceback.format_exc()
       print(error_msg)
       return JSONResponse(status_code=200, content=error_msg)

if __name__ == "__main__":
    uvicorn.run(app, host=config['HOST'], port=int(config['PORT']))