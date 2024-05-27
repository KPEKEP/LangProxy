import re
import yaml
import httpx
import uvicorn
import traceback
import logging
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse
from tqdm.auto import tqdm
from transformers import SeamlessM4Tv2Model, AutoProcessor

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from YAML file
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# Configuration parameters
DEVICE = config['DEVICE']
ENDPOINT = config['ENDPOINT']
TIMEOUT = httpx.Timeout(
    connect=float(config['CONNECT_TIMEOUT']),
    read=float(config['READ_TIMEOUT']),
    write=float(config['WRITE_TIMEOUT']),
    pool=float(config['POOL_TIMEOUT'])
)
SRC_LANGUAGE = config['SRC_LANGUAGE']
TARGET_LANGUAGE = config['TARGET_LANGUAGE']
MAX_TRANSLATIONS_CACHE = int(config['MAX_TRANSLATIONS_CACHE'])

# Translation cache class
class TranslationCache:
    def __init__(self, max_cache_size: int):
        self.translations_dict = {}  # Dictionary to store translations
        self.translations_list = []  # List to maintain order of translations
        self.max_cache_size = max_cache_size  # Maximum cache size

    def get_cached_translation(self, key: str) -> str:
        # Retrieve a translation from the cache
        return self.translations_dict.get(key)

    def cache_translation(self, text: str, translation: str):
        # Add a new translation to the cache
        if len(self.translations_list) > self.max_cache_size:
            oldest_key = self.translations_list.pop(0)  # Remove oldest entry if cache is full
            self.translations_dict.pop(oldest_key)  # Remove from dictionary
        self.translations_dict[text] = translation  # Add new translation to dictionary
        self.translations_list.append(text)  # Add new translation to list

# Initialize the translation cache
cache = TranslationCache(MAX_TRANSLATIONS_CACHE)

# Initialize FastAPI app
app = FastAPI()

# Load model and processor
logger.info("Loading M4T model and processor...")
processor = AutoProcessor.from_pretrained(config['MODEL_NAME'])
model = SeamlessM4Tv2Model.from_pretrained(config['MODEL_NAME']).to(DEVICE)
logger.info("Model and processor loaded.")

def translate(text: str, src_language: str, target_language: str) -> str:
    """
    Translate text from source language to target language.
    """
    cached_translation = cache.get_cached_translation(text)
    if cached_translation:
        return cached_translation

    # Prepare inputs for the model
    inputs = processor(text=text, src_lang=src_language, return_tensors="pt").to(DEVICE)
    # Generate translation
    outputs = model.generate(**inputs, tgt_lang=target_language, generate_speech=False)
    translated_text = processor.decode(outputs[0][0], skip_special_tokens=True)
    # Cache the translation
    cache.cache_translation(text, translated_text)
    return translated_text

def process_message(text: str, src_language: str, target_language: str) -> str:
    """
    Process a message by translating text portions while keeping code blocks intact.
    """
    triple_backtick_pattern = r"(```.*?```)"  # Pattern for triple backtick code blocks
    single_backtick_pattern = r"(?<!`)`(?!`)"

    output_parts = []
    last_index = 0

    # Find and translate text outside of code blocks
    for match in re.finditer(triple_backtick_pattern, text, re.DOTALL):
        start, end = match.span()
        text_before = text[last_index:start].replace('`', '"')
        translated_text_before = translate(text_before, src_language, target_language)
        output_parts.append(translated_text_before + "\n")
        output_parts.append(match.group() + "\n")
        last_index = end

    # Translate any remaining text after the last code block
    if last_index < len(text):
        text_tail = text[last_index:].replace('`', '"')
        translated_text_tail = translate(text_tail, src_language, target_language)
        output_parts.append(translated_text_tail)

    return ''.join(output_parts)

@app.post("/chat/completions")
async def chat_proxy(request: Request):
    """
    Proxy endpoint for chat completions.
    """
    try:
        incoming_data = await request.json()  # Get JSON data from the request
        headers = {key: val for key, val in request.headers.items() if key.lower() != 'content-length'}
        
        # Translate request messages
        for message in tqdm(incoming_data["messages"], desc="Translating request messages"):
            message['content'] = process_message(message['content'], SRC_LANGUAGE, TARGET_LANGUAGE)

        # Send translated request to the endpoint
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(ENDPOINT, json=incoming_data, headers=headers)

        # Handle the response
        if response.status_code == 200:
            response_data = response.json()
            # Translate response messages
            for choice in tqdm(response_data['choices'], desc="Translating response messages"):
                choice['message']['content'] = process_message(
                    choice['message']['content'], TARGET_LANGUAGE, SRC_LANGUAGE
                )
            return JSONResponse(status_code=response.status_code, content=response_data)
        else:
            return JSONResponse(status_code=response.status_code, content=response.text)
    except Exception:
        error_message = "Error while processing: " + traceback.format_exc()
        logger.error(error_message, exc_info=True)
        return JSONResponse(status_code=500, content={"message": error_message})

if __name__ == "__main__":
    uvicorn.run(app, host=config['HOST'], port=int(config['PORT']))