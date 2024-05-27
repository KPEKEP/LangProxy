# Translation API Server

This project implements a FastAPI server that translates messages using the SeamlessM4Tv2 model from the Hugging Face Transformers library. It is designed to handle both requests and responses, translating incoming message content and translating output messages back into the source language.

## Features

- Translation of text between source and target languages using a pre-trained language model.
- Caching of translations to improve response times for repeated requests.
- Asynchronous handling of HTTP requests

## Requirements

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- FastAPI
- Uvicorn for running the server
- HTTPX for handling HTTP requests

A complete list of Python package dependencies can be found in `requirements.txt`.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/KPEKEP/LangProxy.git
   cd your-repository-directory
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `config.yaml` file in the root directory with the following structure:
   ```yaml
	HOST: "0.0.0.0"
	PORT: 8000
	MODEL_NAME: "facebook/seamless-m4t-v2-large"
	DEVICE: "cuda"
	CONNECT_TIMEOUT: 10000
	READ_TIMEOUT: 30000
   	WRITE_TIMEOUT: 30000
   	POOL_TIMEOUT: 30000
	ENDPOINT: "http://localhost:4321/v1/chat/completions"
	SRC_LANGUAGE : "rus"
	TARGET_LANGUAGE : "eng"
	MAX_TRANSLATIONS_CACHE: 262144
   ```

## Usage

To start the server, run:

```bash
python3 LangProxy.py
```

This will start the FastAPI server on the host and port specified in the `config.yaml` file. You can now send POST requests to the `/chat/completions` endpoint to get translated messages.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit pull requests with any enhancements or bug fixes.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.
