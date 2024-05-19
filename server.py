import requests
import boto3
import json
import os
import time
from flask import Flask, send_file, render_template, request


# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
)

# Call Mistral model
def call_mistral_8x7b(prompt):
    prompt_config = {
        "prompt": prompt,
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.8,
    }

    body = json.dumps(prompt_config)

    modelId = "mistral.mixtral-8x7b-instruct-v0:1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("outputs")[0].get("text")
    return results

# Call Mistral model
def call_mistral_7b(prompt):
    prompt_config = {
        "prompt": prompt,
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.8,
    }

    body = json.dumps(prompt_config)

    modelId = "mistral.mistral-7b-instruct-v0:2"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("outputs")[0].get("text")
    return results

# Call Claude model
def call_claude_haiku(prompt):

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-haiku-20240307-v1:0"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("content")[0].get("text")
    return results

def claude_prompt_format(prompt: str) -> str:
    # Add headers to start and end of prompt
    return "\n\nHuman: " + prompt + "\n\nAssistant:"

# Call Claude model
def call_claude(prompt):
    prompt_config = {
        "prompt": claude_prompt_format(prompt),
        "max_tokens_to_sample": 4096,
        "temperature": 0.5,
        "top_k": 250,
        "top_p": 0.5,
        "stop_sequences": [],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-instant-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("completion")
    return results

# Call Titan model
def call_titan(prompt):
    prompt_config = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 4096,
            "stopSequences": [],
            "temperature": 0,
            "topP": 1,
        },
    }

    body = json.dumps(prompt_config)

    modelId = "amazon.titan-text-express-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("results")[0].get("outputText")
    return results


def call_llama2(prompt):
    prompt_config = {
        "prompt": prompt,
        "max_gen_len": 2048,
        "top_p": 0.9,
        "temperature": 0.2,
    }

    body = json.dumps(prompt_config)

    modelId = "meta.llama2-13b-chat-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body["generation"].strip()
    return results

from flask import Flask, render_template, send_file

app = Flask(__name__, static_url_path='/images', static_folder='images')
if __name__ == '__main__':
    app.run(debug=True)
    
@app.route('/')
def hello_world():
    return render_template('main.html', html_file='lyrics.html')

@app.route('/exam')
def exam():
    return render_template('main.html', html_file='exam.html')

@app.route('/prompt', methods=['post'])
def prompt():
    prompt = request.data.decode()
    result = call_claude(prompt)
    return result

@app.route('/style.css')
def style():
    return send_file('style.css')

@app.route('/tailwind.css')
def taiwind():
    return send_file('tailwind.css')