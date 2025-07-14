import ast
import base64
import json
import os
import re
from io import BytesIO

import matplotlib.pyplot as plt
import openai
import requests
from PIL import Image


def extract_parsed_output(output_string):
    cleaned_string = output_string.replace("Output:", "").strip()
    pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', cleaned_string)
    result_dict = {key: value for key, value in pairs}
    return result_dict


def concept_prompt_parsing(concept_dict, concept_opt):

    prompt = concept_dict["base"]["step2_base_prompt"]

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {concept_opt.gpt_key}"}

    templatev0_1 = f"""
    You are a prompt object parser.

    Parse the following caption and extract all object phrases wrapped with tags.
    Return the result as a list of key-value pairs, where each key is the tag (e.g., <mciz>) and the value is the associated phrase.
    Make sure to include the background in the output.
    Make if there is no interaction between objects just return the object phrases.

    ---Example---

    Caption: A <mciz> cat wearing <qmto> sunglasses.
    Output: ["concept0": "A <mciz> cat wearing sunglasses.","concept1": "A <qmto> sunglasses.","bg": "A cat wearing sunglasses."]

    Caption: A <mciz> cat and a <acdi> dog.
    Output: ["concept0": "A <mciz> cat.", "concept1": "A <acdi> dog.","bg": "A cat and a dog."]

    Caption: {prompt}
    """

    payload = {
        "model": "gpt-4o-2024-05-13",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": templatev0_1,
                    },
                ],
            }
        ],
        "max_tokens": 1000,
        "seed": 0,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    content = response.json()["choices"][0]["message"]["content"]
    print("Input: ", prompt)
    print(content)
    content_dict = extract_parsed_output(content)

    # Update the concept prompts
    concept_dict["base"]["step2_bg_prompt"] = content_dict["bg"]

    for i in range(concept_opt.concept_num):
        concept_key = f"concept{i}"
        concept_dict[concept_key]["step2_concept_prompt"] = content_dict[concept_key]
