import pandas as pd
import json
from openai import OpenAI
import os
import argparse
import time
from utils import clean_comment
import random

client = OpenAI(
    api_key="sk-cMXZOdxBK06opbOZfNb76mny4V5a5SdCbQ6Nxx9mpcNVAQ0E",
    base_url="https://api.chatanywhere.tech/v1"
)
# 定义不同的询问语句，直接告诉有无漏洞
q_vulnerable = 'Here is a vulnerable function source code. Please analyze the potential reasons for its vulnerable prediction using concise and clear language.  Focus on specific issues such as logical flaws, input validation gaps, or other security risks.  Do not describe it as benign. Provide the analysis in a single, coherent paragraph without using bullet points or markdown.'
q_vulnerability_free = 'Here is a function source code that is free of vulnerabilities. Please analyze the functionality of the function using concise and clear language.  Focus on its logical soundness and describe what the function does.  Do not speculate about vulnerabilities. Provide the analysis in a single, coherent paragraph without using bullet points or markdown.'
# 让chatgpt预测是否有漏洞
q_llm = ("Please determine whether there are vulnerabilities in the source code of the function below. "
     "If vulnerabilities exist, please indicate their locations and the reasons for their occurrence; "
     "if there are no vulnerabilities, please explain the function's functionality. "
     "Please strictly follow the template below: "
     "if there is a vulnerability, output: VULNERABLE-YES. Include the location and reason for the vulnerability; "
     "if there are no vulnerabilities, output: VULNERABLE-NO. Include an explanation of the function's functionality. "
     "Please adhere strictly to the template. "
     "Please use as concise language as possible. "
     "Do not use bullet points or markdown format. Provide the description in a single paragraph.")

def split_message(message, max_length):
    split_messages = []
    while message:
        if len(message) <= max_length:
            split_messages.append(message)
            break
        else:
            split_messages.append(message[:max_length])
            message = message[max_length:]
    return split_messages

def send_message(message):
    rsp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in source code vulnerability detection. You can analyze the functionality of a function. You can identify the locations of vulnerabilities within the function and summarize the reasons for their occurrence."},
            {"role": "user", "content": message}
        ]
    )
    return rsp.choices[0].message.content


def chatgpt_generate_reason_gt(processed_count, function_list, label_list, output_data, output_file_path):
    for idx in range(processed_count, len(function_list)):
        function_code = function_list[idx]

        if label_list[idx] == 1:
            q = q_vulnerable
        else:
            q = q_vulnerability_free
        message = q + '\n' + function_code
        response = send_message(message)
        reason = response  
        
        cleaned_function_code = clean_comment(function_code)
        output_data.append({
            'function': cleaned_function_code,
            'label': label_list[idx],
            'reason': reason
        })
        
        random.shuffle(output_data)
        with open(output_file_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Processed {idx + 1}/{len(function_list)} messages.", flush=True)
        time.sleep(3)

    print("Processing complete! The new JSON file has been saved.")

def chatgpt_generate_reason_llm(processed_count, function_list, label_list, output_data, output_file_path):
    for idx in range(processed_count, len(function_list)):
        function_code = function_list[idx]
        message = q_llm + '\n' + function_code
        response = send_message(message)
        reason = response

        if "VULNERABLE-YES" in reason:
            llm_label =  1 
        elif "VULNERABLE-NO" in reason:
            llm_label =  0
        else:
            llm_label =  0

        cleaned_function_code = clean_comment(function_code)
        output_data.append({
            'function': cleaned_function_code,
            'label': label_list[idx],
            'reason': reason,
            'llm_label': llm_label
        })
        random.shuffle(output_data)
        with open(output_file_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        print(f"Processed {idx + 1}/{len(function_list)} messages.", flush=True)
        time.sleep(3)

    print("Processing complete! The new JSON file has been saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    # 文件路径定义
    input_file_path = args.input_file
    output_file_path = args.output_file

    

    with open(input_file_path, 'r') as f:
        data = json.load(f)

    function_list = [item['function'] for item in data]  # 提取所有 function
    label_list = [item['label'] for item in data]  # 提取所有 label

    output_data = []
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r') as f:
            output_data = json.load(f)
        processed_count = len(output_data)
    else:
        processed_count = 0

    chatgpt_generate_reason_llm(processed_count, function_list, label_list, output_data, output_file_path)


if __name__ == '__main__':
    main()