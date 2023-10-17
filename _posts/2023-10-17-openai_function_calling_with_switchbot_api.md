---
layout: post
title: "LLM기반 IoT 제어 - OpenAI 함수호출과 스위치봇 API 연결하기"
author: Taeyoung Kim
date: 2023-10-17 00:00:00
categories: chatgpt, IoT, switchbot
comments: true
---

```python
# Prompts Azure OpenAI with a request and synthesizes the response.
def ask_openai(llm_model, messages, user_message, functions = ''):

    proc_messages = messages
    
    if user_message != '':
        proc_messages.append({"role": "user", "content": user_message})
    
    if functions == '':    
        # Ask Azure OpenAI
        response = openai.ChatCompletion.create(model=llm_model, messages=proc_messages, temperature = 1.0)
    else:
        response = openai.ChatCompletion.create(model=llm_model, messages=proc_messages, functions=functions, function_call="auto")
    
    response_message = response['choices'][0]['message']
    
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors

        available_functions = {
            "get_current_temperature": get_current_temperature,
            "get_current_humidity": get_current_humidity,
            "turn_on_light": turn_on_light,
            "turn_off_light": turn_off_light
        }  # only one function in this example, but you can have multiple
        
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        
        if 'user_prompt' in function_args:
            function_response = fuction_to_call(function_args.get('user_prompt'))
        else:
            function_response = fuction_to_call()
        
        proc_messages.append(response_message)  # extend conversation with assistant's reply
        proc_messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
        second_response = openai.ChatCompletion.create(
            model=llm_model,
            messages=messages,
        )  # get a new response from GPT where it can see the function response
        
        assistant_message = second_response['choices'][0]['message']['content']
    else:
        assistant_message = response_message['content']
    
    text = assistant_message.replace('\n', ' ').replace(' .', '.').strip()
    
    print(Fore.GREEN + text + Style.RESET_ALL)
    
    proc_messages.append({"role": "assistant", "content": assistant_message})

    return proc_messages, text
```


