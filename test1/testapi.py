# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

client = OpenAI(
    api_key="sk-2a21890213e9404483ae637fd55c87d3", base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False,
)

print(response.choices[0].message.content)
