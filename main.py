import sys
import dataclasses
import openai
import os
import re
import yaml

from pydantic import BaseModel, ValidationError
from typing import Optional


@dataclasses.dataclass
class Message:
  role: str
  content: str

def llm_call(model: str, messages: list[Message]):
  client = openai.OpenAI(api_key=os.environ["TOGETHER_API_KEY"], base_url="https://api.together.xyz/v1")
  res = client.chat.completions.create(model=model, messages=[dataclasses.asdict(x) for x in messages], temperature=0.8, max_tokens=4096)
  return res.choices[0].message.content

def main():
    question = " ".join(sys.argv[1:])
    
    model = "NousResearch/Hermes-3-Llama-3.1-405B-Turbo"
    system_prompt = "You are a highly intelligent and concise Q&A system. Only provide clear, direct answers without any additional commentary or fluff."
    
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=question)
    ]
    
    response = llm_call(model, messages)
    print(response)

if __name__ == '__main__':
    main()
