import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI
import tiktoken

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set it in the .env file.")
client = OpenAI(api_key=api_key)

# Initialize the GPT-4 tokenizer
enc = tiktoken.encoding_for_model("gpt-4")

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

def truncate_text(text, max_tokens=3000):
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        truncated_text = enc.decode(truncated_tokens)
        return truncated_text
    return text

def summarize_text(text):
    instructions = (
        "Based on the following text, generate a summary paragraph that includes the following information about the individual: "
        "1. Name "
        "2. Institution "
        "3. Main research interests and areas of expertise "
        "4. Notable publications or achievements "
        "5. Any other relevant details that highlight their professional profile and contributions. "
        "The summary should be concise and informative, making it easy to understand the individual's primary focus and suitability for collaboration or mentorship. "
        "You do not need to list entire publication names just key terms and ideas to inform the summary."
    )
    
    # Truncate the input text to 3000 tokens
    truncated_text = truncate_text(text, max_tokens=3000)
    
    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant skilled at summarizing and extracting key information from text.",
    }
    user_message = {"role": "user", "content": f"{instructions}\n{truncated_text}"}
    messages = [system_prompt, user_message]
    
    response = completion_with_backoff(
        model="gpt-4", messages=messages, temperature=1.0
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Add any test code here if needed
    pass
