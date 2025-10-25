import requests, time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b-instruct"   # switched to Qwen

def ollama_generate(
    prompt,
    model: str = MODEL,
    num_predict: int = 100,       # qwen is fine with 50–80; 60 is a good default
    temperature: float = 0.45,   # slightly lower = more focused, less fluff
    retries: int = 2
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "1h",      # keep model warm in RAM
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
            "top_p": 0.9,
            # stop sequences help prevent lists/prefaces
            "stop": ["\n\n", "\n1.", "\n- ", "As a helpful AI assistant", "Based on the customer feedback"]
        }
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["response"].strip()
        except Exception as e:
            if attempt < retries:
                time.sleep(1.2 * (attempt + 1))
                continue
            return f"[GENERATION_ERROR] {e}"
        

# The ollama_utils.py script acts as a support module for the main gen_strategies_batched_robust.py 
# script. 
# It provides a function called ollama_generate() that handles all the communication between your Python code and the local Ollama model (in this case, Qwen 2.5:3B Instruct). 
# Whenever the strategy generator script needs to create a new customer retention strategy, it calls this function instead of directly managing the API request. 
# The function sends a properly formatted prompt to the Ollama API running on your computer, waits for the model’s text output, and returns the generated response. 
# It also includes retry logic to handle connection errors or timeouts automatically. 
# In simple terms, ollama_utils.py works as a bridge that connects the main strategy generation code to the AI model, ensuring smooth and reliable interaction between them.
