import os
import json
import requests
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from openai import OpenAI
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_KEY_1 = os.getenv("OPENROUTER_KEY_1")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROKCLOUD_API_KEY = os.getenv("GROKCLOUD_API_KEY")

openrouter_client = OpenAI(
    api_key=OPENROUTER_KEY_1,
    base_url="https://openrouter.ai/api/v1"
)

genai.configure(api_key=GEMINI_API_KEY)

def query_openrouter(model_name, prompt):
    try:
        response = openrouter_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: OpenRouter API failed - {str(e)}"

def query_grokcloud(prompt):
    try:
        headers = {
            "Authorization": f"Bearer {GROKCLOUD_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": prompt}]
        }
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: GrokCloud API failed - {str(e)}"

def query_cohere(prompt):
    try:
        headers = {
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "message": prompt,
            "model": "command-r-plus",
            "temperature": 0.5
        }
        resp = requests.post("https://api.cohere.ai/v1/chat", json=data, headers=headers)
        resp.raise_for_status()
        return resp.json()["text"]
    except Exception as e:
        return f"Error: Cohere API failed - {str(e)}"

def query_gemini(prompt, model="gemini-2.0-flash"):
    try:
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: Gemini API failed - {str(e)}"

class Generator:
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template="Provide a detailed and accurate answer to: {query}"
        )

    def generate_response(self, query):
        prompt = self.prompt_template.format(query=query)
        gemini_response = query_gemini(prompt)
        openrouter_response = query_openrouter("meta-llama/llama-3.1-8b-instruct:free", prompt)
        return {
            "gemini_response": gemini_response,
            "openrouter_response": openrouter_response,
            "combined": f"Gemini: {gemini_response}\nOpenRouter (LLaMA): {openrouter_response}"
        }

class Critic:
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["response", "query"],
            template="Critique this response to '{query}': {response}\nIdentify specific flaws and suggest actionable improvements."
        )

    def critique_response(self, response, query):
        prompt = self.prompt_template.format(response=response, query=query)
        gemini_critique = query_gemini(prompt)
        cohere_critique = query_cohere(prompt)
        return {
            "gemini_critique": gemini_critique,
            "cohere_critique": cohere_critique,
            "combined": f"Gemini Critique: {gemini_critique}\nCohere Critique: {cohere_critique}"
        }

class Validator:
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["response", "query"],
            template="Validate this response to '{query}': {response}\nIs it accurate, coherent, and complete? Suggest final improvements."
        )

    def validate_response(self, response, query):
        prompt = self.prompt_template.format(response=response, query=query)
        openrouter_validation = query_openrouter("meta-llama/llama-3.1-8b-instruct:free", prompt)
        cohere_validation = query_cohere(prompt)
        return {
            "openrouter_validation": openrouter_validation,
            "cohere_validation": cohere_validation,
            "combined": f"OpenRouter (LLaMA) Validation: {openrouter_validation}\nCohere Validation: {cohere_validation}"
        }

class Orchestrator:
    def __init__(self, default_debate_rounds=2):
        self.generator = Generator()
        self.critic = Critic()
        self.validator = Validator()
        self.default_debate_rounds = max(1, int(default_debate_rounds)) 

    def run_debate(self, query, debate_rounds=None):
        rounds = max(1, int(debate_rounds)) if debate_rounds is not None else self.default_debate_rounds
        reasoning_log = []
        
        gen_output = self.generator.generate_response(query)
        current_response = gen_output["combined"]
        reasoning_log.append({"step": "Initial Generation", "response": current_response})

        for round in range(rounds):
            critique_output = self.critic.critique_response(current_response, query)
            reasoning_log.append({
                "step": f"Critique Round {round + 1}",
                "critique": critique_output["combined"]
            })
            current_response = f"Refined: {current_response}\nCritique: {critique_output['combined']}"
            reasoning_log.append({
                "step": f"Refined Response Round {round + 1}",
                "response": current_response
            })

        validation_output = self.validator.validate_response(current_response, query)
        reasoning_log.append({
            "step": "Final Validation",
            "validation": validation_output["combined"]
        })

        grokcloud_prompt = (
            f"Evaluate the following two validations for the query '{query}' and select the better one based on accuracy, coherence, and relevance. "
            f"Explain your reasoning and clearly state which validation is chosen.\n\n"
            f"Validation 1 (OpenRouter LLaMA): {validation_output['openrouter_validation']}\n\n"
            f"Validation 2 (Cohere): {validation_output['cohere_validation']}"
        )
        grokcloud_response = query_grokcloud(grokcloud_prompt)
        reasoning_log.append({
            "step": "GrokCloud Selection",
            "grokcloud_response": grokcloud_response
        })

        chosen_validation = validation_output["cohere_validation"]
        grokcloud_reasoning = grokcloud_response
        if "Error:" in grokcloud_response:
            grokcloud_reasoning = f"GrokCloud failed: {grokcloud_response}. Defaulting to Cohere validation."
        else:
            if "Validation 1" in grokcloud_response or "OpenRouter" in grokcloud_response:
                chosen_validation = validation_output["openrouter_validation"]
            elif "Validation 2" in grokcloud_response or "Cohere" in grokcloud_response:
                chosen_validation = validation_output["cohere_validation"]

        final_response = (
            f"Final Response: {current_response}\n"
            f"Chosen Validation: {chosen_validation}\n"
            f"GrokCloud Reasoning: {grokcloud_reasoning}"
        )

        return {
            "final_response": final_response,
            "reasoning_log": reasoning_log
        }

if __name__ == "__main__":
    orchestrator = Orchestrator(default_debate_rounds=2)
    
    while True:
        query = input("Enter your debate query (or 'quit' to exit): ").strip()
        if query.lower() == 'quit':
            print("Exiting...")
            break
        if not query:
            query = "What is the best way to learn Python?"
            print(f"Using default query: {query}")

        rounds_input = input("Enter number of debate rounds (default is 2): ").strip()
        try:
            debate_rounds = int(rounds_input)
            if debate_rounds < 1:
                raise ValueError("Rounds must be at least 1")
        except (ValueError, TypeError):
            debate_rounds = 2
            print(f"Invalid input, using default {debate_rounds} rounds")

        print(f"\nRunning debate for query: '{query}' with {debate_rounds} rounds...")
        result = orchestrator.run_debate(query, debate_rounds=debate_rounds)
        
        print("\nFinal Response:", result["final_response"])
        print("\nReasoning Log:")
        for log in result["reasoning_log"]:
            print(json.dumps(log, indent=2))
        print("\n" + "="*50 + "\n")