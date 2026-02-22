import os
import json
import requests
import time
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from openai import OpenAI
import google.generativeai as genai
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_KEY_1 = os.getenv("OPENROUTER_KEY_1")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROKCLOUD_API_KEY = os.getenv("GROKCLOUD_API_KEY")

logger.debug(f"GEMINI_API_KEY: {'set' if GEMINI_API_KEY else 'not set'}")
logger.debug(f"OPENROUTER_KEY_1: {'set' if OPENROUTER_KEY_1 else 'not set'}")
logger.debug(f"COHERE_API_KEY: {'set' if COHERE_API_KEY else 'not set'}")
logger.debug(f"GROKCLOUD_API_KEY: {'set' if GROKCLOUD_API_KEY else 'not set'}")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.error("GEMINI_API_KEY is not set in .env file")

response_cache = {}

def api_call(config, prompt, retries=3, timeout=10):
    error_details = ""
    for attempt in range(retries):
        try:
            if config["type"] == "openrouter":
                client = OpenAI(api_key=OPENROUTER_KEY_1, base_url="https://openrouter.ai/api/v1")
                models = [config["model"], "mistralai/mixtral-8x7b-instruct"]  
                for model in models:
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            timeout=timeout
                        )
                        if not response.choices or len(response.choices) == 0:
                            raise ValueError("Empty choices in OpenRouter response")
                        text = response.choices[0].message.content
                        words = text.split()
                        if len(words) > 50:
                            words = words[:50]
                            text = " ".join(words) + "..." 
                        else:
                            text = " ".join(words)
                        time.sleep(2)  
                        return text
                    except Exception as e:
                        error_details = f"Model {model} failed: {str(e)}"
                        logger.error(f"OpenRouter model {model} attempt {attempt + 1} failed: {str(e)}")
                        continue
                raise ValueError(f"All OpenRouter models failed: {error_details}")
            elif config["type"] == "grokcloud":
                headers = {
                    "Authorization": f"Bearer {GROKCLOUD_API_KEY}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "llama3-70b-8192",
                    "messages": [{"role": "user", "content": prompt}]
                }
                resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers, timeout=timeout)
                resp.raise_for_status()
                response = resp.json()
                if not response.get("choices") or len(response["choices"]) == 0:
                    raise ValueError("Empty choices in GrokCloud response")
                text = response["choices"][0]["message"]["content"]
                words = text.split()
                if len(words) > 50:
                    words = words[:50]
                    text = " ".join(words) + "..." 
                else:
                    text = " ".join(words)
                return text
            elif config["type"] == "cohere":
                headers = {
                    "Authorization": f"Bearer {COHERE_API_KEY}",
                    "Content-Type": "application/json"
                }
                data = {
                    "message": prompt,
                    "model": "command-r-plus",
                    "temperature": 0.5
                }
                resp = requests.post("https://api.cohere.ai/v1/chat", json=data, headers=headers, timeout=timeout)
                resp.raise_for_status()
                response = resp.json()
                if not response.get("text"):
                    raise ValueError("Empty text in Cohere response")
                text = response["text"]
                words = text.split()
                if len(words) > 50:
                    words = words[:50]
                    text = " ".join(words) + "..."  
                else:
                    text = " ".join(words)
                return text
            elif config["type"] == "gemini":
                if not GEMINI_API_KEY:
                    raise ValueError("GEMINI_API_KEY is not set")
                model_instance = genai.GenerativeModel(config["model"])
                response = model_instance.generate_content(prompt)
                if not response.text:
                    raise ValueError("Empty text in Gemini response")
                text = response.text
                words = text.split()
                if len(words) > 50:
                    words = words[:50]
                    text = " ".join(words) + "..."  
                else:
                    text = " ".join(words)
                return text
        except Exception as e:
            error_details = f"{config['type'].capitalize()} API failed - {str(e)}. Response: {resp.text if 'resp' in locals() else 'No response'}"
            logger.error(f"{config['type'].capitalize()} attempt {attempt + 1} failed: {str(e)}")
            if attempt == retries - 1:
                error_details_words = error_details.split()[:50]
                return f"Error: {' '.join(error_details_words)}..."
            time.sleep(2 ** attempt)  
    error_details_words = error_details.split()[:50]
    return f"Error: {config['type'].capitalize()} API failed after {retries} retries - {' '.join(error_details_words)}..."

class QueryClassifier:
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template=(
                "Is this query debatable (requiring discussion or multiple perspectives) or non-debatable (having a clear, factual answer)? "
                "Explain your reasoning and return 'Debatable' or 'Non-debatable' as the final word.\nQuery: {query}"
            )
        )
        self.non_debatable_keywords = ["what is", "define", "calculate", "who is", "when is", "where is"]

    def is_likely_non_debatable(self, query):
        return any(keyword in query.lower() for keyword in self.non_debatable_keywords)

    def classify(self, query):
        cache_key = f"classify:{query}"
        if cache_key in response_cache:
            return response_cache[cache_key]

        if self.is_likely_non_debatable(query):
            result = {
                "classification": "Non-debatable",
                "reasoning": f"Query contains non-debatable keywords: {query}. Assumed non-debatable without API call."
            }
            response_cache[cache_key] = result
            return result

        prompt = self.prompt_template.format(query=query)
        response = api_call({"type": "grokcloud"}, prompt)
        classification = "Debatable"
        reasoning = response
        if "Error:" in response:
            reasoning = f"Classification failed: {response}. Defaulting to Debatable."
        else:
            lines = response.strip().split('\n')
            first_line = lines[0].strip()
            if first_line in ["Debatable", "Non-debatable"]:
                classification = first_line
            else:
                response_lower = response.lower()
                if "non-debatable" in response_lower:
                    classification = "Non-debatable"
                elif "debatable" in response_lower:
                    classification = "Debatable"
        result = {
            "classification": classification,
            "reasoning": reasoning
        }
        response_cache[cache_key] = result
        return result

class ConvergenceChecker:
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["critiques", "round", "query"],
            template=(
                "Evaluate these critiques for the query '{query}' in round {round}:\n{critiques}\n"
                "Do they indicate convergence (e.g., only minor or stylistic suggestions, repetitive issues, or no significant flaws) "
                "or require further refinement (e.g., major factual errors, missing key arguments)? "
                "Explain your reasoning and return 'Converged' or 'Continue' as the final word."
            )
        )

    def check_convergence(self, critiques, round, query):
        prompt = self.prompt_template.format(critiques=critiques, round=round, query=query)
        response = api_call({"type": "grokcloud"}, prompt)
        decision = "Continue"
        reasoning = response
        if "Error:" in response:
            reasoning = f"Convergence check failed: {response}. Defaulting to Continue."
        else:
            lines = response.strip().split('\n')
            last_line = lines[-1].strip() if lines else ""  
            if last_line in ["Converged", "Continue"]:
                decision = last_line
            else:
                response_lower = response.lower()
                if "converged" in response_lower:
                    decision = "Converged"
                elif "continue" in response_lower:
                    decision = "Continue"
        return {
            "decision": decision,
            "reasoning": reasoning
        }

class Generator:
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template="Provide a detailed and accurate answer to: {query}"
        )

    def generate_response(self, query):
        cache_key = f"generate:{query}"
        if cache_key in response_cache:
            return response_cache[cache_key]

        prompt = self.prompt_template.format(query=query)
        gemini_response = api_call({"type": "gemini", "model": "gemini-2.0-flash"}, prompt)
        openrouter_response = api_call({"type": "openrouter", "model": "meta-llama/llama-3.1-8b-instruct:free"}, prompt)
        if "Error:" in gemini_response and "Error:" in openrouter_response:
            result = {
                "gemini_response": "Error: Gemini API failed",
                "openrouter_response": "Error: OpenRouter API failed",
                "combined": "Unable to generate response due to API errors. Please try again."
            }
        else:
            result = {
                "gemini_response": gemini_response,
                "openrouter_response": openrouter_response,
                "combined": f"Gemini: {gemini_response}\nOpenRouter (LLaMA): {openrouter_response}"
            }
        response_cache[cache_key] = result
        return result

class Critic:
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["response", "query"],
            template=(
                "Critique this response to '{query}': {response}\n"
                "Identify specific flaws (e.g., factual errors, missing arguments, verbosity) and suggest actionable improvements. "
                "Provide a concise critique."
            )
        )

    def critique_response(self, response, query):
        prompt = self.prompt_template.format(response=response, query=query)
        critique = api_call({"type": "grokcloud"}, prompt)
        return {
            "grokcloud_critique": critique,
            "combined": f"GrokCloud Critique: {critique}"
        }

class Validator:
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["response", "query"],
            template="Validate this response to '{query}': {response}\nIs it accurate, coherent, and complete? Suggest final improvements."
        )

    def validate_response(self, response, query):
        prompt = self.prompt_template.format(response=response, query=query)
        openrouter_validation = api_call({"type": "openrouter", "model": "meta-llama/llama-3.1-8b-instruct:free"}, prompt)
        cohere_validation = api_call({"type": "cohere"}, prompt)

        if "Error:" in openrouter_validation:
            grokcloud_validation = api_call({"type": "grokcloud"}, prompt)
            openrouter_validation = grokcloud_validation if not "Error:" in grokcloud_validation else "Error: Fallback to GrokCloud failed"
        
        return {
            "openrouter_validation": openrouter_validation,
            "cohere_validation": cohere_validation,
            "combined": f"OpenRouter (LLaMA) Validation: {openrouter_validation}\nCohere Validation: {cohere_validation}"
        }

class Orchestrator:
    def __init__(self):
        self.generator = Generator()
        self.critic = Critic()
        self.validator = Validator()
        self.classifier = QueryClassifier()
        self.convergence_checker = ConvergenceChecker()
        self.max_rounds = 5

    def synthesize_response(self, current_response, critique, query):
        prompt = (
            f"For the query '{query}', synthesize the following response and critique into a concise, improved response:\n"
            f"Current Response: {current_response}\n"
            f"Critique: {critique}\n"
            f"Address the critique's suggestions clearly and avoid verbosity."
        )
        return api_call({"type": "grokcloud"}, prompt)

    def run_debate(self, query):
        reasoning_log = []

        # Step 0: Classify query
        classification_output = self.classifier.classify(query)
        reasoning_log.append({
            "step": "Query Classification",
            "classification": classification_output["classification"],
            "reasoning": classification_output["reasoning"]
        })

        if classification_output["classification"] == "Non-debatable":
            prompt = f"Provide a clear and concise answer to: {query}"
            cache_key = f"non_debatable:{query}"
            if cache_key in response_cache:
                answer = response_cache[cache_key]  
            else:
                answer = api_call({"type": "grokcloud"}, prompt)
                response_cache[cache_key] = answer
            reasoning_log.append({
                "step": "Straightforward Answer",
                "answer": answer
            })
            final_response = (
                f"Final Response: {answer}\n"
                f"Classification Reasoning: {classification_output['reasoning']}"
            )
            return {
                "final_response": final_response,
                "reasoning_log": reasoning_log
            }

        gen_output = self.generator.generate_response(query)
        current_response = gen_output["combined"]
        reasoning_log.append({"step": "Initial Generation", "response": current_response})

        for round in range(self.max_rounds):
            critique_output = self.critic.critique_response(current_response, query)
            reasoning_log.append({
                "step": f"Critique Round {round + 1}",
                "critique": critique_output["combined"]
            })

            convergence_output = self.convergence_checker.check_convergence(
                critiques=critique_output["combined"],
                round=round + 1,
                query=query
            )
            reasoning_log.append({
                "step": f"Convergence Check Round {round + 1}",
                "decision": convergence_output["decision"],
                "reasoning": convergence_output["reasoning"]
            })

            if convergence_output["decision"] == "Converged":
                break

            current_response = self.synthesize_response(
                current_response, critique_output["combined"], query
            )
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
        grokcloud_response = api_call({"type": "grokcloud"}, grokcloud_prompt)
        reasoning_log.append({
            "step": "GrokCloud Selection",
            "grokcloud_response": grokcloud_response.strip() 
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
            f"Final Response: {chosen_validation}\n"
            f"GrokCloud Reasoning: {grokcloud_reasoning}"
        )

        return {
            "final_response": final_response,
            "reasoning_log": reasoning_log
        }

if __name__ == "__main__":
    orchestrator = Orchestrator()
    
    while True:
        query = input("Enter your debate query (or 'quit' to exit): ").strip()
        if query.lower() == 'quit':
            print("Exiting...")
            break
        if not query:
            query = "What is the best way to learn Python?"
            print(f"Using default query: {query}")

        print(f"\nRunning debate for query: '{query}'...")
        result = orchestrator.run_debate(query)
        
        print("\nFinal Response:", result["final_response"])
        print("\nReasoning Log:")
        for log in result["reasoning_log"]:
            print(json.dumps(log, indent=2))
        print("\n" + "="*50 + "\n")