import os
import json
import requests
import time
import uuid
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()
OPENROUTER_KEY_1 = os.getenv("OPENROUTER_KEY_1")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([OPENROUTER_KEY_1, COHERE_API_KEY, GEMINI_API_KEY]):
    raise ValueError("Missing one or more API keys in .env file")

class MemoryManager:
    def __init__(self, clear_memory=False, max_memories=1000):
        self.max_memories = max_memories
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
        self.collection_name = "neurodialectic_memory"
        
        try:
            self.client = QdrantClient(":memory:")
            self._initialize_collection(clear_memory)
        except Exception as e:
            print(f"Warning: Failed to initialize Qdrant client: {e}")
            raise ValueError("Qdrant initialization failed")

    def _initialize_collection(self, clear_memory):
        try:
            collections = self.client.get_collections()
            if self.collection_name in [c.name for c in collections.collections]:
                if clear_memory:
                    self.client.delete_collection(self.collection_name)
                    print(f"Cleared existing collection: {self.collection_name}")
                else:
                    return
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dim,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created new collection: {self.collection_name}")
        except Exception as e:
            print(f"Warning: Failed to initialize collection: {e}")
            raise

    def store_semantic(self, fact, source, tags):
        try:
            embedding = self.embedding_model.encode(fact).tolist()
            point_id = str(uuid.uuid4())
            payload = {
                "type": "semantic",
                "content": fact,
                "source": source,
                "tags": tags,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "relevance_score": 1.0
            }
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            self._prune_memories()
        except Exception as e:
            print(f"Warning: Failed to store semantic memory: {e}")

    def store_episodic(self, query, response, tags):
        try:
            content = f"Query: {query}\nResponse: {response}"
            embedding = self.embedding_model.encode(content).tolist()
            point_id = str(uuid.uuid4())
            payload = {
                "type": "episodic",
                "content": content,
                "tags": tags,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "relevance_score": 1.0
            }
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            self._prune_memories()
        except Exception as e:
            print(f"Warning: Failed to store episodic memory: {e}")

    def store_agent_communication(self, query, communication, agent, tags):
        try:
            embedding = self.embedding_model.encode(communication).tolist()
            point_id = str(uuid.uuid4())
            payload = {
                "type": "agent_communication",
                "content": communication,
                "agent": agent,
                "tags": tags,
                "query": query,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "relevance_score": 1.0
            }
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            self._prune_memories()
        except Exception as e:
            print(f"Warning: Failed to store agent communication: {e}")

    def retrieve_context(self, query, k=3):
        context = {"semantic": [], "episodic": [], "agent_communication": []}
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=k,
                score_threshold=0.5,
                with_payload=True,
                with_vectors=False
            ).points
            
            for point in search_result:
                payload = point.payload
                mem_type = payload.get("type")
                similarity = point.score
                
                if mem_type == "semantic":
                    context["semantic"].append({
                        "fact": payload["content"],
                        "source": payload.get("source", ""),
                        "tags": payload.get("tags", []),
                        "similarity": similarity
                    })
                elif mem_type == "episodic":
                    context["episodic"].append({
                        "content": payload["content"],
                        "timestamp": payload.get("timestamp", ""),
                        "tags": payload.get("tags", []),
                        "similarity": similarity
                    })
                elif mem_type == "agent_communication":
                    context["agent_communication"].append({
                        "content": payload["content"],
                        "agent": payload.get("agent", ""),
                        "tags": payload.get("tags", []),
                        "query": payload.get("query", ""),
                        "similarity": similarity
                    })
        except Exception as e:
            print(f"Warning: Failed to retrieve context: {e}")
        
        return context

    def _prune_memories(self):
        try:
            count = self.client.count(collection_name=self.collection_name).count
            if count <= self.max_memories:
                return
            
            points = []
            offset = None
            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                points.extend(scroll_result.points)
                offset = scroll_result.next_page_offset
                if offset is None:
                    break
            
            sorted_points = sorted(
                points,
                key=lambda x: (
                    x.payload.get("relevance_score", 0.0),
                    x.payload.get("timestamp", "")
                )
            )
            
            points_to_delete = sorted_points[:count - self.max_memories]
            point_ids = [point.id for point in points_to_delete]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids)
            )
            print(f"Pruned {len(point_ids)} memories to maintain {self.max_memories} entries.")
        except Exception as e:
            print(f"Warning: Failed to prune memories: {e}")

response_cache = {}

def api_call(config, prompt, retries=3, timeout=10):
    error_details = []
    for attempt in range(retries):
        try:
            if config["type"] == "openrouter":
                client = OpenAI(api_key=OPENROUTER_KEY_1, base_url="https://openrouter.ai/api/v1")
                models = ["meta-llama/llama-3.3-8b-instruct", "deepseek/deepseek-r1"]
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
                        words = text.split()[:250]
                        time.sleep(2)
                        return " ".join(words)
                    except Exception as e:
                        error_details.append(f"Model {model} failed: {str(e)}")
                        continue
                raise ValueError(f"All OpenRouter models failed: {', '.join(error_details)}")
            elif config["type"] == "gemini":
                headers = {"Content-Type": "application/json"}
                data = {"contents": [{"parts": [{"text": prompt}]}]}
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
                resp = requests.post(url, json=data, headers=headers, timeout=timeout)
                resp.raise_for_status()
                response = resp.json()
                if not response.get("candidates") or not response["candidates"][0].get("content"):
                    raise ValueError("Empty content in Gemini response")
                text = response["candidates"][0]["content"]["parts"][0]["text"]
                words = text.split()[:250]
                return " ".join(words)
            elif config["type"] == "cohere":
                headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
                data = {"message": prompt, "model": "command-r-plus", "temperature": 0.5}
                resp = requests.post("https://api.cohere.ai/v1/chat", json=data, headers=headers, timeout=timeout)
                resp.raise_for_status()
                response = resp.json()
                if not response.get("text"):
                    raise ValueError("Empty text in Cohere response")
                text = response["text"]
                words = text.split()[:250]
                return " ".join(words)
        except Exception as e:
            error_details.append(f"{config['type'].capitalize()} API failed - {str(e)}")
            if attempt == retries - 1:
                return f"Error: {', '.join(error_details)}"[:250]
            time.sleep(2 ** attempt)
    return f"Error: {', '.join(error_details)}"[:250]

class QueryClassifier:
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template=(
                "[Plan] Analyze the query to determine if it is debatable or non-debatable.\n"
                "[Reasoning] Consider whether the query requires discussion or has a clear, factual answer. Explain in 250 words or less.\n"
                "[Classification] Return 'Debatable' or 'Non-debatable'.\n"
                "Query: {query}"
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
            communication = (
                f"[Plan] Check for non-debatable keywords in the query.\n"
                f"[Reasoning] Query contains non-debatable keywords: {query}. Assumed non-debatable without API call.\n"
                f"[Classification] Non-debatable"
            )
            result = {
                "classification": "Non-debatable",
                "reasoning": communication
            }
            response_cache[cache_key] = result
            return result

        prompt = self.prompt_template.format(query=query)
        response = api_call({"type": "gemini"}, prompt)
        classification = "Debatable"
        reasoning = response
        if "Error:" in response:
            reasoning = (
                f"[Plan] Attempt to classify query using Gemini API.\n"
                f"[Reasoning] Classification failed: {response}. Defaulting to Debatable.\n"
                f"[Classification] Debatable"
            )
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
            reasoning = (
                f"[Plan] Analyze query using Gemini API.\n"
                f"[Reasoning] {response}\n"
                f"[Classification] {classification}"
            )
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
                "[Plan] Evaluate critiques to determine convergence for '{query}' in round {round}.\n"
                "[Reasoning] Check if critiques indicate only minor suggestions or require major refinement. Explain in 250 words or less.\n"
                "[Decision] Return 'Converged' or 'Continue'.\n"
                "Critiques: {critiques}"
            )
        )

    def check_convergence(self, critiques, round, query):
        prompt = self.prompt_template.format(critiques=critiques, round=round, query=query)
        response = api_call({"type": "gemini"}, prompt)
        decision = "Continue"
        reasoning = response
        if "Error:" in response:
            reasoning = (
                f"[Plan] Evaluate critiques using Gemini API.\n"
                f"[Reasoning] Convergence check failed: {response}. Defaulting to Continue.\n"
                f"[Decision] Continue"
            )
        else:
            lines = response.strip().split('\n')
            first_line = lines[0].strip()
            if first_line in ["Converged", "Continue"]:
                decision = first_line
            else:
                response_lower = response.lower()
                if "converged" in response_lower:
                    decision = "Converged"
                elif "continue" in response_lower:
                    decision = "Continue"
            reasoning = (
                f"[Plan] Evaluate critiques for convergence.\n"
                f"[Reasoning] {response}\n"
                f"[Decision] {decision}"
            )
        return {
            "decision": decision,
            "reasoning": reasoning
        }

class Generator:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "[Plan] Generate a detailed and accurate answer to the query using the provided context.\n"
                "[Response] Provide the answer in 250 words or less.\n"
                "Query: {query}\nContext: {context}"
            )
        )

    def generate_response(self, query):
        cache_key = f"generate:{query}"
        if cache_key in response_cache:
            return response_cache[cache_key]

        context = self.memory_manager.retrieve_context(query)
        context_str = json.dumps(context, indent=2)
        prompt = self.prompt_template.format(query=query, context=context_str)
        gemini_response = api_call({"type": "gemini"}, prompt)
        openrouter_response = api_call({"type": "openrouter", "model": "meta-llama/llama-3.3-8b-instruct"}, prompt)
        communication = (
            f"[Plan] Generate responses using Gemini and OpenRouter APIs.\n"
            f"[Response] Gemini: {gemini_response}\nOpenRouter (LLaMA): {openrouter_response}"
        )
        result = {
            "gemini_response": gemini_response,
            "openrouter_response": openrouter_response,
            "combined": communication
        }
        response_cache[cache_key] = result
        self.memory_manager.store_episodic(query, result["combined"], ["generator_response"])
        self.memory_manager.store_agent_communication(query, communication, "Generator", ["generator_response"])
        return result

class Critic:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.prompt_template = PromptTemplate(
            input_variables=["response", "query", "context"],
            template=(
                "[Plan] Critique the response to '{query}' using the provided context.\n"
                "[Critique] Identify specific flaws (e.g., factual errors, missing arguments, verbosity) and suggest improvements in 250 words or less.\n"
                "Response: {response}\nContext: {context}"
            )
        )

    def critique_response(self, response, query):
        context = self.memory_manager.retrieve_context(query)
        context_str = json.dumps(context, indent=2)
        prompt = self.prompt_template.format(response=response, query=query, context=context_str)
        critique = api_call({"type": "cohere"}, prompt)
        communication = (
            f"[Plan] Critique the provided response using Cohere API.\n"
            f"[Critique] {critique}"
        )
        result = {
            "cohere_critique": critique,
            "combined": communication
        }
        self.memory_manager.store_episodic(query, result["combined"], ["critic_response"])
        self.memory_manager.store_agent_communication(query, communication, "Critic", ["critic_response"])
        return result

class Validator:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.prompt_template = PromptTemplate(
            input_variables=["response", "query", "context"],
            template=(
                "[Plan] Validate the response to '{query}' using the provided context.\n"
                "[Validation] Assess accuracy, coherence, and completeness, suggesting improvements in 250 words or less.\n"
                "Response: {response}\nContext: {context}"
            )
        )

    def validate_response(self, response, query):
        context = self.memory_manager.retrieve_context(query)
        context_str = json.dumps(context, indent=2)
        prompt = self.prompt_template.format(response=response, query=query, context=context_str)
        openrouter_validation = api_call({"type": "openrouter", "model": "meta-llama/llama-3.3-8b-instruct"}, prompt)
        cohere_validation = api_call({"type": "cohere"}, prompt)

        if "Error:" in openrouter_validation:
            gemini_validation = api_call({"type": "gemini"}, prompt)
            openrouter_validation = gemini_validation if not "Error:" in gemini_validation else "Error: Fallback to Gemini failed"
        
        communication = (
            f"[Plan] Validate response using OpenRouter and Cohere APIs.\n"
            f"[Validation] OpenRouter (LLaMA): {openrouter_validation}\nCohere: {cohere_validation}"
        )
        result = {
            "openrouter_validation": openrouter_validation,
            "cohere_validation": cohere_validation,
            "combined": communication
        }
        self.memory_manager.store_episodic(query, result["combined"], ["validator_response"])
        self.memory_manager.store_agent_communication(query, communication, "Validator", ["validator_response"])
        return result

class Orchestrator:
    def __init__(self, clear_memory=False):
        self.memory_manager = MemoryManager(clear_memory=clear_memory)
        self.generator = Generator(self.memory_manager)
        self.critic = Critic(self.memory_manager)
        self.validator = Validator(self.memory_manager)
        self.classifier = QueryClassifier()
        self.convergence_checker = ConvergenceChecker()
        self.max_rounds = 3

        self.memory_manager.store_semantic(
            fact="AI systems can be built with modular architectures.",
            source="General Knowledge",
            tags=["ai", "system_design"]
        )
        self.memory_manager.store_semantic(
            fact="Debating AI agents require iterative refinement and validation.",
            source="General Knowledge",
            tags=["ai", "debate"]
        )

    def synthesize_response(self, current_response, critique, query, scratchpad):
        context = self.memory_manager.retrieve_context(query)
        context_str = json.dumps(context, indent=2)
        prompt = (
            f"[Plan] Synthesize a new response for '{query}' based on the current response, critique, and scratchpad.\n"
            f"[Synthesis] Address the critique's suggestions and incorporate prior agent communications in 250 words or less.\n"
            f"Current Response: {current_response}\n"
            f"Critique: {critique}\n"
            f"Scratchpad: {scratchpad}\n"
            f"Context: {context_str}"
        )
        response = api_call({"type": "cohere"}, prompt)
        communication = (
            f"[Plan] Synthesize response using Cohere API.\n"
            f"[Synthesis] {response}"
        )
        self.memory_manager.store_agent_communication(query, communication, "Orchestrator", ["synthesis"])
        return response, communication

    def run_debate(self, query):
        reasoning_log = []
        scratchpad = ""

        classification_output = self.classifier.classify(query)
        scratchpad += f"[Classifier] {classification_output['reasoning']}\n"
        reasoning_log.append({
            "step": "Query Classification",
            "classification": classification_output["classification"],
            "reasoning": classification_output["reasoning"]
        })
        self.memory_manager.store_agent_communication(query, classification_output["reasoning"], "Classifier", ["classification"])

        if classification_output["classification"] == "Non-debatable":
            context = self.memory_manager.retrieve_context(query)
            context_str = json.dumps(context, indent=2)
            prompt = (
                f"[Plan] Provide a clear and concise answer to the non-debatable query.\n"
                f"[Response] Answer in 250 words or less.\n"
                f"Query: {query}\nContext: {context_str}\nScratchpad: {scratchpad}"
            )
            cache_key = f"non_debatable:{query}"
            if cache_key in response_cache:
                answer = response_cache[cache_key]
            else:
                answer = api_call({"type": "gemini"}, prompt)
                response_cache[cache_key] = answer
            communication = (
                f"[Plan] Generate answer for non-debatable query using Gemini API.\n"
                f"[Response] {answer}"
            )
            scratchpad += f"[Orchestrator] {communication}\n"
            reasoning_log.append({
                "step": "Straightforward Answer",
                "answer": answer
            })
            self.memory_manager.store_episodic(query, answer, ["non_debatable_response"])
            self.memory_manager.store_agent_communication(query, communication, "Orchestrator", ["non_debatable_response"])
            final_response = (
                f"Final Response: {answer}\n"
                f"Classification Reasoning: {classification_output['reasoning']}\n"
                f"Scratchpad: {scratchpad}"
            )
            return {
                "final_response": final_response,
                "reasoning_log": reasoning_log
            }

        gen_output = self.generator.generate_response(query)
        current_response = gen_output["combined"]
        scratchpad += f"[Generator] {current_response}\n"
        reasoning_log.append({"step": "Initial Generation", "response": current_response})

        for round in range(self.max_rounds):
            critique_output = self.critic.critique_response(current_response, query)
            scratchpad += f"[Critic] {critique_output['combined']}\n"
            reasoning_log.append({
                "step": f"Critique Round {round + 1}",
                "critique": critique_output["combined"]
            })

            convergence_output = self.convergence_checker.check_convergence(
                critiques=critique_output["combined"],
                round=round + 1,
                query=query
            )
            scratchpad += f"[ConvergenceChecker] {convergence_output['reasoning']}\n"
            reasoning_log.append({
                "step": f"Convergence Check Round {round + 1}",
                "decision": convergence_output["decision"],
                "reasoning": convergence_output["reasoning"]
            })

            if convergence_output["decision"] == "Converged":
                break

            current_response, synthesis_communication = self.synthesize_response(
                current_response, critique_output["combined"], query, scratchpad
            )
            scratchpad += f"[Orchestrator] {synthesis_communication}\n"
            reasoning_log.append({
                "step": f"Refined Response Round {round + 1}",
                "response": current_response
            })

        validation_output = self.validator.validate_response(current_response, query)
        scratchpad += f"[Validator] {validation_output['combined']}\n"
        reasoning_log.append({
            "step": "Final Validation",
            "validation": validation_output["combined"]
        })

        prompt = (
            f"[Plan] Select the better validation for '{query}' based on accuracy, coherence, and relevance.\n"
            f"[Selection] Explain the choice in 250 words or less.\n"
            f"Validation 1 (OpenRouter LLaMA): {validation_output['openrouter_validation']}\n"
            f"Validation 2 (Cohere): {validation_output['cohere_validation']}\n"
            f"Scratchpad: {scratchpad}"
        )
        selection_response = api_call({"type": "gemini"}, prompt)
        chosen_validation = validation_output["cohere_validation"]
        reasoning = selection_response
        if "Error:" in selection_response:
            reasoning = (
                f"[Plan] Attempt to select validation using Gemini API.\n"
                f"[Reasoning] Selection failed: {selection_response}. Defaulting to Cohere validation.\n"
                f"[Selection] Cohere"
            )
        else:
            if "Validation 1" in selection_response or "OpenRouter" in selection_response:
                chosen_validation = validation_output["openrouter_validation"]
            elif "Validation 2" in selection_response or "Cohere" in selection_response:
                chosen_validation = validation_output["cohere_validation"]
            reasoning = (
                f"[Plan] Select better validation using Gemini API.\n"
                f"[Reasoning] {selection_response}\n"
                f"[Selection] {'OpenRouter' if chosen_validation == validation_output['openrouter_validation'] else 'Cohere'}"
            )

        communication = (
            f"[Plan] Finalize response by selecting the better validation.\n"
            f"[Selection] {reasoning}"
        )
        scratchpad += f"[Orchestrator] {communication}\n"
        self.memory_manager.store_episodic(query, chosen_validation, ["final_response"])
        self.memory_manager.store_agent_communication(query, communication, "Orchestrator", ["final_response"])
        final_response = (
            f"Final Response: {chosen_validation}\n"
            f"Selection Reasoning: {reasoning}\n"
            f"Scratchpad: {scratchpad}"
        )

        return {
            "final_response": final_response,
            "reasoning_log": reasoning_log
        }

if __name__ == "__main__":
    response_cache.clear()
    orchestrator = Orchestrator(clear_memory=True)
    
    query = input("Enter your debate query: ").strip()
    if not query:
        print("No query provided. Please enter a valid query.")
    else:
        print(f"\nRunning debate for query: '{query}'...")
        result = orchestrator.run_debate(query)
        print("\nFinal Response:", result["final_response"])
        print("\nReasoning Log:")
        for log in result["reasoning_log"]:
            print(json.dumps(log, indent=2))