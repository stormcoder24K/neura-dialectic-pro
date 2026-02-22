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

class RLAgent:
    def __init__(self, memory_manager, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.memory_manager = memory_manager
        self.actions = actions 
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table_collection = "rl_q_table"
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
        
        try:
            collections = self.memory_manager.client.get_collections()
            if self.q_table_collection not in [c.name for c in collections.collections]:
                self.memory_manager.client.create_collection(
                    collection_name=self.q_table_collection,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dim,
                        distance=models.Distance.COSINE
                    )
                )
                print(f"Created Q-table collection: {self.q_table_collection}")
        except Exception as e:
            print(f"Warning: Failed to initialize Q-table collection: {e}")
            raise

    def get_state_embedding(self, query, context_vectors):
        """Generate state embedding from query and context vectors."""
        query_embedding = self.embedding_model.encode(query).tolist()
        context_mean = np.mean([v["vector"] for v in context_vectors], axis=0) if context_vectors else np.zeros(self.embedding_dim)
        state_embedding = (np.array(query_embedding) + context_mean).tolist()
        print(f"RL State Embedding (first 5 dims): {state_embedding[:5]}")
        return state_embedding

    def choose_action(self, state_embedding):
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
            print(f"RL Action (Random Exploration): {action}")
            return action
        
        try:
            search_result = self.memory_manager.client.query_points(
                collection_name=self.q_table_collection,
                query=state_embedding,
                limit=len(self.actions),
                with_payload=True,
                with_vectors=False
            ).points
            q_values = {action: 0.0 for action in self.actions}
            for point in search_result:
                action = point.payload.get("action")
                if action in self.actions:
                    q_values[action] = point.payload.get("q_value", 0.0)
            action = max(q_values, key=q_values.get)
            print(f"RL Action (Greedy): {action}, Q-values: {q_values}")
            return action
        except Exception as e:
            print(f"Warning: Failed to retrieve Q-values: {e}")
            action = np.random.choice(self.actions)
            print(f"RL Action (Fallback Random): {action}")
            return action

    def compute_reward(self, critique_vector, validation_vector, response):
        """Compute reward based on critique and validation vectors."""
        try:
            reward_details = {}
            critique_score = 0.0
            if critique_vector is not None:
                critique_norm = np.linalg.norm(critique_vector)
                critique_score = max(0, 5 - critique_norm * 10)  
                reward_details["critique_score"] = critique_score
                reward_details["critique_norm"] = critique_norm
            else:
                reward_details["critique_score"] = 0.0
                reward_details["critique_norm"] = 0.0

            positive_ref = self.embedding_model.encode("accurate coherent complete").tolist()
            coherence_score = 0.0
            if validation_vector is not None:
                validation_sim = 1 - cosine(validation_vector, positive_ref)
                coherence_score = validation_sim * 5  
                reward_details["coherence_score"] = coherence_score
                reward_details["validation_similarity"] = validation_sim
            else:
                reward_details["coherence_score"] = 0.0
                reward_details["validation_similarity"] = 0.0

            # Length penalty
            word_count = len(response.split())
            length_penalty = -2 if word_count > 250 else 0
            reward_details["length_penalty"] = length_penalty
            reward_details["word_count"] = word_count

            total_reward = critique_score + coherence_score + length_penalty
            reward_details["total_reward"] = total_reward
            print(f"RL Reward: {total_reward}, Details: {reward_details}")
            return total_reward, reward_details
        except Exception as e:
            print(f"Warning: Failed to compute reward: {e}")
            reward_details = {"total_reward": 0.0, "error": str(e)}
            print(f"RL Reward: 0.0, Details: {reward_details}")
            return 0.0, reward_details

    def update_q_table(self, state_embedding, action, reward, next_state_embedding):
        """Update Q-table with new experience."""
        try:
            search_result = self.memory_manager.client.query_points(
                collection_name=self.q_table_collection,
                query=state_embedding,
                limit=1,
                with_payload=True,
                with_vectors=False,
                query_filter=models.Filter(
                    must=[models.FieldCondition(key="action", match=models.MatchValue(value=action))]
                )
            ).points
            current_q = search_result[0].payload.get("q_value", 0.0) if search_result else 0.0

            next_search = self.memory_manager.client.query_points(
                collection_name=self.q_table_collection,
                query=next_state_embedding,
                limit=len(self.actions),
                with_payload=True,
                with_vectors=False
            ).points
            next_q_values = [point.payload.get("q_value", 0.0) for point in next_search]
            max_next_q = max(next_q_values) if next_q_values else 0.0

            new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
            print(f"RL Q-Table Update: Action={action}, Old Q={current_q}, New Q={new_q}, Reward={reward}")

            point_id = str(uuid.uuid4())
            self.memory_manager.client.upsert(
                collection_name=self.q_table_collection,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=state_embedding,
                        payload={"action": action, "q_value": new_q}
                    )
                ]
            )
        except Exception as e:
            print(f"Warning: Failed to update Q-table: {e}")

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
            print(f"Storing Semantic Vector (first 5 dims): {embedding[:5]}, Content: {fact}")
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
            print(f"Storing Episodic Vector (first 5 dims): {embedding[:5]}, Content: {content[:100]}...")
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

    def store_agent_communication(self, query, vector, agent, content, tags):
        try:
            point_id = str(uuid.uuid4())
            payload = {
                "type": "agent_communication",
                "vector": vector,
                "agent": agent,
                "content": content,
                "tags": tags,
                "query": query,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "relevance_score": 1.0
            }
            print(f"Storing Agent Communication Vector (first 5 dims): {vector[:5]}, Agent: {agent}, Content: {content[:100]}...")
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            self._prune_memories()
        except Exception as e:
            print(f"Warning: Failed to store agent communication: {e}")

    def store_rl_experience(self, state_embedding, action, reward, next_state_embedding):
        try:
            point_id = str(uuid.uuid4())
            payload = {
                "type": "rl",
                "state": state_embedding,
                "action": action,
                "reward": float(reward),
                "next_state": next_state_embedding,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            print(f"Storing RL Experience: Action={action}, Reward={reward}, State (first 5 dims): {state_embedding[:5]}")
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=state_embedding,
                        payload=payload
                    )
                ]
            )
            self._prune_memories()
        except Exception as e:
            print(f"Warning: Failed to store RL experience: {e}")

    def retrieve_context(self, query, k=3):
        context = {"semantic": [], "episodic": [], "agent_communication": []}
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            print(f"Retrieving Context for Query: {query}, Query Vector (first 5 dims): {query_embedding[:5]}")
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=k,
                score_threshold=0.5,
                with_payload=True,
                with_vectors=True
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
                        "vector": point.vector,
                        "agent": payload.get("agent", ""),
                        "content": payload.get("content", ""),
                        "tags": payload.get("tags", []),
                        "query": payload.get("query", ""),
                        "similarity": similarity
                    })
            print(f"Context Retrieved: {len(context['semantic'])} semantic, {len(context['episodic'])} episodic, {len(context['agent_communication'])} agent_communication")
            for item in context["agent_communication"]:
                print(f"Agent Communication Context: Agent={item['agent']}, Similarity={item['similarity']}, Content={item['content'][:100]}...")
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
            input_variables=["query", "context"],
            template=(
                "Is this query debatable or non-debatable? Use the context to inform your decision. "
                "Explain in 250 words or less and return 'Debatable' or 'Non-debatable' as the final word.\n"
                "Query: {query}\nContext: {context}"
            )
        )
        self.non_debatable_keywords = ["what is", "define", "calculate", "who is", "when is", "where is"]
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def is_likely_non_debatable(self, query):
        return any(keyword in query.lower() for keyword in self.non_debatable_keywords)

    def classify(self, query, context_vectors):
        cache_key = f"classify:{query}"
        if cache_key in response_cache:
            print(f"Classifier Cache Hit: {query}")
            return response_cache[cache_key]

        if self.is_likely_non_debatable(query):
            content = f"Query contains non-debatable keywords: {query}. Assumed non-debatable."
            vector = self.embedding_model.encode(content).tolist()
            print(f"Classifier Output: {content}")
            print(f"Classifier Vector (first 5 dims): {vector[:5]}")
            result = {
                "classification": "Non-debatable",
                "vector": vector,
                "content": content
            }
            response_cache[cache_key] = result
            return result

        context = json.dumps([v["content"] for v in context_vectors], indent=2)
        prompt = self.prompt_template.format(query=query, context=context)
        response = api_call({"type": "gemini"}, prompt)
        classification = "Debatable"
        content = response
        if "Error:" in response:
            content = f"Classification failed: {response}. Defaulting to Debatable."
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
        vector = self.embedding_model.encode(content).tolist()
        print(f"Classifier Output: {content}")
        print(f"Classifier Vector (first 5 dims): {vector[:5]}")
        result = {
            "classification": classification,
            "vector": vector,
            "content": content
        }
        response_cache[cache_key] = result
        return result

class ConvergenceChecker:
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["critique_content", "round", "query"],
            template=(
                "Evaluate the critique for '{query}' in round {round}. "
                "Does it indicate convergence (minor suggestions) or require refinement? "
                "Explain in 250 words or less and return 'Converged' or 'Continue' as the final word.\n"
                "Critique: {critique_content}"
            )
        )
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def check_convergence(self, critique_vector, critique_content, round, query):
        prompt = self.prompt_template.format(critique_content=critique_content, round=round, query=query)
        response = api_call({"type": "gemini"}, prompt)
        decision = "Continue"
        content = response
        if "Error:" in response:
            content = f"Convergence check failed: {response}. Defaulting to Continue."
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
        vector = self.embedding_model.encode(content).tolist()
        print(f"Convergence Checker Output: {content}")
        print(f"Convergence Checker Vector (first 5 dims): {vector[:5]}")
        return {
            "decision": decision,
            "vector": vector,
            "content": content
        }

class Generator:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "Provide a detailed and accurate answer to: {query} in 250 words or less.\n"
                "Context: {context}"
            )
        )
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_response(self, query, context_vectors):
        cache_key = f"generate:{query}"
        if cache_key in response_cache:
            print(f"Generator Cache Hit: {query}")
            return response_cache[cache_key]

        context = json.dumps([v["content"] for v in context_vectors], indent=2)
        prompt = self.prompt_template.format(query=query, context=context)
        gemini_response = api_call({"type": "gemini"}, prompt)
        openrouter_response = api_call({"type": "openrouter", "model": "meta-llama/llama-3.3-8b-instruct"}, prompt)
        content = f"Gemini: {gemini_response}\nOpenRouter (LLaMA): {openrouter_response}"
        vector = self.embedding_model.encode(content).tolist()
        print(f"Generator Output: {content[:200]}...")
        print(f"Generator Vector (first 5 dims): {vector[:5]}")
        result = {
            "gemini_response": gemini_response,
            "openrouter_response": openrouter_response,
            "vector": vector,
            "content": content
        }
        response_cache[cache_key] = result
        self.memory_manager.store_episodic(query, content, ["generator_response"])
        self.memory_manager.store_agent_communication(query, vector, "Generator", content, ["generator_response"])
        return result

class Critic:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.prompt_template = PromptTemplate(
            input_variables=["response", "query", "context"],
            template=(
                "Critique this response to '{query}' in 250 words or less: {response}\n"
                "Context: {context}\n"
                "Identify specific flaws (e.g., factual errors, missing arguments, verbosity) and suggest actionable improvements."
            )
        )
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def critique_response(self, response_vector, response_content, query, context_vectors):
        context = json.dumps([v["content"] for v in context_vectors], indent=2)
        prompt = self.prompt_template.format(response=response_content, query=query, context=context)
        critique = api_call({"type": "cohere"}, prompt)
        content = f"Cohere Critique: {critique}"
        vector = self.embedding_model.encode(content).tolist()
        print(f"Critic Output: {content[:200]}...")
        print(f"Critic Vector (first 5 dims): {vector[:5]}")
        result = {
            "cohere_critique": critique,
            "vector": vector,
            "content": content
        }
        self.memory_manager.store_episodic(query, content, ["critic_response"])
        self.memory_manager.store_agent_communication(query, vector, "Critic", content, ["critic_response"])
        return result

class Validator:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.prompt_template = PromptTemplate(
            input_variables=["response", "query", "context"],
            template=(
                "Validate this response to '{query}' in 250 words or less: {response}\n"
                "Context: {context}\n"
                "Is it accurate, coherent, complete? Suggest final improvements."
            )
        )
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def validate_response(self, response_vector, response_content, query, context_vectors):
        context = json.dumps([v["content"] for v in context_vectors], indent=2)
        prompt = self.prompt_template.format(response=response_content, query=query, context=context)
        openrouter_validation = api_call({"type": "openrouter", "model": "meta-llama/llama-3.3-8b-instruct"}, prompt)
        cohere_validation = api_call({"type": "cohere"}, prompt)

        if "Error:" in openrouter_validation:
            gemini_validation = api_call({"type": "gemini"}, prompt)
            openrouter_validation = gemini_validation if not "Error:" in gemini_validation else "Error: Fallback to Gemini failed"
        
        content = f"OpenRouter (LLaMA): {openrouter_validation}\nCohere: {cohere_validation}"
        vector = self.embedding_model.encode(content).tolist()
        print(f"Validator Output: {content[:200]}...")
        print(f"Validator Vector (first 5 dims): {vector[:5]}")
        result = {
            "openrouter_validation": openrouter_validation,
            "cohere_validation": cohere_validation,
            "vector": vector,
            "content": content
        }
        self.memory_manager.store_episodic(query, content, ["validator_response"])
        self.memory_manager.store_agent_communication(query, vector, "Validator", content, ["validator_response"])
        return result

class Orchestrator:
    def __init__(self, clear_memory=False):
        self.memory_manager = MemoryManager(clear_memory=clear_memory)
        self.rl_agent = RLAgent(self.memory_manager, actions=['openrouter', 'gemini', 'cohere'])
        self.generator = Generator(self.memory_manager)
        self.critic = Critic(self.memory_manager)
        self.validator = Validator(self.memory_manager)
        self.classifier = QueryClassifier()
        self.convergence_checker = ConvergenceChecker()
        self.max_rounds = 3
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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

    def synthesize_response(self, current_response_vector, current_response_content, critique_vector, critique_content, query, context_vectors):
        state_embedding = self.rl_agent.get_state_embedding(query, context_vectors)
        action = self.rl_agent.choose_action(state_embedding)
        
        context = json.dumps([v["content"] for v in context_vectors], indent=2)
        prompt = (
            f"Synthesize a new response for '{query}' in 250 words or less based on:\n"
            f"Current Response: {current_response_content}\n"
            f"Critique: {critique_content}\n"
            f"Context: {context}"
        )
        response = api_call({"type": action}, prompt)
        response_vector = self.embedding_model.encode(response).tolist()
        
        next_context = self.memory_manager.retrieve_context(query)
        next_state_embedding = self.rl_agent.get_state_embedding(query, next_context["agent_communication"])
        reward, reward_details = self.rl_agent.compute_reward(critique_vector, response_vector, response)
        self.rl_agent.update_q_table(state_embedding, action, reward, next_state_embedding)
        self.rl_agent.memory_manager.store_rl_experience(state_embedding, action, reward, next_state_embedding)
        self.memory_manager.store_agent_communication(query, response_vector, "Orchestrator", response, ["synthesis"])
        
        print(f"Synthesis Output: {response[:200]}...")
        print(f"Synthesis Vector (first 5 dims): {response_vector[:5]}")
        return response, response_vector, {
            "action": action,
            "reward_details": reward_details,
            "state_embedding": state_embedding[:5],
            "next_state_embedding": next_state_embedding[:5]
        }

    def run_debate(self, query):
        reasoning_log = []

        context = self.memory_manager.retrieve_context(query)
        context_vectors = context["agent_communication"]
        reasoning_log.append({
            "step": "Context Retrieval",
            "context": {
                "semantic": context["semantic"],
                "episodic": context["episodic"],
                "agent_communication": [{"agent": v["agent"], "content": v["content"], "similarity": v["similarity"]} for v in context["agent_communication"]]
            }
        })
        
        classification_output = self.classifier.classify(query, context_vectors)
        reasoning_log.append({
            "step": "Query Classification",
            "classification": classification_output["classification"],
            "content": classification_output["content"],
            "vector": classification_output["vector"][:5]
        })
        self.memory_manager.store_agent_communication(
            query, classification_output["vector"], "Classifier", classification_output["content"], ["classification"]
        )

        if classification_output["classification"] == "Non-debatable":
            state_embedding = self.rl_agent.get_state_embedding(query, context_vectors)
            action = self.rl_agent.choose_action(state_embedding)
            
            context_str = json.dumps([v["content"] for v in context_vectors], indent=2)
            prompt = f"Provide a clear and concise answer to: {query} in 250 words or less.\nContext: {context_str}"
            cache_key = f"non_debatable:{query}"
            if cache_key in response_cache:
                answer = response_cache[cache_key]
            else:
                answer = api_call({"type": action}, prompt)
                response_cache[cache_key] = answer
            
            answer_vector = self.embedding_model.encode(answer).tolist()
            next_context = self.memory_manager.retrieve_context(query)
            next_state_embedding = self.rl_agent.get_state_embedding(query, next_context["agent_communication"])
            reward, reward_details = self.rl_agent.compute_reward(None, answer_vector, answer)
            self.rl_agent.update_q_table(state_embedding, action, reward, next_state_embedding)
            self.memory_manager.store_rl_experience(state_embedding, action, reward, next_state_embedding)
            self.memory_manager.store_agent_communication(query, answer_vector, "Orchestrator", answer, ["non_debatable_response"])
            
            print(f"Non-Debatable Answer: {answer[:200]}...")
            print(f"Answer Vector (first 5 dims): {answer_vector[:5]}")
            reasoning_log.append({
                "step": "Straightforward Answer",
                "answer": answer,
                "vector": answer_vector[:5],
                "rl_details": {
                    "action": action,
                    "reward_details": reward_details,
                    "state_embedding": state_embedding[:5],
                    "next_state_embedding": next_state_embedding[:5]
                }
            })
            self.memory_manager.store_episodic(query, answer, ["non_debatable_response"])
            final_response = (
                f"Final Response: {answer}\n"
                f"Classification Reasoning: {classification_output['content']}"
            )
            return {
                "final_response": final_response,
                "reasoning_log": reasoning_log
            }

        gen_output = self.generator.generate_response(query, context_vectors)
        current_response_vector = gen_output["vector"]
        current_response_content = gen_output["content"]
        reasoning_log.append({
            "step": "Initial Generation",
            "content": current_response_content,
            "vector": current_response_vector[:5]
        })

        for round in range(self.max_rounds):
            critique_output = self.critic.critique_response(current_response_vector, current_response_content, query, context_vectors)
            reasoning_log.append({
                "step": f"Critique Round {round + 1}",
                "content": critique_output["content"],
                "vector": critique_output["vector"][:5]
            })

            convergence_output = self.convergence_checker.check_convergence(
                critique_output["vector"], critique_output["content"], round + 1, query
            )
            reasoning_log.append({
                "step": f"Convergence Check Round {round + 1}",
                "decision": convergence_output["decision"],
                "content": convergence_output["content"],
                "vector": convergence_output["vector"][:5]
            })
            self.memory_manager.store_agent_communication(
                query, convergence_output["vector"], "ConvergenceChecker", convergence_output["content"], ["convergence"]
            )

            if convergence_output["decision"] == "Converged":
                break

            current_response_content, current_response_vector, synthesis_details = self.synthesize_response(
                current_response_vector, current_response_content, critique_output["vector"], critique_output["content"], query, context_vectors
            )
            reasoning_log.append({
                "step": f"Refined Response Round {round + 1}",
                "content": current_response_content,
                "vector": current_response_vector[:5],
                "rl_details": synthesis_details
            })

        validation_output = self.validator.validate_response(current_response_vector, current_response_content, query, context_vectors)
        reasoning_log.append({
            "step": "Final Validation",
            "content": validation_output["content"],
            "vector": validation_output["vector"][:5]
        })

        state_embedding = self.rl_agent.get_state_embedding(query, context_vectors)
        action = self.rl_agent.choose_action(state_embedding)
        
        prompt = (
            f"Select the better validation for '{query}' in 250 words or less based on accuracy, coherence, and relevance:\n"
            f"Validation 1 (OpenRouter LLaMA): {validation_output['openrouter_validation']}\n"
            f"Validation 2 (Cohere): {validation_output['cohere_validation']}"
        )
        selection_response = api_call({"type": action}, prompt)
        chosen_validation = validation_output["cohere_validation"]
        content = selection_response
        if "Error:" in selection_response:
            content = f"Selection failed: {selection_response}. Defaulting to Cohere validation."
        else:
            if "Validation 1" in selection_response or "OpenRouter" in selection_response:
                chosen_validation = validation_output["openrouter_validation"]
            elif "Validation 2" in selection_response or "Cohere" in selection_response:
                chosen_validation = validation_output["cohere_validation"]
        
        selection_vector = self.embedding_model.encode(content).tolist()
        next_context = self.memory_manager.retrieve_context(query)
        next_state_embedding = self.rl_agent.get_state_embedding(query, next_context["agent_communication"])
        reward, reward_details = self.rl_agent.compute_reward(None, selection_vector, chosen_validation)
        self.rl_agent.update_q_table(state_embedding, action, reward, next_state_embedding)
        self.memory_manager.store_rl_experience(state_embedding, action, reward, next_state_embedding)
        self.memory_manager.store_agent_communication(query, selection_vector, "Orchestrator", content, ["final_response"])
        self.memory_manager.store_episodic(query, chosen_validation, ["final_response"])
        
        print(f"Validation Selection Output: {content[:200]}...")
        print(f"Selection Vector (first 5 dims): {selection_vector[:5]}")
        reasoning_log.append({
            "step": "Validation Selection",
            "content": content,
            "vector": selection_vector[:5],
            "rl_details": {
                "action": action,
                "reward_details": reward_details,
                "state_embedding": state_embedding[:5],
                "next_state_embedding": next_state_embedding[:5]
            }
        })

        final_response = (
            f"Final Response: {chosen_validation}\n"
            f"Selection Reasoning: {content}"
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