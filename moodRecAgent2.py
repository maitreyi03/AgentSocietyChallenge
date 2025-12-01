import json
from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.agent.modules.memory_modules import MemoryBase
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.llm.llm import GeminiLLM, LLMBase
from websocietysimulator.simulator import Simulator

class RecPlanning(PlanningBase):
    """Inherits from PlanningBase"""
    
    def __init__(self, llm):
        super().__init__(llm=llm)
    
    def create_prompt(self, task_type, description, feedback, few_shot):
        """
        Create a planning prompt that decomposes a mood-based recommendation task
        into subtasks. Optionally incorporate prior feedback (reflection) and few-shot examples.
        """
        if feedback:
            reflexion_section = (
                "end\n--------------------\n"
                f"Reflexion (previous feedback from evaluation): {feedback}\n\n"
            )
        else:
            reflexion_section = ""
        
        example_section = ""
        if few_shot:
            example_section = f"Here are some example decompositions:\n{few_shot}\n\n"
        
        prompt = (
f"""You are a mood recommender agent who divides a {task_type} into several subtasks.
You need to recommend books based on the user's desired mood.

{example_section}
For example:
- User mood: 'cozy, fall-themed, heartwarming'
  - sub-task 1: {{
      "description": "Retrieve books whose mood vector is closest to 'cozy, fall-themed, heartwarming'",
      "reasoning_instruction": "Use vector similarity search on the mood feature space, prioritizing results with 'cozy' and 'heartwarming' tags."
    }}
  - sub-task 2: {{
      "description": "Apply a scoring mechanism that heavily weights mood alignment and moderately weights novelty.",
      "reasoning_instruction": "Calculate the final score S = (0.8 * Mood_Similarity) + (0.2 * Novelty_Score)."
    }}
  - sub-task 3: {{
      "description": "Return the top 10 ranked items.",
      "reasoning_instruction": "Sort candidates by S in descending order."
    }}

{reflexion_section}
Now, decompose the following task into a concise list of subtasks in JSON format:

Task description: {description}

Output FORMAT:
A JSON array of objects, each with fields:
- "description"
- "reasoning_instruction"

Do NOT include any additional text outside the JSON.
"""
        )
        return prompt


class RecReasoning(ReasoningBase):
    """Inherits from ReasoningBase"""
    
    def __init__(self, profile_type_prompt, memory: MemoryBase, llm): 
        super().__init__(profile_type_prompt=profile_type_prompt, memory=memory, llm=llm) 
        self.memory = memory
        
    def __call__(self, description: str):
        recent_context = "No recent context available."
        long_term_profile = (
            "Couldn't find history of the user's interests. "
            "Rely only on the current task description."
        )

        if self.memory:
            try:
                recent_context = self.memory.retrieve_context(
                    query=f"user's request: {description}"
                )
                long_term_profile = self.memory.get_long_term_profile(
                    profile_type=self.profile_type_prompt
                )
            except Exception as e:
                print(f"Warning: Memory retrieval failed: {e}")
        
        prompt = f"""
You are a profile inference agent. Your task is to determine the user's desired MOOD
for book recommendations.

If the request is generic, you must analyze the historical profile and recent context
to infer a mood that is highly likely to appeal to the user.

User's current request:
{description}

User's long-term preferences:
{long_term_profile}

Recent context / interactions:
{recent_context}

Output ONLY a short mood description (no explanation, no bullet points),
for example: "cozy, slow-paced, bittersweet".
        
Inferred Mood:
"""
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.6,
            max_tokens=100
        )
        
        # If GeminiLLM returns a dict, adjust accordingly; this assumes a string:
        if isinstance(reasoning_result, dict):
            mood = reasoning_result.get("content", "").strip()
        else:
            mood = str(reasoning_result).strip()
        
        return mood
    

class MoodRecAgent(RecommendationAgent):
    def __init__(self, llm: LLMBase, memory=None): 
        super().__init__(llm=llm) 
        self.memory = memory 
        self.planning = RecPlanning(llm=self.llm)
        self.reasoning = RecReasoning(
            profile_type_prompt="mood",
            memory=self.memory,
            llm=self.llm
        )
        
    def workflow(self):
        """
        Main agent workflow.
        Returns:
            list[int/str]: Sorted list of item IDs (descending relevance).
        """
        # 1) Infer user's mood
        inferred_mood = self.reasoning(description=self.task['description'])
        print(f"Inferred Mood: {inferred_mood}")
        
        # 2) Build planning prompt and get plan steps as JSON
        plan_prompt = self.planning.create_prompt(
            task_type='Mood-Based Recommendation Task',
            description=f"Please recommend books that match the following mood: '{inferred_mood}'",
            feedback='',
            few_shot=''
        )
        
        plan_raw = self.llm(messages=[{"role": "user", "content": plan_prompt}])
        print(f"Plan raw: {plan_raw}")
        
        try:
            if isinstance(plan_raw, dict):
                plan_text = plan_raw.get("content", "")
            else:
                plan_text = str(plan_raw)
            plan_steps = json.loads(plan_text)
        except Exception:
            # Fallback: use raw text if JSON parsing fails
            plan_steps = plan_raw
        print(f"Plan Steps (parsed): {plan_steps}")
        
        # 3) Collect candidate items + mood tags
        mood_data = []
        for i in self.task['candidate_list']:
            item = self.interaction_tool.get_item(item_id=i)
            keys = ['item_id', 'title', 'description', 'average_rating']
            filtered_item = {k: item.get(k) for k in keys}
            filtered_item['mood_tag'] = item.get('mood_tag', [])
            mood_data.append(filtered_item)
            
        # 4) Final ranking prompt
        ranking_prompt = f"""
You are a Mood Recommender Agent.

User's inferred mood:
{inferred_mood}

Plan (subtasks to follow):
{plan_steps}

Candidate books (JSON list):
{json.dumps(mood_data, ensure_ascii=False)}

INSTRUCTIONS:
1. Prioritize books whose 'mood_tag' list best matches the user's inferred mood.
2. Break ties by:
   - Higher average_rating
   - Better alignment with the *variety* in mood_tag (avoid recommending 10 nearly identical books).
3. Consider novelty: slightly favor books whose mood_tag distribution is not identical
   to what you'd expect from a user who ALWAYS reads the same mood profile.
   (You may infer this abstractly from the inferred mood; you do NOT need exact history.)
4. Return ONLY a JSON array of item_id values in DESCENDING order of relevance.
   Example: ["12345", "45678", "99999"]

Do NOT include any additional explanation or text outside the JSON array.
"""
        
        ranking_raw = self.llm(messages=[{"role": "user", "content": ranking_prompt}])
        print(f"Ranking raw: {ranking_raw}")
        
        # 5) Parse ranking result into a Python list
        if isinstance(ranking_raw, dict):
            ranking_text = ranking_raw.get("content", "")
        else:
            ranking_text = str(ranking_raw)
        
        try:
            sorted_candidate_list = json.loads(ranking_text)
        except Exception as e:
            print(f"Warning: could not parse ranking JSON, error: {e}")
            # Fallback: if parsing fails, just return original candidate_list
            sorted_candidate_list = self.task['candidate_list']
        
        return sorted_candidate_list
