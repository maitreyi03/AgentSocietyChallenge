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
        """Initialize the planning module"""
        super().__init__(llm=llm)
    
    # S = (0.8 * Mood_Similarity) + (0.2 * Novelty_Score).
    # we check the how well the book matches the mood vector and how similar it is to the last three books read
    # this is the score that will prioritizethe books that match that mood
    # can be adjusted later
    def create_prompt(self, task_type, description, feedback, few_shot):
        """Override the parent class's create_prompt method"""
        if feedback == '':
            prompt = '''You are a mood recommender agent who divides a {task_type} task into several subtasks.
            You need to recommend books based on the user's desired mood.
            The following are some examples:
            Recommend a list of books that matches with the user's desired mood: 'cozy, fall-themed, heartwarming'.
            sub-task 1: {{"description": "First, retrieve all the books whose mood vector is closest to the 'cozy, fall-themed, heartwarming'", "reasoning instruction": "Use vector similarity search on the mood feature space, prioritizing results with a 'cozy' and 'heartwarming' tag."}} 
            sub-task 2: {{"description": "Next, apply a scoring mechanism that heavily weights mood alignment and moderately weights novelty.", "reasoning instruction": "Calculate the final score S = (0.8 * Mood_Similarity) + (0.2 * Novelty_Score)."}} 
            sub-task 3: {{"description": "Finally, return the top 10 ranked items.", "reasoning instruction": "Sort the final candidate list by the calculated score S in descending order."}} 
            {reflexion_section} Task: Recommend a list of books matching the inferred profile: 
            {description}
'''
        if feedback == '':
            reflexion_section = ""
        else:
            reflexion_section = f"end\n--------------------\n Reflexion: {feedback} "
            
        prompt = prompt.format(example=few_shot, description=description, task_type=task_type, reflexion_section=reflexion_section) 
        return prompt

class RecReasoning(ReasoningBase):
    """Inherits from ReasoningBase"""
    
    def __init__(self, profile_type_prompt, memory: MemoryBase, llm): 
        super().__init__(profile_type_prompt=profile_type_prompt, memory=memory, llm=llm) 
        self.memory = memory
        
    def __call__(self, description: str):
        """Override the parent class's __call__ method"""
        recent_context = "No recent context available."
        long_term_profile = "Couldn't find history of the user's interests. Rely only on the current task description."

        if self.memory:
            try:
                recent_context = self.memory.retrieve_context(query=f"user's request: {description}")
                long_term_profile = self.memory.get_long_term_profile(profile_type=self.profile_type_prompt)
            except Exception as e:
                print(f"Warning: Memory retrieval failed: {e}")
        
        prompt = '''
        You are a profile inference agent. Your task is to determine the user's mood based on what the user is requesting.
        If the request is generic, you must analyze the historical profile and recent context to infer a mood that is highly likely to appeal to the user, even if not explicitly stated.
        User's request: {description}
        User's long-term preference: {long_term_profile}
        Recent context/interactions: {recent_context}
        Output a string that describes the desired mood so that this can be used to search the book database for matching books.
        
        Inferred Mood:
'''
        prompt = prompt.format(
            description=description,
            long_term_profile=long_term_profile,
            recent_context=recent_context
        )
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.6,
            max_tokens=500
        )
        
        return reasoning_result
    
class MoodRecAgent(RecommendationAgent):
    def __init__(self, llm: LLMBase, memory=None): 
        super().__init__(llm=llm) 
        self.memory = memory 
        self.planning = RecPlanning(llm=self.llm)
        self.reasoning = RecReasoning(profile_type_prompt="mood", memory=self.memory, llm=self.llm)
        
    def workflow(self):
        """
        Simulate user behavior
        Returns:
            list: Sorted list of item IDs
        """
        # Advanced Reasoning to infer user's desired mood
        inferred_mood = self.reasoning(description=self.task['description'])
        print(f"Inferred Mood: {inferred_mood}")
        
        # Planning to create a plan for recommendation
        plan = self.planning.create_prompt(
            task_type='Mood-Based Recommendation Task',
            description=f"Please recommend books that match the following mood: '{inferred_mood}'",
            feedback='',
            few_shot='')
        
        # Use llm to execute the plan and get the steps
        plan_steps = self.llm(
            messages=[{"role": "user", "content": plan}])
        print(f"Plan Steps: {plan_steps}")

        # retrieve the user's metadata
        mood_data = []
        for i in self.task['candidate_list']:
            item = self.interaction_tool.get_item(item_id=i)
            keys = ['item_id','title','description','average_rating']
            filtered_item = {k: item[k] for k in keys}
            # adding mood tags
            filtered_item['mood_tag'] = item.get('mood_tag',[])
            mood_data.append(filtered_item)
            
        description = f'''
        Your task is to act as a Mood Recommender Agent. 
        
        Rank the books based on moods that aligns the most with the following user's profile.
        User's inferred mood: {inferred_mood}
        Plan: {plan_steps}
        
        Candidate books and their aligned moods: {mood_data}
        
        INSTRUCTION: Apply the ranking strategy detailed in the plan by prioritizing books whose 'mood_tag' match the Inferred Mood. 
        Then, apply a penalty/boost based on novelty by checking how different the book is from the user's historical preferences, 
        which you can infer from the long-term memory that can be accessed.
        Finally, return a sorted list of item IDs in descending order of relevance to the inferred mood.
        '''
        
        sorted_candidate_list = self.llm(messages=[{"role": "user", "content": description}])
        return sorted_candidate_list
    
    
if __name__ == "__main__":
    task_set = "goodreads" # "goodreads" or "yelp"
    # Initialize Simulator
    simulator = Simulator(data_dir="../../data/processed-data", device="auto", cache=True)

    # Load scenarios
    simulator.set_task_and_groundtruth(task_dir=f"./track2/{task_set}/tasks", groundtruth_dir=f"./track2/{task_set}/groundtruth")

    # Set your custom agent
    simulator.set_agent(MoodRecAgent)

    # Set LLM client
    simulator.set_llm(GeminiLLM(api_key="AIzaSyCfnSOUK3nrqHQRkxS1Qp6LwzBSqPbg4_I"))

    # Run evaluation
    # If you don't set the number of tasks, the simulator will run all tasks.
    agent_outputs = simulator.run_simulation(number_of_tasks=50, enable_threading=True, max_workers=10)

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(f'./evaluation_results_track2_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
