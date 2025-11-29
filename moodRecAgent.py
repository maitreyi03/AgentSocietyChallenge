import json
import os
import re
from dotenv import load_dotenv
from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.agent.modules.memory_modules import MemoryBase
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.llm.llm import GeminiLLM, LLMBase
from websocietysimulator.simulator import Simulator
import tiktoken
import ast

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except:
        print(encoding.encode(string))
    return a

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
        
        INSTRUCTION : Your goal is to infer a very specific mood (which aligns the most with the user's emotional state and preferences) that can be used to recommend books.
        Strategy : 1. If the user's request is specific for eg, ' I want to read scary book', output the corresponding mood for eg 'spooky, thrilling, dark, halloween'.
        2. If the user's request is generic and memory is strong for eg, ' I want to read a book', analyze the long-term profile and recent context to infer a mood that is likely to resonate with the user.
        (eg, if the long-term profile indicates a preference for uplifting and heartwarming stories, and recent context shows interest in cozy settings, infer a mood like 'cozy, heartwarming, uplifting'.)
        3. If the user's request is generic and memory is weak, infer a mood that is diverse and exploratory instead of neutral moods.
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
        
        plan = [
            {'description': 'First I need to find user information'},
            {'description': 'Next, I need to find item information'},
            {'description': 'Next, I need to find review information'}
        ]
         
        user = ''
        item_list = []
        history_review = ''
        for sub_task in plan:
            
            if 'user' in sub_task['description']:
                user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                input_tokens = num_tokens_from_string(user)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    user = encoding.decode(encoding.encode(user)[:12000])

            elif 'item' in sub_task['description']:
                for n_bus in range(len(self.task['candidate_list'])):
                    item = self.interaction_tool.get_item(item_id=self.task['candidate_list'][n_bus])
                    keys_to_extract = ['item_id', 'name','stars','review_count','attributes','title', 'average_rating', 'rating_number','description','ratings_count','title_without_series']
                    filtered_item = {key: item[key] for key in keys_to_extract if key in item}
                item_list.append(filtered_item)
                # print(item)
            elif 'review' in sub_task['description']:
                history_review = str(self.interaction_tool.get_reviews(user_id=self.task['user_id']))
                input_tokens = num_tokens_from_string(history_review)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    history_review = encoding.decode(encoding.encode(history_review)[:12000])
            else:
                pass 
            
        task_description = f'''
        You are a real user on goodreads. 
        You want to read a book that you haven't read before and that matches your mood. 
        Your profile information is as follows: {user}.
        Your historical item review text and stars are as follows: {history_review}. 
        Now you need to rank the following 20 items: {self.task['candidate_list']} according to their match degree to your preference.
        Please rank the more interested items more front in your rank list.
        The information of the above 20 candidate items is as follows: {item_list}.

        Your final output should be ONLY a ranked item list of {self.task['candidate_list']} with the following format, DO NOT introduce any other item ids!
        DO NOT output your analysis process!

        The correct output format:

        ['item id1', 'item id2', 'item id3', ...]

        '''      
            
        # Advanced Reasoning to infer user's desired mood
        inferred_mood = self.reasoning(task_description)
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
        You are a user who wants to read a book that matches with your mood. 
        
        Rank the books based on moods that aligns the most with the following user's profile.
        User's inferred mood: {inferred_mood}
        Plan: {plan_steps}
        
        Candidate books and their aligned moods: {mood_data}
        
        INSTRUCTION: Apply the ranking strategy detailed in the plan by prioritizing books whose 'mood_tag' match the Inferred Mood. 
        Then, apply a penalty/boost based on novelty by checking how different the book is from the user's historical preferences, 
        which you can infer from the long-term memory that can be accessed.
        Finally, return a sorted list of item IDs in descending order of relevance to the inferred mood.
        
        Your final output should be ONLY a ranked item list of {self.task['candidate_list']} with the following format, DO NOT introduce any other item ids!
        DO NOT output your analysis process!

        The correct output format:

        ['item id1', 'item id2', 'item id3', ...]

        '''
        output = self.llm(messages=[{"role": "user", "content": description}])
                

        try:
            match = re.search(r"\[.*\]", output, re.DOTALL)
            if match:
                result = match.group()
            else:
                print("No list found.")
            print('Processed Output:',eval(result))
            bookid_title_map = {item['item_id']: item['title'] for item in mood_data}
            recommended_titles = [bookid_title_map[item_id] for item_id in eval(result) if item_id in bookid_title_map]
            print(f"Recommended Book Titles: {recommended_titles}")
            return eval(result)
        except:
            print('format error')
            return ['']
    
    
if __name__ == "__main__":
    load_dotenv()
    api_key = os.environ['api_key']
    task_set = "goodreads" # "goodreads" or "yelp"
    # Initialize Simulator
    simulator = Simulator(data_dir="../../data/processed-data", device="auto", cache=True)

    # Load scenarios
    simulator.set_task_and_groundtruth(task_dir=f"./track2/{task_set}/tasks", groundtruth_dir=f"./track2/{task_set}/groundtruth")

    # Set your custom agent
    simulator.set_agent(MoodRecAgent)

    # Set LLM client
    simulator.set_llm(GeminiLLM(api_key=api_key))

    # Run evaluation
    # If you don't set the number of tasks, the simulator will run all tasks.
    agent_outputs = simulator.run_simulation(number_of_tasks=400, enable_threading=True, max_workers=10)

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(f'./evaluation_results_track2_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
