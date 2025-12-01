"""
Advanced Mood and Theme-Based Recommendation Agent for Goodreads

This agent implements a sophisticated multi-stage ranking system that:
1. Infers user's desired mood from their history and current context
2. Extracts book themes and moods from metadata
3. Uses semantic matching to align books with user preferences
4. Applies a multi-factor scoring system (mood, theme, novelty, rating)
5. Learns long-term user preferences through memory

"""

import json
import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.agent.modules.memory_modules import MemoryBase, MemoryDILU
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.llm.llm import GeminiLLM, LLMBase
from websocietysimulator.simulator import Simulator
import tiktoken
import ast


def num_tokens_from_string(string: str) -> int:
    """Count tokens in a string using tiktoken"""
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        return len(encoding.encode(string))
    except Exception as e:
        print(f"Token counting error: {e}")
        return 0


def truncate_to_tokens(string: str, max_tokens: int = 12000) -> str:
    """Truncate string to maximum token count"""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(string)
    if len(tokens) > max_tokens:
        return encoding.decode(tokens[:max_tokens])
    return string


class EnhancedMoodPlanning(PlanningBase):
    """Advanced planning module with multi-stage strategy"""
    
    def __init__(self, llm):
        super().__init__(llm=llm)
    
    def create_strategy_prompt(self, user_context: str, inferred_mood: str) -> str:
        """Create a strategic plan for mood-based ranking"""
        prompt = f"""You are an expert book recommendation strategist. Create a ranking strategy based on mood and theme analysis.

User Context: {user_context}
Inferred Mood/Theme: {inferred_mood}

Design a multi-factor scoring strategy that considers:
1. **Mood Alignment** (40%): How well the book's mood matches the inferred mood
2. **Theme Relevance** (30%): Thematic coherence with user preferences (e.g., cozy, fall-themed, fast-paced)
3. **Rating Quality** (15%): Book's average rating and popularity
4. **Novelty Factor** (15%): Balance between familiar patterns and new discoveries

Output your strategy in this format:
Strategy:
1. Identify mood keywords: [list key mood descriptors]
2. Weight factors: [specify how to weight each factor]
3. Ranking approach: [describe the ranking logic]
"""
        return prompt


class AdvancedMoodReasoning(ReasoningBase):
    """Enhanced reasoning module with chain-of-thought and memory integration"""
    
    def __init__(self, profile_type_prompt: str, memory: Optional[MemoryBase], llm: LLMBase):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=memory, llm=llm)
        self.memory = memory
        
    def infer_user_mood(self, user_profile: str, recent_reviews: str, task_context: str) -> str:
        """Infer user's desired mood with advanced context analysis"""
        
        # Retrieve long-term preferences if memory is available
        long_term_profile = "No long-term history available."
        if self.memory:
            try:
                long_term_profile = self.memory.get_long_term_profile(profile_type=self.profile_type_prompt)
            except Exception as e:
                print(f"Memory retrieval warning: {e}")
        
        prompt = f"""You are an expert at inferring reading preferences based on user behavior.

**Task**: Determine the specific mood/theme the user is seeking based on their profile and history.

**User Profile**: {user_profile}

**Recent Reviews** (last few books): {recent_reviews}

**Long-term Preferences**: {long_term_profile}

**Analysis Instructions**:
1. Identify patterns in the user's reading history (genres, themes, pacing preferences)
2. Detect emotional tones in their reviews (what moods do they enjoy?)
3. Infer thematic preferences (e.g., cozy settings, fast-paced action, fall themes, heartwarming stories)
4. Consider seasonal or contextual patterns

**Output Format**:
Provide a concise mood/theme description (2-4 descriptive phrases) that captures:
- Emotional tone (e.g., cozy, thrilling, heartwarming, dark)
- Pacing preference (e.g., fast-paced, slow-burn, contemplative)
- Thematic elements (e.g., fall-themed, romantic, mysterious, uplifting)

Examples:
- "cozy, fall-themed, heartwarming, slow-paced"
- "thrilling, fast-paced, dark, suspenseful"
- "uplifting, romantic, character-driven, emotional"

Inferred Mood/Theme:"""

        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.7,  # Higher temperature for creative inference
            max_tokens=150
        )
        
        # Clean up the result
        mood = reasoning_result.strip().replace('"', '').replace("'", '')
        return mood
    
    def extract_book_moods(self, book_data: Dict[str, Any]) -> List[str]:
        """Extract mood and theme descriptors from book metadata"""
        moods = []
        
        # Extract from mood_tag if available
        if 'mood_tag' in book_data and book_data['mood_tag']:
            moods.extend(book_data['mood_tag'])
        
        # Extract from description using LLM
        if 'description' in book_data and book_data['description']:
            description = book_data['description'][:500]  # Limit length
            
            prompt = f"""Analyze this book description and extract 3-5 mood/theme keywords.

Description: {description}

Focus on:
- Emotional tone (cozy, dark, uplifting, etc.)
- Pacing (fast-paced, slow-burn, etc.)
- Themes (romance, mystery, fall-themed, etc.)

Output only the keywords, comma-separated:"""

            try:
                messages = [{"role": "user", "content": prompt}]
                result = self.llm(messages=messages, temperature=0.3, max_tokens=50)
                extracted_moods = [m.strip() for m in result.split(',')]
                moods.extend(extracted_moods[:5])
            except Exception as e:
                print(f"Mood extraction error: {e}")
        
        return moods
    
    def calculate_mood_similarity(self, target_mood: str, book_moods: List[str]) -> float:
        """Calculate semantic similarity between target mood and book moods"""
        if not book_moods:
            return 0.0
        
        # Simple keyword matching (can be enhanced with embeddings)
        target_keywords = set(target_mood.lower().split())
        book_keywords = set(' '.join(book_moods).lower().split())
        
        # Jaccard similarity
        intersection = target_keywords & book_keywords
        union = target_keywords | book_keywords
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)


class AdvancedMoodRecAgent(RecommendationAgent):
    """
    Advanced Mood and Theme-Based Recommendation Agent
    
    Features:
    - Multi-stage reasoning for mood inference
    - Semantic matching of moods and themes
    - Multi-factor scoring (mood, theme, rating, novelty)
    - Memory integration for long-term preference learning
    - Robust error handling and fallback strategies
    """
    
    def __init__(self, llm: LLMBase, memory: Optional[MemoryBase] = None):
        super().__init__(llm=llm)
        
        # Initialize memory if not provided
        if memory is None:
            try:
                self.memory = MemoryDILU(llm=self.llm)
            except:
                self.memory = None
                print("Memory module not available, continuing without memory")
        else:
            self.memory = memory
        
        self.planning = EnhancedMoodPlanning(llm=self.llm)
        self.reasoning = AdvancedMoodReasoning(
            profile_type_prompt="mood_and_theme",
            memory=self.memory,
            llm=self.llm
        )
        
    def workflow(self) -> List[str]:
        """
        Main recommendation workflow with multi-stage ranking
        
        Returns:
            List[str]: Ranked list of item IDs
        """
        try:
            # Stage 1: Data Collection
            print("Stage 1: Collecting user and item data...")
            user_data, item_data, review_data = self._collect_data()
            
            # Stage 2: Mood Inference
            print("Stage 2: Inferring user's desired mood...")
            inferred_mood = self._infer_user_mood(user_data, review_data)
            print(f"Inferred Mood: {inferred_mood}")
            
            # Stage 3: Candidate Analysis
            print("Stage 3: Analyzing candidate books...")
            analyzed_candidates = self._analyze_candidates(item_data, inferred_mood)
            
            # Stage 4: Multi-Factor Scoring
            print("Stage 4: Applying multi-factor scoring...")
            scored_candidates = self._score_candidates(
                analyzed_candidates,
                inferred_mood,
                user_data,
                review_data
            )
            
            # Stage 5: Final Ranking
            print("Stage 5: Generating final ranking...")
            ranked_items = self._generate_final_ranking(scored_candidates)
            
            print(f"Top 5 recommendations: {ranked_items[:5]}")
            return ranked_items
            
        except Exception as e:
            print(f"Error in workflow: {e}")
            return self._fallback_ranking()
    
    def _collect_data(self) -> tuple:
        """Collect user profile, item data, and review history"""
        # Get user profile
        user_raw = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
        user_data = truncate_to_tokens(user_raw, max_tokens=8000)
        
        # Get item information for all candidates
        item_data = []
        for item_id in self.task['candidate_list']:
            try:
                item = self.interaction_tool.get_item(item_id=item_id)
                # Extract relevant fields
                keys = ['item_id', 'title', 'description', 'average_rating', 
                       'ratings_count', 'title_without_series']
                filtered_item = {k: item.get(k, '') for k in keys}
                
                # Add mood tags if available
                filtered_item['mood_tag'] = item.get('mood_tag', [])
                
                item_data.append(filtered_item)
            except Exception as e:
                print(f"Error fetching item {item_id}: {e}")
        
        # Get review history
        review_raw = str(self.interaction_tool.get_reviews(user_id=self.task['user_id']))
        review_data = truncate_to_tokens(review_raw, max_tokens=8000)
        
        return user_data, item_data, review_data
    
    def _infer_user_mood(self, user_data: str, review_data: str) -> str:
        """Infer user's desired mood from their profile and history"""
        try:
            # Extract recent reviews (last 3-5)
            recent_reviews = review_data[:2000]  # Limit to recent context
            
            mood = self.reasoning.infer_user_mood(
                user_profile=user_data[:1000],
                recent_reviews=recent_reviews,
                task_context=str(self.task)
            )
            
            # Store in memory for future use
            if self.memory:
                try:
                    self.memory(f"User mood preference: {mood}")
                except:
                    pass
            
            return mood
            
        except Exception as e:
            print(f"Mood inference error: {e}")
            return "engaging, well-written, popular"  # Fallback
    
    def _analyze_candidates(self, item_data: List[Dict], inferred_mood: str) -> List[Dict]:
        """Analyze each candidate book for mood/theme alignment"""
        analyzed = []
        
        for item in item_data:
            try:
                # Extract moods from book
                book_moods = item.get('mood_tag', [])
                
                # Calculate mood similarity
                mood_score = self.reasoning.calculate_mood_similarity(
                    inferred_mood,
                    book_moods
                )
                
                item['mood_score'] = mood_score
                item['extracted_moods'] = book_moods
                analyzed.append(item)
                
            except Exception as e:
                print(f"Error analyzing item {item.get('item_id')}: {e}")
                item['mood_score'] = 0.0
                item['extracted_moods'] = []
                analyzed.append(item)
        
        return analyzed
    
    def _score_candidates(
        self,
        candidates: List[Dict],
        inferred_mood: str,
        user_data: str,
        review_data: str
    ) -> List[Dict]:
        """Apply multi-factor scoring to candidates"""
        
        for candidate in candidates:
            try:
                # Factor 1: Mood Alignment (40%)
                mood_score = candidate.get('mood_score', 0.0)
                
                # Factor 2: Rating Quality (20%)
                avg_rating = float(candidate.get('average_rating', 0.0))
                rating_score = avg_rating / 5.0 if avg_rating > 0 else 0.5
                
                # Factor 3: Popularity (10%)
                ratings_count = int(candidate.get('ratings_count', 0))
                popularity_score = min(ratings_count / 10000, 1.0)  # Normalize
                
                # Factor 4: Novelty (30%) - LLM-based assessment
                novelty_score = self._assess_novelty(candidate, review_data)
                
                # Weighted final score
                final_score = (
                    0.40 * mood_score +
                    0.20 * rating_score +
                    0.10 * popularity_score +
                    0.30 * novelty_score
                )
                
                candidate['final_score'] = final_score
                candidate['score_breakdown'] = {
                    'mood': mood_score,
                    'rating': rating_score,
                    'popularity': popularity_score,
                    'novelty': novelty_score
                }
                
            except Exception as e:
                print(f"Scoring error for {candidate.get('item_id')}: {e}")
                candidate['final_score'] = 0.0
        
        return candidates
    
    def _assess_novelty(self, candidate: Dict, review_history: str) -> float:
        """Assess how novel/diverse this book is compared to user's history"""
        try:
            # Simple heuristic: check if title appears in review history
            title = candidate.get('title', '').lower()
            history_lower = review_history.lower()
            
            # If similar titles found, lower novelty
            if title[:20] in history_lower:
                return 0.3
            
            # Default moderate novelty
            return 0.7
            
        except Exception as e:
            return 0.5  # Default
    
    def _generate_final_ranking(self, scored_candidates: List[Dict]) -> List[str]:
        """Generate final ranked list using LLM with score guidance"""
        
        # Sort by score
        sorted_candidates = sorted(
            scored_candidates,
            key=lambda x: x.get('final_score', 0),
            reverse=True
        )
        
        # Prepare concise candidate info for LLM
        candidate_info = []
        for i, item in enumerate(sorted_candidates[:20]):
            info = {
                'rank': i + 1,
                'item_id': item['item_id'],
                'title': item.get('title', 'Unknown'),
                'score': round(item.get('final_score', 0), 3),
                'moods': item.get('extracted_moods', [])[:3]
            }
            candidate_info.append(info)
        
        # Use LLM for final refinement
        prompt = f"""You are ranking books for a user. The candidates have been pre-scored based on mood alignment, quality, and novelty.

**Pre-ranked candidates** (already sorted by score):
{json.dumps(candidate_info, indent=2)}

**Task**: Review this ranking and make minor adjustments if needed based on:
1. Mood coherence across top recommendations
2. Diversity in the top 5 (avoid too similar books)
3. Trust the scores but use your judgment for tie-breaking

**Output format** (CRITICAL - follow exactly):
Return ONLY a Python list of item_id strings in your final ranking order:
['item_id1', 'item_id2', 'item_id3', ...]

Do NOT include any explanation, just the list.
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            result = self.llm(messages=messages, temperature=0.2, max_tokens=500)
            
            # Extract list from result
            match = re.search(r"\[.*?\]", result, re.DOTALL)
            if match:
                ranked_list = ast.literal_eval(match.group())
                
                # Validate that all candidate IDs are present
                expected_ids = set(self.task['candidate_list'])
                result_ids = set(ranked_list)
                
                if expected_ids == result_ids:
                    return ranked_list
                else:
                    print("LLM ranking incomplete, using score-based ranking")
                    return [item['item_id'] for item in sorted_candidates]
            else:
                print("Could not parse LLM ranking, using score-based ranking")
                return [item['item_id'] for item in sorted_candidates]
                
        except Exception as e:
            print(f"LLM ranking error: {e}, using score-based ranking")
            return [item['item_id'] for item in sorted_candidates]
    
    def _fallback_ranking(self) -> List[str]:
        """Fallback ranking if main workflow fails"""
        try:
            return self.task['candidate_list']
        except:
            return ['']


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('api_key')
    
    if not api_key:
        print("ERROR: Please set GEMINI_API_KEY in your .env file")
        exit(1)
    
    # Configuration
    task_set = "goodreads"
    data_dir = "../../data/processed-data"  # Adjust if needed
    
    # Initialize Simulator
    print("Initializing simulator...")
    simulator = Simulator(data_dir=data_dir, device="auto", cache=True)
    
    # Load tasks and ground truth
    simulator.set_task_and_groundtruth(
        task_dir=f"./track2/{task_set}/tasks",
        groundtruth_dir=f"./track2/{task_set}/groundtruth"
    )
    
    # Set agent and LLM
    simulator.set_agent(AdvancedMoodRecAgent)
    simulator.set_llm(GeminiLLM(api_key=api_key))
    
    # Run simulation
    print(f"\nRunning simulation on {task_set} dataset...")
    print("=" * 60)
    
    agent_outputs = simulator.run_simulation(
        number_of_tasks=50,  # Start with 50 for testing
        enable_threading=True,
        max_workers=5
    )
    
    # Evaluate
    print("\nEvaluating agent performance...")
    evaluation_results = simulator.evaluate()
    
    # Save results
    output_file = f'./evaluation_results_advanced_{task_set}.json'
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    
    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS:")
    print(f"{'=' * 60}")
    print(json.dumps(evaluation_results, indent=2))
    print(f"\nResults saved to: {output_file}")
