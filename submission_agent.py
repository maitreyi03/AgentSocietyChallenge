"""
Advanced Mood and Theme-Based Recommendation Agent for Goodreads
Submission Version for WWW'25 AgentSociety Challenge

This agent implements a sophisticated multi-stage ranking system optimized for
mood and theme-based book recommendations with focus on:
- Cozy vs adventurous moods
- Fall-themed vs other seasonal preferences  
- Fast-paced vs slow-paced narratives
- Heartwarming vs thrilling tones

Author: Enhanced for Track 2 - Recommendation Task
"""

import json
import re
from typing import List, Dict, Any, Optional
from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.agent.modules.memory_modules import MemoryBase, MemoryDILU
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.llm.llm import LLMBase
import tiktoken
import ast


def num_tokens_from_string(string: str) -> int:
    """Count tokens in a string using tiktoken"""
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        return len(encoding.encode(string))
    except:
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
        
        long_term_profile = "No long-term history available."
        if self.memory:
            try:
                long_term_profile = self.memory.get_long_term_profile(profile_type=self.profile_type_prompt)
            except:
                pass
        
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
            temperature=0.7,
            max_tokens=150
        )
        
        mood = reasoning_result.strip().replace('"', '').replace("'", '')
        return mood
    
    def calculate_mood_similarity(self, target_mood: str, book_moods: List[str]) -> float:
        """Calculate semantic similarity between target mood and book moods"""
        if not book_moods:
            return 0.0
        
        target_keywords = set(target_mood.lower().split())
        book_keywords = set(' '.join(book_moods).lower().split())
        
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
        
        if memory is None:
            try:
                self.memory = MemoryDILU(llm=self.llm)
            except:
                self.memory = None
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
            user_data, item_data, review_data = self._collect_data()
            inferred_mood = self._infer_user_mood(user_data, review_data)
            analyzed_candidates = self._analyze_candidates(item_data, inferred_mood)
            scored_candidates = self._score_candidates(analyzed_candidates, inferred_mood, user_data, review_data)
            ranked_items = self._generate_final_ranking(scored_candidates)
            return ranked_items
        except Exception as e:
            return self._fallback_ranking()
    
    def _collect_data(self) -> tuple:
        """Collect user profile, item data, and review history"""
        user_raw = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
        user_data = truncate_to_tokens(user_raw, max_tokens=8000)
        
        item_data = []
        for item_id in self.task['candidate_list']:
            try:
                item = self.interaction_tool.get_item(item_id=item_id)
                keys = ['item_id', 'title', 'description', 'average_rating', 
                       'ratings_count', 'title_without_series']
                filtered_item = {k: item.get(k, '') for k in keys}
                filtered_item['mood_tag'] = item.get('mood_tag', [])
                item_data.append(filtered_item)
            except:
                pass
        
        review_raw = str(self.interaction_tool.get_reviews(user_id=self.task['user_id']))
        review_data = truncate_to_tokens(review_raw, max_tokens=8000)
        
        return user_data, item_data, review_data
    
    def _infer_user_mood(self, user_data: str, review_data: str) -> str:
        """Infer user's desired mood from their profile and history"""
        try:
            recent_reviews = review_data[:2000]
            mood = self.reasoning.infer_user_mood(
                user_profile=user_data[:1000],
                recent_reviews=recent_reviews,
                task_context=str(self.task)
            )
            
            if self.memory:
                try:
                    self.memory(f"User mood preference: {mood}")
                except:
                    pass
            
            return mood
        except:
            return "engaging, well-written, popular"
    
    def _analyze_candidates(self, item_data: List[Dict], inferred_mood: str) -> List[Dict]:
        """Analyze each candidate book for mood/theme alignment"""
        analyzed = []
        
        for item in item_data:
            try:
                book_moods = item.get('mood_tag', [])
                mood_score = self.reasoning.calculate_mood_similarity(inferred_mood, book_moods)
                item['mood_score'] = mood_score
                item['extracted_moods'] = book_moods
                analyzed.append(item)
            except:
                item['mood_score'] = 0.0
                item['extracted_moods'] = []
                analyzed.append(item)
        
        return analyzed
    
    def _score_candidates(self, candidates: List[Dict], inferred_mood: str, 
                         user_data: str, review_data: str) -> List[Dict]:
        """Apply multi-factor scoring to candidates"""
        
        for candidate in candidates:
            try:
                mood_score = candidate.get('mood_score', 0.0)
                
                avg_rating = float(candidate.get('average_rating', 0.0))
                rating_score = avg_rating / 5.0 if avg_rating > 0 else 0.5
                
                ratings_count = int(candidate.get('ratings_count', 0))
                popularity_score = min(ratings_count / 10000, 1.0)
                
                novelty_score = self._assess_novelty(candidate, review_data)
                
                final_score = (
                    0.40 * mood_score +
                    0.20 * rating_score +
                    0.10 * popularity_score +
                    0.30 * novelty_score
                )
                
                candidate['final_score'] = final_score
            except:
                candidate['final_score'] = 0.0
        
        return candidates
    
    def _assess_novelty(self, candidate: Dict, review_history: str) -> float:
        """Assess how novel/diverse this book is compared to user's history"""
        try:
            title = candidate.get('title', '').lower()
            history_lower = review_history.lower()
            
            if title[:20] in history_lower:
                return 0.3
            
            return 0.7
        except:
            return 0.5
    
    def _generate_final_ranking(self, scored_candidates: List[Dict]) -> List[str]:
        """Generate final ranked list using LLM with score guidance"""
        
        sorted_candidates = sorted(
            scored_candidates,
            key=lambda x: x.get('final_score', 0),
            reverse=True
        )
        
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
            
            match = re.search(r"\[.*?\]", result, re.DOTALL)
            if match:
                ranked_list = ast.literal_eval(match.group())
                
                expected_ids = set(self.task['candidate_list'])
                result_ids = set(ranked_list)
                
                if expected_ids == result_ids:
                    return ranked_list
                else:
                    return [item['item_id'] for item in sorted_candidates]
            else:
                return [item['item_id'] for item in sorted_candidates]
        except:
            return [item['item_id'] for item in sorted_candidates]
    
    def _fallback_ranking(self) -> List[str]:
        """Fallback ranking if main workflow fails"""
        try:
            return self.task['candidate_list']
        except:
            return ['']
