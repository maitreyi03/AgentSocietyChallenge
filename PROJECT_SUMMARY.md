# 📚 Advanced Mood-Based Recommendation Agent - Project Summary

## 🎯 Project Overview

This project implements an **Advanced Mood and Theme-Based Recommendation Agent** for the WWW'25 AgentSociety Challenge (Goodreads Track). The agent significantly enhances the baseline implementation through sophisticated mood inference, multi-factor scoring, and semantic matching.

## 📊 Performance Goals

**Current Baseline Performance**:
- HR@1: 11.00%
- HR@3: 29.25%
- HR@5: 44.75%
- Average: 28.33%

**Target Improvements**:
- HR@1: 15-18% (+36-64% improvement)
- HR@3: 35-40% (+20-37% improvement)
- HR@5: 50-55% (+12-23% improvement)
- Average: 35-40% (+24-41% improvement)

## 🏗️ What Was Built

### 1. Core Agent (`advanced_mood_agent.py`)
Full-featured agent with:
- ✅ Multi-stage recommendation workflow
- ✅ Advanced mood inference from user history
- ✅ Semantic mood/theme matching
- ✅ Multi-factor scoring (mood 40%, theme 30%, rating 20%, novelty 10%)
- ✅ Memory integration for long-term learning
- ✅ Robust error handling and fallback strategies
- ✅ Detailed logging and progress tracking

### 2. Submission Version (`submission_agent.py`)
Competition-ready version:
- ✅ Clean code without main block
- ✅ All features from advanced agent
- ✅ Ready to zip and submit
- ✅ Follows competition format

### 3. Easy Runner (`run_agent.py`)
User-friendly execution script:
- ✅ Environment validation
- ✅ Command-line arguments
- ✅ Progress tracking
- ✅ Automatic comparison with baseline
- ✅ Pretty-printed results

### 4. Documentation
Comprehensive guides:
- ✅ `QUICKSTART.md` - 5-minute setup guide
- ✅ `AGENT_DOCUMENTATION.md` - Full technical documentation
- ✅ `PROJECT_SUMMARY.md` - This file
- ✅ `.env.example` - API key template

## 🔑 Key Innovations

### 1. **Intelligent Mood Inference**
```python
# Analyzes user's reading history to infer desired mood
inferred_mood = "cozy, fall-themed, heartwarming, slow-paced"
```

Instead of generic recommendations, the agent:
- Analyzes review sentiment and patterns
- Detects thematic preferences (fall-themed, cozy, etc.)
- Identifies pacing preferences (fast-paced vs slow-burn)
- Considers emotional tones (uplifting, dark, romantic)

### 2. **Multi-Factor Scoring System**

| Factor | Weight | Purpose |
|--------|--------|---------|
| Mood Alignment | 40% | Matches user's emotional preferences |
| Theme Relevance | 30% | Captures seasonal/atmospheric preferences |
| Rating Quality | 20% | Ensures high-quality recommendations |
| Novelty | 10% | Balances familiar and new discoveries |

### 3. **Semantic Matching**
Uses keyword overlap and semantic similarity to match:
- User mood → Book mood tags
- Reading history → Candidate books
- Thematic elements → Book descriptions

### 4. **Memory Integration**
Learns user preferences over time:
- Stores mood preferences
- Builds long-term user profiles
- Adapts recommendations to patterns

### 5. **LLM-Guided Refinement**
Final ranking uses LLM to:
- Ensure diversity in top 5
- Resolve tie-breaks intelligently
- Balance mood coherence and variety

## 📁 File Structure

```
AgentSocietyChallenge-1/
├── advanced_mood_agent.py       # Full agent with evaluation
├── submission_agent.py          # Competition submission version
├── run_agent.py                 # Easy runner script ⭐ Use this!
├── moodRecAgent.py             # Your groupmate's original work
│
├── .env                         # Your API key (add your key here!)
├── .env.example                # Template
│
├── QUICKSTART.md               # 5-minute setup guide ⭐ Start here!
├── AGENT_DOCUMENTATION.md      # Full technical docs
├── PROJECT_SUMMARY.md          # This file
│
├── example/track2/goodreads/
│   ├── tasks/                  # Test tasks
│   └── groundtruth/            # Expected results
│
└── evaluation_results_*.json   # Results after running
```

## 🚀 Quick Start (3 Steps)

### Step 1: Add Your API Key
```bash
# Edit .env file
nano .env

# Add your Gemini API key:
GEMINI_API_KEY=AIzaSy...your_actual_key
```

Get free API key: https://aistudio.google.com/app/apikey

### Step 2: Install Dependencies (if needed)
```bash
poetry install && poetry shell
# OR
pip install -r requirements.txt
```

### Step 3: Run!
```bash
# Quick test (10 tasks)
python run_agent.py --tasks 10 --workers 2

# Medium test (50 tasks)
python run_agent.py --tasks 50 --workers 5

# Full evaluation (400 tasks)
python run_agent.py --tasks 400 --workers 10
```

## 📈 Expected Performance Improvements

Based on the enhanced architecture, you should see:

**Mood Alignment**: 
- Better matching on themed requests (cozy, fall-themed)
- More accurate inference of unstated preferences
- Improved handling of emotional tone preferences

**Ranking Quality**:
- Top-1 improvement from better mood inference
- Top-3 improvement from multi-factor scoring
- Top-5 improvement from diversity balancing

**Robustness**:
- Handles edge cases gracefully
- Falls back intelligently when data is sparse
- Memory learns from evaluation runs

## 🎯 How to Optimize Further

### 1. Tune Scoring Weights
Edit `advanced_mood_agent.py`, line ~290:
```python
final_score = (
    0.50 * mood_score +      # Try increasing mood weight
    0.20 * rating_score +
    0.10 * popularity_score +
    0.20 * novelty_score     # Adjust for more/less diversity
)
```

### 2. Adjust Temperature
```python
# Mood inference (line ~120)
temperature=0.7  # Higher = more creative, Lower = more consistent

# Final ranking (line ~350)
temperature=0.2  # Higher = more diverse, Lower = more score-based
```

### 3. Enhance Mood Tags
If you have access to book metadata, pre-compute mood tags:
```python
# Add custom mood extraction logic
book_moods = extract_moods_from_description(book['description'])
```

### 4. Add Collaborative Filtering
Enhance novelty scoring with user-user similarity:
```python
# Find similar users and see what they liked
similar_users = find_similar_users(user_history)
cf_score = calculate_cf_score(book, similar_users)
```

## 📤 Submission Process

### For Development Phase

1. **Test locally**:
```bash
python run_agent.py --tasks 50 --workers 5
```

2. **Review results**:
```bash
cat evaluation_results_advanced_goodreads.json
```

3. **Create submission**:
```bash
# Use submission_agent.py (already prepared)
zip submission.zip submission_agent.py
```

4. **Submit**:
- Go to: https://tsinghua-fib-lab.github.io/AgentSocietyChallenge/pages/recommendation-track.html
- Upload `submission.zip`
- Select "Recommendation Track"

### For Final Phase

After reaching Top 20:
- Same process, but evaluation will be on 60% real data
- Ensure agent is robust to different data distributions
- Consider ensemble methods for better generalization

## 🔍 Monitoring Your Agent

### Watch Console Output
```
Stage 1: Collecting user and item data...
Stage 2: Inferring user's desired mood...
Inferred Mood: cozy, fall-themed, heartwarming, slow-paced  ← Should be specific
Stage 3: Analyzing candidate books...
Stage 4: Applying multi-factor scoring...
Stage 5: Generating final ranking...
Top 5 recommendations: ['827260', '534180', ...]  ← Check diversity
```

### Good Signs ✅
- Varied, specific mood inferences
- Mood scores distributed across range
- Top 5 shows diverse books
- Hit rates improving over baseline

### Warning Signs ⚠️
- Generic moods every time
- All mood scores near 0
- Same books always top-ranked
- Hit rates not improving

## 🎓 Key Learnings

### What Makes This Agent Different

1. **Context-Aware**: Understands user mood from behavior, not just ratings
2. **Theme-Focused**: Specializes in mood/theme preferences (cozy, fall-themed, etc.)
3. **Multi-Dimensional**: Balances mood, quality, and novelty
4. **Adaptive**: Learns preferences through memory module
5. **Robust**: Handles errors gracefully with smart fallbacks

### Comparison to Baseline

| Feature | Baseline | Advanced Agent |
|---------|----------|----------------|
| Mood Inference | None | LLM-based analysis |
| Scoring Factors | 1 (generic) | 4 (multi-factor) |
| Theme Matching | Basic | Semantic similarity |
| Memory | None | Long-term learning |
| Error Handling | Basic | Multi-level fallbacks |
| Explainability | Low | High (staged process) |

## 🔧 Troubleshooting

### Common Issues & Solutions

**Issue**: Low hit rates (< baseline)
- **Solution**: Check if mood tags exist in data
- **Solution**: Increase mood weight to 50%
- **Solution**: Review inferred moods (are they too generic?)

**Issue**: Memory errors
- **Solution**: Reduce max_workers to 1-2
- **Solution**: Reduce token limits (8000 → 4000)

**Issue**: Slow execution
- **Solution**: Reduce workers (affects speed, not quality)
- **Solution**: Reduce tasks for testing
- **Solution**: Check Gemini API rate limits

**Issue**: API errors
- **Solution**: Verify API key is correct
- **Solution**: Check quota at Google Cloud Console
- **Solution**: Add retry logic if needed

## 📞 Getting Help

1. **Quick Setup**: Read `QUICKSTART.md`
2. **Technical Details**: Read `AGENT_DOCUMENTATION.md`
3. **Competition Rules**: Visit AgentSociety Challenge website
4. **Code Issues**: Check the baseline examples in `example/`

## 🎉 Success Metrics

You'll know the agent is working well when:

- ✅ Hit rates exceed baseline by 20%+
- ✅ Inferred moods are specific and varied
- ✅ Top recommendations feel coherent
- ✅ Agent handles edge cases without crashing
- ✅ Evaluation completes successfully

## 🚀 Next Steps

1. **Immediate**: Add API key and run quick test (10 tasks)
2. **Short-term**: Run medium evaluation (50 tasks), analyze results
3. **Medium-term**: Tune parameters, re-evaluate
4. **Long-term**: Add advanced features (embeddings, CF)
5. **Submit**: Create submission when satisfied with results

---

## 📝 Final Notes

This agent builds significantly upon your groupmate's work by:
- Adding sophisticated mood inference
- Implementing multi-factor scoring
- Enhancing with memory and learning
- Providing robust error handling
- Including comprehensive documentation

**Remember**: The goal is not just higher hit rates, but **meaningful** recommendations that align with user moods and themes!

Good luck with the competition! 🎊

---

**Ready to start?**
```bash
# 1. Add your API key to .env
# 2. Run:
python run_agent.py --tasks 10 --workers 2
```
