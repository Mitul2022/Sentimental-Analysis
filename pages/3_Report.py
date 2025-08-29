import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from auth.login import ensure_logged_in, logout_button
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
from collections import Counter
import random
import io
import base64
import spacy
from fpdf import FPDF
import plotly.io as pio
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import plotly.io as pio
import kaleido 

# =========================
# Load spaCy NLP Model (Preinstalled in requirements.txt)
# =========================
nlp = spacy.load("en_core_web_sm")

# =========================
# Load NLP Sentiment Model (Cached)
# =========================
@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)
    return sentiment_pipeline

sentiment_pipeline = load_sentiment_model()

# Function to analyze sentiment with confidence scores
def analyze_sentiment_with_scores(text):
    try:
        results = sentiment_pipeline(str(text))[0]
        scores = {res['label']: float(res['score']) for res in results}
        label_map = {"LABEL_2": "Positive", "LABEL_1": "Neutral", "LABEL_0": "Negative"}
        predicted_label = max(scores, key=scores.get)
        return label_map[predicted_label], scores
    except Exception:
        return "Neutral", {"LABEL_0": 0.33, "LABEL_1": 0.34, "LABEL_2": 0.33}

# =========================
# Calculate NPS from Confidence Scores
# =========================
def calculate_nps(df, pos_col="Positive_Score", neg_col="Negative_Score"):
    promoters = sum(df[pos_col] >= 0.7)   # High positive confidence
    detractors = sum(df[neg_col] >= 0.6)  # High negative confidence
    passives = len(df) - promoters - detractors
    total = len(df)
    if total == 0:
        return 0, 0, 0, 0
    promoters_pct = round((promoters / total) * 100, 1)
    passives_pct = round((passives / total) * 100, 1)
    detractors_pct = round((detractors / total) * 100, 1)
    nps_score = promoters_pct - detractors_pct
    return promoters_pct, passives_pct, detractors_pct, nps_score


# --- Global Configuration ---
NEGATIVE_ADJECTIVES = [
    "slow", "lagging", "unresponsive", "expensive", "overpriced", "confusing",
    "unhelpful", "rude", "bland", "cold", "difficult", "complicated",
    "unstable", "poor", "high", "limited", "small", "unreliable", "buggy",
    "bad", "long", "frustrating", "broken", "missing", "inaccurate", "unprofessional",
    "not good", "not efficient", "not working", "too slow", "too expensive", "insufficient",
    "crashes", "freezes", "unintuitive", "unclear", "unhappy", "unpleasant", "unfavorable"
]

UNIVERSAL_PROBLEMS = [
    "bug", "error", "crash", "glitch", "issue", "problem", "fee", "price",
    "delay", "size", "variety", "cost", "speed", "instability", "wait", "interface",
    "navigation", "design", "staff", "support", "team", "food", "product", "feature",
    "selection", "portions", "instructions", "documentation", "layout", "service", "response_time",
    "system", "app", "software", "processes", "infrastructure", "codebase", "server",
    "features", "value", "tiers", "workflow", "onboarding", "training", "protocols", "time", "quality",
    "internet", "network", "data", "storage", "resources", "updates", "communication", "transparency",
    "performance", "usability", "clarity", "reliability", "maintenance", "billing", "security",
    "experience", "content", "tool", "environment", "resources", "management"
]

ACTION_VERBS = [
    "investigate", "optimize", "improve", "review", "refine", "streamline",
    "update", "train", "standardize", "enhance", "address", "resolve",
    "evaluate", "reassess", "develop", "implement", "monitor", "adjust",
    "simplify", "clarify", "redesign", "ensure", "expand", "prioritize",
    "fix", "resolve", "reduce", "diversify", "audit", "restructure", "allocate", "reengineer", "communicate",
    "upgrade", "provide", "strengthen", "strategize", "debug", "test", "rework", "educate", "analyse", "assess",
    "innovate", "strategize", "cultivate"
]

PROBLEM_CATEGORIES = {
    "performance_efficiency": {
        "keywords": ["slow", "lagging", "unresponsive", "crash", "glitch", "delay", "unstable", "unreliable", "speed", "instability", "performance", "response", "stuttering", "freezing"],
        "spacy_patterns": [
            [{"LEMMA": {"IN": ["slow", "lag", "unresponsive", "poor", "bad"]}}, {"LOWER": {"IN": ["due", "because"]}}, {"LOWER": "to", "OP": "?"}, {"LOWER": {"IN": ["of", "to"]}}, {"POS": {"IN": ["NOUN", "PROPN", "ADJ"]}, "OP": "+", "DEP": {"IN": ["pobj", "attr", "amod", "nsubj"]}}],
            [{"LEMMA": {"IN": ["bug", "error", "crash", "glitch", "issue"]}}, {"LEMMA": {"IN": ["cause", "lead"]}}, {"LOWER": "to", "OP": "?"}, {"POS": {"IN": ["NOUN", "PROPN", "ADJ"]}, "OP": "+", "DEP": {"IN": ["dobj", "attr", "nsubj"]}}],
            [{"LEMMA": {"IN": ["unresponsive", "unstable", "bad", "poor", "slow"]}}, {"POS": {"IN": ["NOUN", "PROPN"]}, "DEP": {"IN": ["attr", "dobj", "nsubj", "pobj"]}}],
            [{"LEMMA": "performance"}, {"DEP": "amod", "OP": "?"}, {"LEMMA": {"IN": ["slow", "bad", "poor", "unreliable", "lagging", "unresponsive"]}}]
        ],
        "analysis_themes": {
            "performance_cause": "Users consistently report **{aspect_name}** performance being **{problem_adj}**, often attributed to underlying **{cause_or_effect_noun}**.",
            "problem_effect": "A prevalent concern is the **{problem_noun}** in **{aspect_name}** leading to **{cause_or_effect_noun}**, significantly impacting user experience.",
            "simple_adj_noun": "Feedback points to **{aspect_name}** suffering from a **{problem_adj} {problem_noun}**.",
            "simple_noun_adj": "The **{aspect_name}**'s **{problem_noun}** is frequently described as **{problem_adj}**, affecting overall **{aspect_name}** efficiency.",
            "general": "Users highlight general performance issues with **{aspect_name}**, describing it as **{problem_adj}**."
        },
        "recommendation_types": {
            "recommendation_phrases": [
                ("performance_cause", "1. **{fix_verb}** the **{cause_or_effect_noun}** identified as the root cause for **{aspect_name}**'s **{problem_adj}** performance.", ["optimizing code", "upgrading infrastructure", "improving network stability"]),
                ("problem_effect", "1. Implement immediate **{fix_verb}** for the **{problem_noun}** in **{aspect_name}** to prevent it from **{cause_or_effect_verb} {cause_or_effect_noun}**.", ["debugging crashes", "stabilizing glitches", "fixing errors"]),
                ("simple_adj_noun", "1. Prioritize **{fix_verb}** the **{problem_noun}** of **{aspect_name}** to enhance its **{problem_adj}** aspects and boost overall **{aspect_name}** reliability.", ["improving response times", "streamlining processes", "enhancing stability"]),
                ("simple_noun_adj", "1. Conduct a deep dive to **{fix_verb}** the underlying causes behind **{aspect_name}**'s **{problem_adj} {problem_noun}**.", ["investigate bottlenecks", "address resource consumption", "resolve system conflicts"]),
                (None, "2. **{fix_verb}** overall **{area_of_focus}** for **{aspect_name}** through continuous monitoring, load testing, and iterative improvements.", ["enhance", "ensure", "monitor"]),
                (None, "3. **{fix_verb}** resource allocation for **{aspect_name}**'s development and testing to prevent future performance bottlenecks.", ["allocate more", "increase funding", "review current"]),
                (None, "4. **{fix_verb}** a clear communication strategy regarding performance improvements and system stability for users.", ["establish", "maintain", "promote"])
            ]
        }
    },
    "cost_value": {
        "keywords": ["expensive", "overpriced", "pricy", "costly", "exorbitant", "fee", "price", "high", "cost", "value"],
        "spacy_patterns": [
            [{"LEMMA": {"IN": ["price", "cost", "fee"]}}, {"LOWER": {"IN": ["is", "are", "seems"]}}, {"LOWER": {"IN": ["too", "very", "quite"]}, "OP": "?"}, {"LEMMA": {"IN": ["high", "expensive", "overpriced"]}}],
            [{"LEMMA": {"IN": ["expensive", "high"]}}, {"LOWER": {"IN": ["for", "relative"]}}, {"LOWER": "to", "OP": "?"}, {"POS": "DET", "OP": "?"}, {"POS": {"IN": ["NOUN", "PROPN", "ADJ"]}, "OP": "+"}],
            [{"LOWER": {"IN": ["lack", "poor", "no"]}}, {"LOWER": "of"}, {"LEMMA": "value"}, {"LOWER": {"IN": ["for", "given"]}}, {"POS": "DET", "OP": "?"}, {"LEMMA": {"IN": ["price", "cost", "fee"]}}]
        ],
        "analysis_themes": {
            "simple_noun_adj": "The **{aspect_name}** is frequently criticized for its **{problem_noun}** being **{problem_adj}**, leading to user dissatisfaction.",
            "cost_value_comparison": "Many users find the **{problem_noun}** associated with **{aspect_name}** to be **{problem_adj}**, especially when considering the **{cause_or_effect_noun}** provided.",
            "lack_of_value": "Feedback highlights a **{problem_adj}** perception of **{aspect_name}** due to a **{problem_noun}** relative to the **{cause_or_effect_noun}**.",
            "general": "Users perceive **{aspect_name}** as **{problem_adj}**, indicating overall cost concerns and a potential mismatch with expectations."
        },
        "recommendation_types": {
            "recommendation_phrases": [
                ("cost_value_comparison", "1. **{fix_verb}** the **{problem_noun}** of **{aspect_name}** by **{random_action_1}** to better align with the perceived **{cause_or_effect_noun}**.", ["adjusting pricing tiers", "bundling more features", "revising subscription models"]),
                ("lack_of_value", "1. **{fix_verb}** the **{area_of_focus}** of **{aspect_name}** to justify its **{problem_noun}**, focusing on **{cause_or_effect_noun}** as a key differentiator.", ["enhancing premium features", "improving service inclusions", "clarifying benefits"]),
                ("simple_noun_adj", "1. **{fix_verb}** the **{problem_noun}** structure of **{aspect_name}** to address user concerns about its **{problem_adj}** nature.", ["revising the pricing model", "offering more flexible plans", "introducing tiered options"]),
                (None, "2. **{fix_verb}** a comprehensive competitive analysis of **{aspect_name}**'s pricing to ensure market alignment and perceived fairness.", ["conduct", "perform", "review"]),
                (None, "3. **{fix_verb}** the communication around **{aspect_name}**'s value proposition to clearly articulate benefits against its cost.", ["clarify", "strengthen", "refine"]),
                (None, "4. **{fix_verb}** loyalty programs or discounts to enhance long-term **{area_of_focus}** for existing users and attract new ones.", ["introduce", "expand", "evaluate"])
            ]
        }
    },
    "usability_complexity": {
        "keywords": ["confusing", "difficult", "complicated", "unclear", "complex", "hard", "clunky", "unintuitive", "frustrating"],
        "spacy_patterns": [
            [{"LEMMA": {"IN": ["interface", "navigation", "design", "workflow", "system"]}}, {"LEMMA": {"IN": ["be", "seem"]}, "OP": "?"}, {"LOWER": {"IN": ["too", "very", "quite"]}, "OP": "?"}, {"LEMMA": {"IN": ["confusing", "difficult", "complicated", "unclear", "clunky", "hard", "unintuitive", "frustrating"]}}],
            [{"LOWER": {"IN": ["hard"]}}, {"LOWER": "to"}, {"LEMMA": {"IN": ["use", "understand", "navigate"]}}, {"LOWER": {"IN": ["due", "because"]}}, {"LOWER": {"IN": ["to", "of"]}}, {"POS": "DET", "OP": "?"}, {"POS": {"IN": ["NOUN", "PROPN", "ADJ"]}, "OP": "+"}],
            [{"LEMMA": {"IN": ["unclear", "confusing", "complex"]}}, {"POS": {"IN": ["NOUN", "PROPN"]}, "DEP": {"IN": ["compound", "amod", "nsubj"]}}]
        ],
        "analysis_themes": {
            "simple_noun_adj": "The **{problem_noun}** within **{aspect_name}** is often described as **{problem_adj}**, causing frustration for users.",
            "usability_cause": "Users report a **{problem_adj}** experience when interacting with **{aspect_name}**, primarily due to the **{cause_or_effect_noun}**.",
            "simple_adj_noun": "Feedback highlights **{aspect_name}** having **{problem_adj} {problem_noun}**, making it challenging to use.",
            "general": "Many users find **{aspect_name}** to be **{problem_adj}**, hindering their overall experience and adoption."
        },
        "recommendation_types": {
            "recommendation_phrases": [
                ("usability_cause", "1. **{fix_verb}** the **{problem_noun}** of **{aspect_name}** by simplifying the **{cause_or_effect_noun}** to improve **{area_of_focus}**.", ["redesigning workflows", "streamlining navigation", "clarifying instructions"]),
                ("simple_noun_adj", "1. **{fix_verb}** the **{problem_noun}** design of **{aspect_name}** to make it less **{problem_adj}** and more intuitive.", ["simplify the interface", "refine the layout", "re-engineer the user flow"]),
                ("simple_adj_noun", "1. **{fix_verb}** clarity in **{aspect_name}**'s **{problem_noun}** to reduce user frustration caused by its **{problem_adj}** nature.", ["enhance user guides", "ensure clear labels", "improve error messages"]),
                (None, "2. **{fix_verb}** user testing and feedback loops to continuously improve the **{area_of_focus}** of **{aspect_name}**.", ["implement A/B testing", "strengthen usability studies", "expand beta programs"]),
                (None, "3. **{fix_verb}** comprehensive onboarding and in-app guidance to help users navigate **{aspect_name}** effectively.", ["develop interactive tutorials", "provide contextual help", "update documentation"]),
                (None, "4. **{fix_verb}** consistent design principles across **{aspect_name}** to reduce complexity and enhance user familiarity.", ["establish UX guidelines", "maintain design consistency", "enforce UI standards"])
            ]
        }
    },
    "service_quality": {
        "keywords": ["unhelpful", "rude", "poor_service", "unresponsive", "staff", "bad", "slow", "long", "wait", "customer support", "unprofessional"],
        "spacy_patterns": [
            [{"LEMMA": {"IN": ["staff", "support", "team", "service", "agents", "representatives"]}}, {"LEMMA": {"IN": ["be", "seem"]}, "OP": "?"}, {"LOWER": {"IN": ["too", "very", "quite"]}, "OP": "?"}, {"LEMMA": {"IN": ["unhelpful", "rude", "unresponsive", "slow", "unprofessional", "bad"]}}],
            [{"LEMMA": {"IN": ["long", "slow"]}}, {"LEMMA": {"IN": ["wait", "response", "time", "delays"]}}],
            [{"LOWER": "poor"}, {"LEMMA": {"IN": ["service", "support"]}}, {"LOWER": {"IN": ["from", "by"]}}, {"POS": "DET", "OP": "?"}, {"POS": {"IN": ["NOUN", "PROPN", "ADJ"]}, "OP": "+"}],
            [{"LEMMA": "customer"}, {"LEMMA": "service"}, {"LEMMA": {"IN": ["be", "seem"]}, "OP": "?"}, {"LEMMA": {"IN": ["unresponsive", "slow", "bad"]}}]
        ],
        "analysis_themes": {
            "simple_noun_adj": "Concerns are consistently raised about the **{problem_noun}** of **{aspect_name}** being **{problem_adj}**.",
            "simple_adj_noun": "Users frequently highlight **{aspect_name}**'s **{problem_adj} {problem_noun}** as a major issue, impacting their satisfaction.",
            "service_cause": "Users frequently report **{problem_adj}** interactions with **{aspect_name}**, often stemming from **{cause_or_effect_noun}**.",
            "general": "Feedback indicates **{aspect_name}** provides **{problem_adj}** service overall, suggesting systemic issues."
        },
        "recommendation_types": {
            "recommendation_phrases": [
                ("service_cause", "1. **{fix_verb}** **{aspect_name}**'s **{problem_noun}** by **{random_action_1}** to address the identified **{cause_or_effect_noun}**.", ["enhancing staff training", "increasing team size", "improving communication tools"]),
                ("simple_noun_adj", "1. **{fix_verb}** the **{problem_noun}** interactions to improve **{aspect_name}**'s overall **{problem_adj}** service quality.", ["standardize protocols", "provide ongoing training", "implement quality control checks"]),
                ("simple_adj_noun", "1. **{fix_verb}** the **{problem_adj} {problem_noun}** in **{aspect_name}** by **{random_action_1}** to boost user satisfaction.", ["optimizing response times", "improving communication channels", "streamlining support workflows"]),
                (None, "2. **{fix_verb}** a robust feedback system for **{aspect_name}**'s customer interactions to ensure continuous **{area_of_focus}**.", ["implement", "strengthen", "establish"]),
                (None, "3. **{fix_verb}** the empowerment of **{aspect_name}** staff to resolve issues efficiently and professionally, reducing the need for escalations.", ["enhance", "promote", "support"]),
                (None, "4. **{fix_verb}** communication with users about expected **{area_of_focus}** and any service improvements.", ["transparent", "clear", "proactive"])
            ]
        }
    },
    "product_content_quality": {
        "keywords": ["bland", "cold", "small", "limited", "poor", "lack of variety", "buggy", "bad", "missing", "broken", "quality", "unreliable", "inaccurate", "insufficient"],
        "spacy_patterns": [
            [{"LEMMA": {"IN": ["food", "product", "feature", "content", "information", "selection"]}}, {"LEMMA": {"IN": ["be", "seem"]}, "OP": "?"}, {"LOWER": {"IN": ["too", "very", "quite"]}, "OP": "?"}, {"LEMMA": {"IN": ["bland", "cold", "poor", "buggy", "broken", "missing", "inaccurate", "bad", "unreliable", "insufficient"]}}],
            [{"LEMMA": {"IN": ["limited", "small", "lack"]}}, {"LEMMA": {"IN": ["selection", "portion", "feature", "content", "option", "variety", "features"]}}],
            [{"LOWER": "lack"}, {"LOWER": "of"}, {"LEMMA": "variety"}, {"LOWER": {"IN": ["in", "for"]}}, {"POS": "DET", "OP": "?"}, {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}]
        ],
        "analysis_themes": {
            "simple_noun_adj": "Disappointment is frequently expressed regarding the **{problem_noun}** of **{aspect_name}** as it's often described as **{problem_adj}**, impacting user satisfaction.",
            "simple_adj_noun": "The overall quality of **{aspect_name}** is frequently criticized for its **{problem_adj} {problem_noun}**.",
            "lack_of_variety": "Users are dissatisfied with **{aspect_name}**'s offerings, citing a lack of variety in **{cause_or_effect_noun}**.",
            "general": "Feedback consistently rates **{aspect_name}** as **{problem_adj}**, indicating a fundamental quality issue that needs addressing."
        },
        "recommendation_types": {
            "recommendation_phrases": [
                ("lack_of_variety", "1. **{fix_verb}** the **{area_of_focus}** of **{aspect_name}** by **{random_action_1}** to address the perceived **{problem_adj}** selection.", ["expanding offerings", "diversifying content sources", "introducing new features"]),
                ("simple_noun_adj", "1. **{fix_verb}** the **{problem_noun}** quality of **{aspect_name}** to resolve its **{problem_adj}** characteristics.", ["implementing stricter QA", "refining content creation processes", "improving manufacturing standards"]),
                ("simple_adj_noun", "1. **{fix_verb}** the **{problem_adj} {problem_noun}** in **{aspect_name}** by **{random_action_1}** to meet user expectations.", ["ensuring consistency", "improving accuracy", "fixing defects"]),
                (None, "2. **{fix_verb}** continuous quality assurance processes for **{aspect_name}** to maintain high standards and address bugs promptly.", ["establish", "strengthen", "audit"]),
                (None, "3. **{fix_verb}** user feedback on **{aspect_name}**'s **{area_of_focus}** to guide future product enhancements and content development.", ["collect", "analyze", "integrate"]),
                (None, "4. **{fix_verb}** updates on new features or content additions to keep users informed and engaged with **{aspect_name}**.", ["regularly communicate", "proactively announce", "showcase"])
            ]
        }
    }
}

def generate_dynamic_content_universal(aspect_name, negative_contexts):
    aspect_display_name = aspect_name.replace('_', ' ').title()
    all_negative_text = " ".join(negative_contexts).lower()
    doc = nlp(all_negative_text)
    word_counts = Counter(token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.lemma_ != aspect_name.lower().replace('_', ' '))
    problem_words_in_context = [word for word in word_counts.keys() if word in NEGATIVE_ADJECTIVES or word in UNIVERSAL_PROBLEMS]
    problem_words_in_context.sort(key=lambda x: word_counts[x], reverse=True)
    potential_subject_nouns = [word for word in word_counts.keys() if word not in NEGATIVE_ADJECTIVES and word not in UNIVERSAL_PROBLEMS and not word.isnumeric() and len(word) > 2 and nlp(word)[0].pos_ == "NOUN"]
    potential_subject_nouns.sort(key=lambda x: word_counts[x], reverse=True)
    matcher = spacy.matcher.Matcher(nlp.vocab)
    identified_patterns = Counter()
    for category_name, category_data in PROBLEM_CATEGORIES.items():
        for i, spacy_pattern in enumerate(category_data["spacy_patterns"]):
            pattern_id = f"{category_name}_{i}"
            matcher.add(pattern_id, [spacy_pattern])
            matches = matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                problem_adj = None
                problem_noun = None
                cause_or_effect_noun = None
                pattern_analysis_key = "general"
                extracted_type_val = None
                if any(token.lemma_ in ["due", "because"] for token in span):
                    if category_name == "performance_efficiency": 
                        pattern_analysis_key = "performance_cause"
                    elif category_name == "usability_complexity": 
                        pattern_analysis_key = "usability_cause"
                    elif category_name == "service_quality": 
                        pattern_analysis_key = "service_cause"
                elif any(token.lemma_ in ["cause", "lead"] for token in span):
                    if category_name == "performance_efficiency": 
                        pattern_analysis_key = "problem_effect"
                elif "for what you get" in span.text.lower() or "for the price" in span.text.lower():
                    if category_name == "cost_value": 
                        pattern_analysis_key = "cost_value_comparison"
                elif "lack of value" in span.text.lower() or "poor value" in span.text.lower():
                    if category_name == "cost_value": 
                        pattern_analysis_key = "lack_of_value"
                elif "lack of variety" in span.text.lower():
                    if category_name == "product_content_quality": 
                        pattern_analysis_key = "lack_of_variety"
                elif any(adj.lemma_ in NEGATIVE_ADJECTIVES for adj in span if adj.pos_ == "ADJ"):
                    if any(pn.lemma_ in UNIVERSAL_PROBLEMS for pn in span if pn.pos_ == "NOUN"):
                        pattern_analysis_key = "simple_adj_noun"
                    elif any(pn.lemma_ == aspect_name.lower().replace('_', ' ') for pn in span):
                        pattern_analysis_key = "simple_noun_adj"
                for token in span:
                    if token.dep_ == "amod" and token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
                        problem_adj = token.lemma_
                        problem_noun = token.head.lemma_
                    elif token.pos_ == "ADJ" and token.lemma_ in NEGATIVE_ADJECTIVES and not problem_adj:
                        problem_adj = token.lemma_
                    elif token.pos_ == "NOUN" and token.lemma_ in UNIVERSAL_PROBLEMS and not problem_noun:
                        problem_noun = token.lemma_
                    elif token.lemma_ == aspect_name.lower().replace('_', ' ') and not problem_noun:
                        problem_noun = token.lemma_
                if not problem_noun:
                    problem_noun = aspect_name.lower().replace('_', ' ')
                for token in span:
                    if token.lemma_ in ["due", "because"]:
                        for child in token.children:
                            if child.dep_ == "pobj" and child.pos_ == "NOUN":
                                cause_or_effect_noun = child.lemma_
                                extracted_type_val = "cause"
                                break
                        if cause_or_effect_noun: break
                    elif token.lemma_ in ["cause", "lead"]:
                        for child in token.children:
                            if child.dep_ == "dobj" and child.pos_ == "NOUN":
                                cause_or_effect_noun = child.lemma_
                                extracted_type_val = "effect"
                                break
                        if cause_or_effect_noun: break
                if not problem_adj and problem_words_in_context:
                    for token in span:
                        if token.pos_ == "ADJ" and token.lemma_ in NEGATIVE_ADJECTIVES:
                            problem_adj = token.lemma_
                            break
                    if not problem_adj and problem_words_in_context:
                        problem_adj = problem_words_in_context[0]
                pattern_key = (category_name, pattern_analysis_key, problem_adj, problem_noun, cause_or_effect_noun, extracted_type_val)
                identified_patterns[pattern_key] += 1
            try:
                matcher.remove(pattern_id)
            except KeyError:
                pass
    sorted_identified_patterns = identified_patterns.most_common()
    analysis_lines = []
    recommendation_lines = []
    primary_pattern_info = None
    dominant_category = None
    if sorted_identified_patterns:
        primary_pattern_info = sorted_identified_patterns[0][0]
        if len(primary_pattern_info) != 6:
            primary_pattern_info = (None, "general", problem_words_in_context[0] if problem_words_in_context else "poor", aspect_name.lower().replace('_', ' '), None, None)
        dominant_category = primary_pattern_info[0]
        primary_analysis_key = primary_pattern_info[1]
        primary_problem_adj = primary_pattern_info[2]
        primary_problem_noun = primary_problem_noun = primary_pattern_info[3]
        primary_cause_or_effect_noun = primary_pattern_info[4]
        primary_extracted_type = primary_pattern_info[5]
        cause_or_effect_verb = "leads to" if primary_extracted_type == "effect" else "stems from"
        if dominant_category in PROBLEM_CATEGORIES:
            analysis_themes_dict = PROBLEM_CATEGORIES[dominant_category]["analysis_themes"]
        else:
            analysis_themes_dict = PROBLEM_CATEGORIES["product_content_quality"]["analysis_themes"]
        analysis_template_1 = analysis_themes_dict.get(primary_analysis_key, analysis_themes_dict["general"])
        analysis_lines.append(analysis_template_1.format(
            aspect_name=aspect_display_name,
            problem_adj=primary_problem_adj if primary_problem_adj else "unspecified negative",
            problem_noun=primary_problem_noun if primary_problem_noun else "issue",
            cause_or_effect_noun=primary_cause_or_effect_noun if primary_cause_or_effect_noun else "general factors",
            cause_or_effect_verb=cause_or_effect_verb
        ))
        secondary_patterns_added = 0
        used_analysis_themes = {primary_analysis_key}
        for idx in range(1, len(sorted_identified_patterns)):
            if secondary_patterns_added >= 2: break
            sec_pattern_info = sorted_identified_patterns[idx][0]
            if len(sec_pattern_info) != 6: continue
            sec_category, sec_analysis_key, sec_adj, sec_noun, sec_cause_or_effect, _ = sec_pattern_info
            if sec_analysis_key not in used_analysis_themes and sec_adj and sec_noun and sec_category == dominant_category:
                if sec_category in PROBLEM_CATEGORIES:
                    sec_analysis_template = PROBLEM_CATEGORIES[sec_category]["analysis_themes"].get(sec_analysis_key, "general")
                else:
                    sec_analysis_template = "general"
                secondary_analysis_text = sec_analysis_template.format(
                    aspect_name=aspect_display_name,
                    problem_adj=sec_adj if sec_adj else "unspecified negative",
                    problem_noun=sec_noun if sec_noun else "issue",
                    cause_or_effect_noun=sec_cause_or_effect if sec_cause_or_effect else 'general factors',
                    cause_or_effect_verb="leads to" if _ == "effect" else "stems from"
                ).replace(aspect_display_name + "'s ", '').replace(aspect_display_name + ' ', '').strip()
                if secondary_analysis_text not in analysis_lines[0]:
                    prefix = "Furthermore, " if secondary_patterns_added == 0 else "Additionally, "
                    analysis_lines.append(f"{prefix}users also frequently highlight: {secondary_analysis_text.lower()}.")
                    used_analysis_themes.add(sec_analysis_key)
                    secondary_patterns_added += 1
        if len(analysis_lines) < 3:
            total_negative_mentions = len(negative_contexts)
            analysis_lines.append(f"Overall, this indicates a significant area of concern for **{aspect_display_name}**, with {total_negative_mentions} negative mentions pointing to these issues.")
        if len(analysis_lines) < 4:
            analysis_lines.append(f"Addressing these identified challenges is crucial for improving overall user satisfaction and perception of **{aspect_display_name}**.")
    elif problem_words_in_context:
        primary_problem = problem_words_in_context[0]
        primary_subject = potential_subject_nouns[0] if potential_subject_nouns else "the overall experience"
        analysis_lines.append(f"The primary concern for **{aspect_display_name}** is its **{primary_problem}** aspect, often related to **{primary_subject}**.")
        analysis_lines.append("This suggests a need for deeper investigation into the specific instances mentioned by users.")
        analysis_lines.append(f"A thorough review of negative feedback for {aspect_display_name} is highly recommended.")
    else:
        analysis_lines.append(f"Negative feedback for **{aspect_name}** is diverse, without clear dominant problem phrases or keywords.")
        analysis_lines.append("A thorough manual review of comments is recommended to uncover the core problems.")
        analysis_lines.append("Consider implementing more structured feedback collection for this aspect.")
    if dominant_category and primary_pattern_info and len(primary_pattern_info) == 6:
        if dominant_category in PROBLEM_CATEGORIES:
            recommendation_phrases = PROBLEM_CATEGORIES[dominant_category]["recommendation_types"]["recommendation_phrases"]
        else:
            recommendation_phrases = PROBLEM_CATEGORIES["product_content_quality"]["recommendation_types"]["recommendation_phrases"]
        relevant_recs = [rec for rec in recommendation_phrases if rec[0] == primary_analysis_key]
        general_recs = [rec for rec in recommendation_phrases if rec[0] is None]
        used_rec_templates = set()
        if relevant_recs:
            _, current_rec_template, random_actions = random.choice(relevant_recs)
            fix_verb = random.choice(ACTION_VERBS).title()
            area_of_focus = random.choice(UNIVERSAL_PROBLEMS)
            random_action_1_phrase = random.choice(random_actions) if random_actions else ""
            try:
                line = current_rec_template.format(
                    fix_verb=fix_verb,
                    aspect_name=aspect_display_name,
                    problem_adj=primary_problem_adj if primary_problem_adj else "unspecified",
                    problem_noun=primary_problem_noun if primary_problem_noun else "issue",
                    cause_or_effect_noun=primary_cause_or_effect_noun if primary_cause_or_effect_noun else "factors",
                    area_of_focus=area_of_focus,
                    random_action_1=random_action_1_phrase
                )
                recommendation_lines.append(line)
                used_rec_templates.add(current_rec_template)
            except KeyError:
                pass
        added = len(recommendation_lines)
        for _ in range(4 - added):
            available = [r for r in general_recs if r[1] not in used_rec_templates]
            if not available: break
            _, template, actions = random.choice(available)
            fix_verb = random.choice(ACTION_VERBS).title()
            area_of_focus = random.choice(UNIVERSAL_PROBLEMS)
            action = random.choice(actions) if actions else "improve"
            try:
                line = template.format(
                    fix_verb=fix_verb,
                    aspect_name=aspect_display_name,
                    area_of_focus=area_of_focus,
                    random_action_1=action
                )
                recommendation_lines.append(line)
                used_rec_templates.add(template)
            except KeyError:
                continue
    else:
        recommendation_lines.extend([
            f"1. Conduct a focused root cause analysis on all negative feedback for **{aspect_display_name}**.",
            "2. Implement a structured feedback mechanism to gather actionable insights.",
            "3. Prioritize corrective actions based on frequency and severity.",
            "4. Engage with users who provided negative feedback.",
            f"5. {random.choice(ACTION_VERBS).title()} an internal workshop to brainstorm solutions."
        ])
    final_recommendation_lines = []
    for i, line in enumerate(recommendation_lines):
        stripped = re.sub(r'^\s*\d+\.\s*', '', line.strip())
        final_recommendation_lines.append(f"{i+1}. {stripped}")
    analysis_text = "\n".join(analysis_lines)
    recommendations_text = "\n".join(final_recommendation_lines)
    return f"""
**Analysis Overview:**
{analysis_text}
**Actionable Recommendations:**
{recommendations_text}
"""

# ----------------------------------------------------------------------
# Streamlit App UI and Logic Begins
# ----------------------------------------------------------------------
st.set_page_config(page_title="📊 Sentiment Report", layout="wide")
st.title("📋 Sentiment Analysis Report")

# Initialize session state
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "summary_data" not in st.session_state:
    st.session_state.summary_data = None
if "chart_data" not in st.session_state:
    st.session_state.chart_data = {}
if "negative_analysis_data" not in st.session_state:
    st.session_state.negative_analysis_data = {}

# Use sample data if not present
if st.session_state.processed_data is None or st.session_state.summary_data is None:
    st.warning("⚠️ No processed data found. Using sample data for demonstration.")
    dummy_data = {
        "Aspect": [
            "performance", "cost", "user_interface", "performance", "support", "cost",
            "food", "food", "staff", "menu", "food", "cost", "performance", "app", "service",
            "user_interface", "product", "support", "performance", "product_quality"
        ],
        "Aspect_Sentiment": [
            "Positive", "Negative", "Neutral", "Negative", "Negative", "Negative",
            "Negative", "Positive", "Negative", "Neutral", "Negative", "Negative", "Negative", "Negative", "Negative",
            "Negative", "Negative", "Negative", "Negative", "Negative"
        ],
        "Aspect_Context": [
            "The system is fast and reliable.",
            "The subscription cost is way too expensive, it's a huge fee.",
            "The UI is ok, not bad, not good.",
            "The performance is incredibly slow and it keeps crashing, a real bug.",
            "Customer support was unhelpful and slow to respond.",
            "The price is prohibitive and exorbitant for what you get.",
            "The food was bland and cold, very disappointing and poor quality.",
            "The taste was amazing, loved it!",
            "The staff seemed uninterested and rude, very poor service.",
            "The menu is a bit limited, lacking variety.",
            "The portion sizes were ridiculously small.",
            "The monthly fee is too high, it's overpriced.",
            "The app lags constantly and is very unstable because of server issues.",
            "The app has a lot of bugs and crashes frequently, leading to data loss.",
            "The service provided by the team was very poor and unhelpful due to lack of training.",
            "The user interface is confusing and difficult to navigate for new users.",
            "The product quality is bad, it broke after a week.",
            "Support response time is long and frustrating.",
            "Frequent freezes make the software unusable, a serious glitch.",
            "The content provided is inaccurate and insufficient."
        ]
    }
    st.session_state.processed_data = pd.DataFrame(dummy_data)
    overall_sentiments = []
    for _ in range(len(dummy_data["Aspect"])):
        overall_sentiments.append(random.choice(["Positive", "Negative", "Neutral"]))
    st.session_state.summary_data = pd.DataFrame({
        "Review": [f"Review {i+1}" for i in range(len(dummy_data["Aspect"]))],
        "Final_Sentiment": overall_sentiments
    })
    if st.session_state.summary_data[st.session_state.summary_data["Final_Sentiment"] == "Negative"].empty:
        st.session_state.summary_data.loc[0, "Final_Sentiment"] = "Negative"

processed_df = st.session_state.processed_data
summary_df = st.session_state.summary_data

if processed_df is None or summary_df is None:
    st.error("No processed data found. Please ensure data is loaded or processed.")
    st.stop()

# =========================
# Apply NLP Sentiment Scoring on Aspect_Context
# =========================
if "NLP_Sentiment" not in processed_df.columns:
    st.info("Applying NLP sentiment scoring...")
    results = processed_df["Aspect_Context"].astype(str).map(analyze_sentiment_with_scores)
    processed_df['NLP_Sentiment'] = results.apply(lambda x: x[0])
    processed_df['Positive_Score'] = results.apply(lambda x: x[1].get('LABEL_2', 0.0))
    processed_df['Neutral_Score'] = results.apply(lambda x: x[1].get('LABEL_1', 0.0))
    processed_df['Negative_Score'] = results.apply(lambda x: x[1].get('LABEL_0', 0.0))
    # Ensure float32 to avoid caching issues
    score_cols = ['Positive_Score', 'Neutral_Score', 'Negative_Score']
    processed_df[score_cols] = processed_df[score_cols].astype('float32')
    st.session_state.processed_data = processed_df

# =========================
# NLP & NPS Analysis Section
# =========================
st.header("🧠 NLP-Powered Sentiment & NPS Analysis")

# Average Confidence Scores - Donut Chart
avg_scores = {
    "Positive": processed_df["Positive_Score"].mean(),
    "Neutral": processed_df["Neutral_Score"].mean(),
    "Negative": processed_df["Negative_Score"].mean()
}
fig_avg = go.Figure(go.Pie(
    labels=list(avg_scores.keys()),
    values=list(avg_scores.values()),
    hole=0.5,
    marker_colors=['#4CAF50', "#DACC0D", '#F44336'],
    textinfo='percent',
    hoverinfo='label+value+percent'
))
fig_avg.update_layout(
    title="Average NLP Sentiment Confidence Scores",
    annotations=[dict(text="Avg<br>Scores", x=0.5, y=0.5, font_size=12, showarrow=False)]
)
st.plotly_chart(fig_avg, use_container_width=True)
st.session_state.chart_data['avg_sentiment_scores'] = fig_avg

# NPS Calculation
promoters_pct, passives_pct, detractors_pct, nps_score = calculate_nps(processed_df)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Promoters", f"{promoters_pct}%")
col2.metric("Passives", f"{passives_pct}%")
col3.metric("Detractors", f"{detractors_pct}%")
col4.metric("NPS Score", f"{int(nps_score)}")

# NPS Gauge
nps_fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=nps_score,
    title={'text': "Net Promoter Score"},
    gauge={
        'axis': {'range': [-100, 100]},
        'bar': {'color': "darkgreen" if nps_score > 70 else "green" if nps_score > 0 else "red"},
        'steps': [
            {'range': [-100, 0], 'color': "lightcoral"},
            {'range': [0, 30], 'color': "orange"},
            {'range': [30, 70], 'color': "lightgreen"},
            {'range': [70, 100], 'color': "darkgreen"}
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': nps_score
        }
    }
))
st.plotly_chart(nps_fig, use_container_width=True)
st.session_state.chart_data['nps_gauge'] = nps_fig

# AI Recommendations
if avg_scores["Negative"] > 0.4:
    st.warning("⚠️ High negative sentiment detected – investigate key complaints.")
if avg_scores["Neutral"] > 0.5:
    st.info("😐 Most responses are neutral – consider probing deeper with follow-up surveys.")
if avg_scores["Positive"] > 0.6:
    st.success("🎉 Strong positive sentiment – customers are satisfied.")
if nps_score < 0:
    st.error("🚨 NPS is negative. Immediate action needed.")
elif nps_score < 30:
    st.warning("⚡ NPS is low. Focus on converting passives.")
else:
    st.success("✅ Good NPS! Keep strengthening loyalty.")

# -------------------------------
# Rest of Your Original Report
# -------------------------------
st.header("📌 Key Performance Indicators (KPIs)")
total_reviews = summary_df.shape[0]
if total_reviews > 0:
    positive = summary_df[summary_df["Final_Sentiment"] == "Positive"].shape[0]
    neutral = summary_df[summary_df["Final_Sentiment"] == "Neutral"].shape[0]
    negative = summary_df[summary_df["Final_Sentiment"] == "Negative"].shape[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Total Reviews", total_reviews)
    col2.metric("😊 Positive", f"{positive} ({positive / total_reviews:.0%})")
    col3.metric("😐 Neutral", f"{neutral} ({neutral / total_reviews:.0%})")
    col4.metric("😠 Negative", f"{negative} ({negative / total_reviews:.0%})")
else:
    st.info("No reviews processed to display KPIs.")

    # Show total reviews
    st.metric("📄 Total Reviews", total_reviews)

# --- Aggregated Sentiment by Aspect (Stacked Bar) ---
st.header("Aggregated Sentiment by Aspect")
fig = None
if not processed_df.empty and "Aspect" in processed_df.columns and "Aspect_Sentiment" in processed_df.columns:
    filtered_df = processed_df[processed_df['Aspect'].astype(str).str.lower().isin(['undefined', 'nan', '']) == False]
    filtered_df['Aspect'] = filtered_df['Aspect'].astype(str).str.lower()
    top_10_aspects = filtered_df['Aspect'].value_counts().head(10).index
    aspect_sentiment_counts = filtered_df.groupby(["Aspect", "Aspect_Sentiment"]).size().reset_index(name="Count")
    aspect_sentiment_counts = aspect_sentiment_counts[aspect_sentiment_counts['Aspect'].isin(top_10_aspects)]
    if not aspect_sentiment_counts.empty:
        fig = px.bar(
            aspect_sentiment_counts,
            x="Aspect",
            y="Count",
            color="Aspect_Sentiment",
            title="Top 10 Aspect Sentiment Distribution",
            category_orders={"Aspect": top_10_aspects},
            color_discrete_map={"Positive": "green", "Neutral": "orange", "Negative": "red"},
            barmode="stack"
        )
        total_per_aspect = aspect_sentiment_counts.groupby("Aspect")["Count"].sum()
        for aspect in top_10_aspects:
            if aspect in total_per_aspect:
                fig.add_annotation(
                    x=aspect,
                    y=total_per_aspect[aspect] * 1.08,
                    text=str(total_per_aspect[aspect]),
                    showarrow=False,
                    font=dict(size=12),
                    xanchor="center"
                )
        fig.update_layout(
            xaxis_title="Aspect", 
            yaxis_title="Count", 
            showlegend=True,
            margin=dict(l=100, r=40, t=80, b=80)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.session_state.chart_data['aspect_sentiment_distribution'] = fig
    else:
        st.info("No data available for the top 10 aspects.")
else:
    st.info("No aspect or sentiment data available.")

# --- Top 5 Negative Aspects - Vertical Bar Chart ---
st.header("Top Negative Sentiment by Aspect")
if not processed_df.empty and "Aspect" in processed_df.columns and "Aspect_Sentiment" in processed_df.columns:
    # Filter only negative sentiments
    negative_df = processed_df[processed_df["Aspect_Sentiment"] == "Negative"].copy()
    negative_df["Aspect"] = negative_df["Aspect"].str.lower()

    if not negative_df.empty:
        # Count negative mentions per aspect
        neg_counts = negative_df["Aspect"].value_counts().head(5)
        neg_df = pd.DataFrame({
            "Aspect": [a.replace('_', ' ').title() for a in neg_counts.index],
            "Negative Mentions": neg_counts.values
        })

        # Create vertical bar chart
        fig_top5_neg = px.bar(
            neg_df,
            x="Aspect",
            y="Negative Mentions",
            text="Negative Mentions",
            color="Negative Mentions",
            color_continuous_scale="Reds",
            labels={"Negative Mentions": "Count", "Aspect": "Aspect"},
        )

        # Improve layout
        fig_top5_neg.update_layout(
            xaxis_title="Aspect",
            yaxis_title="Number of Negative Mentions",
            showlegend=False,
            margin=dict(l=100, r=40, t=80, b=80),
            coloraxis_colorbar=dict(title="Frequency")
        )

        st.plotly_chart(fig_top5_neg, use_container_width=True)
        st.session_state.chart_data['top_5_negative_vertical'] = fig_top5_neg
    else:
        st.info("No negative sentiment data available to display.")
else:
    st.info("Not enough data to generate 'Top 5 Negative Sentiment by Aspect' chart.")

# --- Aggregated Aspect Sentiment Breakdown - Styled Table (Percentages Only, No Count Columns) ---
st.header("📋 Aggregated Aspect Sentiment Breakdown")
pivot_df = None
if not processed_df.empty:
    # Create pivot table
    pivot_df = processed_df.pivot_table(
        index=processed_df['Aspect'].str.lower(),
        columns="Aspect_Sentiment",
        aggfunc="size",
        fill_value=0
    ).reset_index()
    for col in ["Positive", "Negative", "Neutral"]:
        if col not in pivot_df.columns:
            pivot_df[col] = 0
    total = pivot_df[["Positive", "Negative", "Neutral"]].sum(axis=1)
    pivot_df["Total_Mentions"] = total

    # Calculate percentages (as numeric, not strings)
    pivot_df["% Positive"] = (pivot_df["Positive"] / total) * 100
    pivot_df["% Neutral"] = (pivot_df["Neutral"] / total) * 100
    pivot_df["% Negative"] = (pivot_df["Negative"] / total) * 100

    # Sort by total mentions
    pivot_df = pivot_df.sort_values(by="Total_Mentions", ascending=False)

    # Prepare display DataFrame with only percentages and total
    display_df = pivot_df[["Aspect", "% Positive", "% Neutral", "% Negative", "Total_Mentions"]].copy()

    # Define styling: Apply gradient on numeric values, THEN format as %
    def style_percentage_table(df):
        return (df
                .style
                .background_gradient(subset=["% Positive"], cmap="Greens", vmin=0, vmax=100)
                .background_gradient(subset=["% Neutral"], cmap="YlOrBr", vmin=0, vmax=100)
                .background_gradient(subset=["% Negative"], cmap="Reds", vmin=0, vmax=100)
                .format({
                    "% Positive": "{:.1f}%",
                    "% Neutral": "{:.1f}%",
                    "% Negative": "{:.1f}%",
                    "Total_Mentions": "{:.0f}"
                })
                .set_properties(**{
                    'text-align': 'center',
                    'font-size': '13px'
                })
                .set_table_styles([
                    {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '14px')]},
                    {'selector': 'td', 'props': [('text-align', 'center')]}
                ])
                )

    # Apply styling
    styled_df = style_percentage_table(display_df)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Optional: Save for PDF or debugging
    st.session_state.chart_data['aspect_breakdown_table'] = display_df.copy()

else:
    st.info("No processed data available to generate the aggregated aspect sentiment table.")
# --- Word Cloud ---
st.header("☁️ Negative Context Word Cloud")
fig_wc_context = None
if not processed_df.empty and "Aspect_Context" in processed_df.columns and "Aspect_Sentiment" in processed_df.columns:
    neg_contexts = processed_df[processed_df["Aspect_Sentiment"] == "Negative"]["Aspect_Context"].dropna().tolist()
    if neg_contexts:
        context_text = " ".join(neg_contexts)
        wc_context = WordCloud(width=800, height=350, background_color='white', stopwords=STOPWORDS).generate(context_text)
        fig_wc_context, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(wc_context, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig_wc_context)
        st.session_state.chart_data['word_cloud'] = fig_wc_context
    else:
        st.info("No negative context data for word cloud.")
else:
    st.info("No negative context data for word cloud.")

# --- Enhanced Treemap ---
st.header("🌳 Aspect-Sentiment Treemap (Top 20 Aspects)")
fig_tree = None
if not processed_df.empty and "Aspect" in processed_df.columns and "Aspect_Sentiment" in processed_df.columns:
    valid_df = processed_df[processed_df['Aspect'].astype(str).str.lower().isin(['undefined', 'nan', 'none', '', ' ']) == False].copy()
    valid_df['Aspect'] = valid_df['Aspect'].astype(str).str.lower()
    top_20_aspects = valid_df['Aspect'].value_counts().head(20).index
    aspect_counts_for_tree = valid_df[valid_df['Aspect'].isin(top_20_aspects)].groupby(['Aspect', 'Aspect_Sentiment']).size().reset_index(name='Count')
    
    if not aspect_counts_for_tree.empty:
        # Add totals for hierarchy
        aspect_totals = aspect_counts_for_tree.groupby('Aspect')['Count'].sum().reset_index()
        aspect_totals['Aspect_Sentiment'] = 'Total'
        aspect_counts_for_tree = pd.concat([aspect_counts_for_tree, aspect_totals], ignore_index=True)

        fig_tree = px.treemap(
            aspect_counts_for_tree,
            path=['Aspect', 'Aspect_Sentiment'],
            values='Count',
            color='Aspect_Sentiment',
            color_discrete_map={'Positive': '#4CAF50', 'Negative': '#F44336', 'Neutral': '#FFC107', 'Total': '#9E9E9E'},
            title="Top 20 Aspects: Sentiment Distribution",
            hover_data={'Count': True}
        )
        fig_tree.update_traces(
            texttemplate="<b>%{label}</b><br>%{value} mentions",
            textfont_size=12,
            selector=dict(type='treemap')
        )
        fig_tree.update_layout(
            margin=dict(t=80, l=40, r=40, b=40),
            title_font_size=16
        )
        st.plotly_chart(fig_tree, use_container_width=True)
        st.session_state.chart_data['treemap'] = fig_tree
    else:
        st.info("No valid data available for treemap visualization after filtering out 'undefined'.")
else:
    st.info("Data unavailable for treemap visualization.")

# --- Smart N-Gram Analysis (Ascending to Descending) ---
st.header("🧠 Relevant Phrase Analysis from Negative Feedback")
if not processed_df.empty and "Aspect_Context" in processed_df.columns and "Aspect_Sentiment" in processed_df.columns:
    def extract_ngrams_from_text(text, n):
        if pd.isna(text): return []
        cleaned = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        doc = nlp(cleaned)
        tokens = [token.text for token in doc if not token.is_stop and token.is_alpha and len(token.text) > 1]
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    negative_contexts = processed_df[processed_df["Aspect_Sentiment"] == "Negative"]["Aspect_Context"].astype(str).tolist()
    if not negative_contexts:
        st.info("No negative feedback available for phrase analysis.")
    else:
        three_word_phrases = []
        four_word_phrases = []

        for context in negative_contexts:
            three_word_phrases.extend(extract_ngrams_from_text(context, 3))
            four_word_phrases.extend(extract_ngrams_from_text(context, 4))

        # Get frequency counts
        three_word_count = Counter(three_word_phrases)
        four_word_count = Counter(four_word_phrases)

        # Convert to list of (phrase, count) and sort ASC → DESC
        three_word_sorted = sorted(three_word_count.items(), key=lambda x: x[1])[:5]   # lowest 5
        three_word_sorted += sorted(three_word_count.items(), key=lambda x: x[1], reverse=True)[:5]  # top 5

        four_word_sorted = sorted(four_word_count.items(), key=lambda x: x[1])[:5]  
        four_word_sorted += sorted(four_word_count.items(), key=lambda x: x[1], reverse=True)[:5]  

        # 3-Word Phrases
        st.subheader("3-Word Phrases (Ascending → Descending)")
        if three_word_sorted:
            df_3w = pd.DataFrame([{"Phrase": phrase, "Count": count} for phrase, count in three_word_sorted])
            fig_3w = px.bar(
                df_3w, 
                x="Count", 
                y="Phrase", 
                orientation='h',
                text="Count", 
                color="Count",  
                color_continuous_scale="Reds", 
                title="Negative 3-Word Phrases (Low → High Frequency)"
            )
            fig_3w.update_layout(
                xaxis_title="Frequency", 
                yaxis_title="Phrase", 
                margin=dict(l=150, r=40, t=80, b=40),
                coloraxis_colorbar=dict(title="Frequency")
            )
            st.plotly_chart(fig_3w, use_container_width=True)
            st.session_state.chart_data['smart_trigrams'] = fig_3w
        else:
            st.info("No 3-word phrases found.")

        # 4-Word Phrases
        st.subheader("4-Word Phrases (Ascending → Descending)")
        if four_word_sorted:
            df_4w = pd.DataFrame([{"Phrase": phrase, "Count": count} for phrase, count in four_word_sorted])
            fig_4w = px.bar(
                df_4w, 
                x="Count", 
                y="Phrase", 
                orientation='h',
                text="Count", 
                color="Count",  
                color_continuous_scale="Reds",
                title="Negative 4-Word Phrases (Low → High Frequency)"
            )
            fig_4w.update_layout(
                xaxis_title="Frequency", 
                yaxis_title="Phrase", 
                margin=dict(l=150, r=40, t=80, b=40),
                coloraxis_colorbar=dict(title="Frequency")
            )
            st.plotly_chart(fig_4w, use_container_width=True)
            st.session_state.chart_data['smart_fourgrams'] = fig_4w
        else:
            st.info("No 4-word phrases found.")
else:
    st.info("No data available for phrase analysis.")

# --- Detailed Negative Aspect Analysis ---
st.header("🔍 Detailed Aspect Analysis & Recommendations (Negative Focus)")
if not processed_df.empty and "Aspect_Sentiment" in processed_df.columns:
    negative_aspects_df = processed_df[processed_df["Aspect_Sentiment"] == "Negative"].copy()
    negative_aspects_df["Aspect"] = negative_aspects_df["Aspect"].astype(str).str.lower()
    if not negative_aspects_df.empty:
        top_negative_aspect_counts = negative_aspects_df["Aspect"].value_counts().head(5)
        st.session_state.negative_analysis_data = {}
        if not top_negative_aspect_counts.empty:
            for i, (aspect, count) in enumerate(top_negative_aspect_counts.items()):
                st.markdown(f"### **{i+1}. Aspect: {aspect.replace('_', ' ').title()}**")
                aspect_negative_contexts = negative_aspects_df[negative_aspects_df["Aspect"] == aspect]["Aspect_Context"].unique().tolist()
                st.write(f"  **Negative Mentions Count:** {count}")
                st.write("  **Sample Negative Feedback:**")
                for j, context in enumerate(aspect_negative_contexts[:3]):
                    st.markdown(f"  - _\"{context}\"_")
                if len(aspect_negative_contexts) > 3:
                    st.markdown(f"  _({len(aspect_negative_contexts) - 3} more negative comments...)_")
                st.write("**Analysis & Actionable Recommendations:**")
                dynamic_output = generate_dynamic_content_universal(aspect, aspect_negative_contexts)
                st.session_state.negative_analysis_data[aspect] = dynamic_output
                st.markdown(dynamic_output)
                st.markdown("---")
        else:
            st.info("No top negative aspects found to analyze in detail.")
    else:
        st.info("No negative sentiment data available in the processed reviews.")
else:
    st.info("No processed data available to perform detailed aspect analysis.")

from fpdf import FPDF
import io
import plotly.io as pio
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

def generate_pdf_report(chart_data, analysis_data, summary_data, processed_df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Title Page ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(0, 10, "Sentiment Analysis Report", 0, 1, 'C')
    pdf.ln(15)

    # --- KPIs ---
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Key Performance Indicators (KPIs)", 0, 1, 'L')
    pdf.ln(5)

    if summary_data is not None and not summary_data.empty:
        total_reviews = summary_data.shape[0]
        if total_reviews > 0:
            positive = summary_data[summary_data["Final_Sentiment"] == "Positive"].shape[0]
            neutral = summary_data[summary_data["Final_Sentiment"] == "Neutral"].shape[0]
            negative = summary_data[summary_data["Final_Sentiment"] == "Negative"].shape[0]

            pdf.set_font("Arial", 'B', 12)
            col_width = pdf.w / 4 - 10
            pdf.cell(col_width, 8, "Total Reviews", 1, 0, 'C')
            pdf.cell(col_width, 8, "Positive", 1, 0, 'C')
            pdf.cell(col_width, 8, "Neutral", 1, 0, 'C')
            pdf.cell(col_width, 8, "Negative", 1, 1, 'C')

            pdf.set_font("Arial", '', 12)
            pdf.cell(col_width, 8, str(total_reviews), 1, 0, 'C')
            pdf.cell(col_width, 8, f"{positive} ({positive/total_reviews:.0%})", 1, 0, 'C')
            pdf.cell(col_width, 8, f"{neutral} ({neutral/total_reviews:.0%})", 1, 0, 'C')
            pdf.cell(col_width, 8, f"{negative} ({negative/total_reviews:.0%})", 1, 1, 'C')
            pdf.ln(10)

    # --- Top Aspects ---
    if processed_df is not None and not processed_df.empty:
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Top 10 Aspects Sentiment Breakdown", 0, 1, 'L')
        pdf.ln(5)

        pivot_df = processed_df.pivot_table(
            index=processed_df['Aspect'].str.lower(),
            columns="Aspect_Sentiment",
            aggfunc="size",
            fill_value=0
        ).reset_index()

        for col in ["Positive", "Negative", "Neutral"]:
            if col not in pivot_df.columns:
                pivot_df[col] = 0

        pivot_df["Total_Mentions"] = pivot_df[["Positive", "Negative", "Neutral"]].sum(axis=1)
        top_10_df = pivot_df.sort_values(by="Total_Mentions", ascending=False).head(10)

        if not top_10_df.empty:
            pdf.set_font("Arial", 'B', 10)
            col_widths = [40, 30, 30, 30, 30]
            pdf.cell(col_widths[0], 7, "Aspect", 1, 0, 'C')
            pdf.cell(col_widths[1], 7, "Total Mentions", 1, 0, 'C')
            pdf.cell(col_widths[2], 7, "Positive", 1, 0, 'C')
            pdf.cell(col_widths[3], 7, "Neutral", 1, 0, 'C')
            pdf.cell(col_widths[4], 7, "Negative", 1, 1, 'C')

            pdf.set_font("Arial", '', 10)
            for _, row in top_10_df.iterrows():
                pdf.cell(col_widths[0], 7, row['Aspect'].replace('_',' ').title(), 1, 0, 'L')
                pdf.cell(col_widths[1], 7, str(row['Total_Mentions']), 1, 0, 'C')
                pdf.cell(col_widths[2], 7, str(row['Positive']), 1, 0, 'C')
                pdf.cell(col_widths[3], 7, str(row['Neutral']), 1, 0, 'C')
                pdf.cell(col_widths[4], 7, str(row['Negative']), 1, 1, 'C')
            pdf.ln(10)

    # --- Charts Section ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Visual Data Summary", 0, 1, 'L')
    pdf.ln(5)

    def add_chart_to_pdf(chart_key, title):
        fig = chart_data.get(chart_key)
        if fig is None:
            pdf.set_font("Arial", '', 10)
            pdf.cell(0, 10, f"{title}: Not Available", 0, 1)
            return

        try:
            buf = io.BytesIO()
            if isinstance(fig, go.Figure):
                pio.write_image(fig, buf, format='png', width=700, height=400, scale=2)
            elif isinstance(fig, plt.Figure):
                fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, title, 0, 1, 'L')
            pdf.image(buf, x=10, w=180)
            pdf.ln(8)
        except Exception as e:
            print(f"⚠️ Chart export failed for {chart_key}: {e}")
            pdf.set_font("Arial", '', 10)
            pdf.cell(0, 10, f"{title}: Failed to render", 0, 1)

    # Add all charts safely
    for key, label in [
        ('avg_sentiment_scores', "Average NLP Sentiment Scores"),
        ('nps_gauge', "Net Promoter Score (NPS)"),
        ('aspect_sentiment_distribution', "Aggregated Sentiment by Aspect"),
        ('top_negative', "Top 3 Most Negative Aspects"),
        ('top_positive', "Top 3 Most Positive Aspects"),
        ('top_neutral', "Top 3 Most Neutral Aspects"),
        ('word_cloud', "Negative Context Word Cloud"),
        ('treemap', "Aspect-Sentiment Treemap (Top 20 Aspects)"),
        ('aspect_heatmap', "Aspect Sentiment Heatmap"),
        ('smart_bigrams', "Top 5 Relevant 2-Word Phrases"),
        ('smart_trigrams', "Top 5 Relevant 3-Word Phrases"),
        ('smart_fourgrams', "Top 5 Relevant 4-Word Phrases"),
    ]:
        add_chart_to_pdf(key, label)

    # --- Recommendations Section ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Detailed Analysis & Recommendations", 0, 1, 'L')
    pdf.ln(5)

    if analysis_data:
        for aspect, content in analysis_data.items():
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, f"Aspect: {aspect.replace('_', ' ').title()}", 0, 1, 'L')
            pdf.set_font("Arial", '', 10)
            pdf.multi_cell(0, 5, content, align='L')
            pdf.ln(5)

    return bytes(pdf.output(dest='S'))

# --- Streamlit Button ---
st.markdown("---")
if st.button("Generate & Download PDF Report"):
    with st.spinner("Generating PDF..."):
        pdf_bytes = generate_pdf_report(
            st.session_state.chart_data,
            st.session_state.negative_analysis_data,
            st.session_state.summary_data,
            processed_df
        )
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name="sentiment_analysis_report.pdf",
            mime="application/pdf",
        )

