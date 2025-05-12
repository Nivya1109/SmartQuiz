import re
import random
from typing import List, Optional

def generate_questions(context, num_questions=5, fixed_questions=None):
    """Generate questions from context or use fixed questions if provided."""
    if fixed_questions:
        return fixed_questions
    # Clean up context
    context = re.sub(r'[.,:;]+\s*', '. ', context)  # Normalize punctuation
    context = re.sub(r'\s+', ' ', context)  # Normalize whitespace
    context = context.strip()
    
    # Use T5's question generation task
    prompts = [
        f"generate question: {context}",
        f"ask about: {context}",
        f"question about: {context}"
    ]
    
    questions = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(
            input_ids=input_ids,
            max_length=64,
            min_length=10,
            num_return_sequences=2,
            num_beams=4,
            temperature=0.7,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
        # Decode and clean up the generated questions
        for output in outputs:
            question = tokenizer.decode(output, skip_special_tokens=True)
            question = question.strip()
            
            # Add question mark if missing
            if not question.endswith('?'):
                question += '?'
            
            # Add to questions list if it passes cleaning
            cleaned = clean_question(question)
            if cleaned:
                questions.append(cleaned)
    
    # Remove duplicates and return
    return list(dict.fromkeys(questions[:num_questions]))

def clean_question(text):
    if not text:
        return None
    
    # Basic cleanup
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    # Must be a reasonable length
    if len(text.split()) < 4 or len(text.split()) > 20:
        return None
    
    # Must start with appropriate question words
    valid_starts = ['what is', 'what are', 'how does', 'how do', 'why is', 'why are']
    if not any(text.lower().startswith(start) for start in valid_starts):
        return None
    
    # Must end with question mark
    if not text.endswith('?'):
        text += '?'
    
    # Filter out unwanted patterns
    unwanted = ['generate', 'question', 'following', 'context', 'true', 'false',
                'example', 'answer', 'format', 'line', 'number', 'task']
    if any(word in text.lower() for word in unwanted):
        return None
    
    # Capitalize first letter
    text = text[0].upper() + text[1:]
    
    return text

def remove_duplicates(questions):
    seen = set()
    filtered = []
    for q in questions:
        if q:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                filtered.append(q)
    return filtered

def extract_key_phrases(text: str) -> List[str]:
    # Clean up text
    text = re.sub(r'\([^)]*\)', '', text)  # Remove parenthetical content
    text = re.sub(r'\?+', '', text)  # Remove question marks
    text = re.sub(r'[.,:;]+\s*', '. ', text)  # Normalize punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    
    # Technical terms to preserve
    tech_terms = {
        'ai': 'AI',
        'artificial intelligence': 'Artificial Intelligence',
        'machine learning': 'Machine Learning',
        'deep learning': 'Deep Learning',
        'neural network': 'Neural Network',
        'natural language processing': 'Natural Language Processing',
        'nlp': 'NLP',
        'computer vision': 'Computer Vision',
        'reinforcement learning': 'Reinforcement Learning',
        'supervised learning': 'Supervised Learning',
        'unsupervised learning': 'Unsupervised Learning',
        'data science': 'Data Science',
        'big data': 'Big Data',
        'data mining': 'Data Mining',
        'data analytics': 'Data Analytics'
    }
    
    # Normalize technical terms
    for term_lower, term_proper in tech_terms.items():
        text = re.sub(r'\b' + term_lower + r'\b', term_proper, text, flags=re.IGNORECASE)
    
    # Split into sentences and handle bullet points/numbered lists
    segments = []
    for sentence in re.split(r'[.!?]+', text):
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Handle bullet points and numbered lists
        items = re.split(r'(?:^|\n)\s*(?:\d+\.|•|-)\s*', sentence)
        for item in items:
            item = item.strip()
            if not item:
                continue
            
            # Remove numbers at start of item
            item = re.sub(r'^\d+\.?\s*', '', item)
            
            # Split on conjunctions while preserving technical terms
            parts = []
            current_part = []
            words = item.split()
            i = 0
            while i < len(words):
                word = words[i]
                # Check for multi-word technical terms
                found_term = False
                for term in sorted(tech_terms.keys(), key=len, reverse=True):
                    term_words = term.split()
                    if i + len(term_words) <= len(words):
                        potential_term = ' '.join(words[i:i+len(term_words)]).lower()
                        if potential_term == term:
                            if current_part:
                                parts.append(' '.join(current_part))
                                current_part = []
                            parts.append(tech_terms[term])
                            i += len(term_words)
                            found_term = True
                            break
                
                if found_term:
                    continue
                
                # Handle regular words
                if word.lower() in ['and', 'or', 'but']:
                    if current_part:
                        parts.append(' '.join(current_part))
                        current_part = []
                elif word.endswith(',') or word.endswith(';'):
                    current_part.append(word[:-1])
                    if current_part:
                        parts.append(' '.join(current_part))
                        current_part = []
                else:
                    current_part.append(word)
                i += 1
            
            if current_part:
                parts.append(' '.join(current_part))
            
            # Process each part
            for part in parts:
                # Clean up
                phrase = part.strip()
                if not phrase:
                    continue
                
                # Remove leading articles
                phrase = re.sub(r'^(the|a|an)\s+', '', phrase, flags=re.IGNORECASE)
                
                # Must be reasonable length
                words = phrase.split()
                if len(words) < 2 or len(words) > 10:
                    continue
                
                # Must contain at least one meaningful word
                if not any(len(word) > 3 for word in words):
                    continue
                    
                # Must not be generic/useless
                lower_phrase = phrase.lower()
                if any(word in lower_phrase for word in [
                    "context", "true", "false", "yes", "no", "following",
                    "example", "next", "previous", "above", "below"
                ]):
                    continue
                    
                # Must not be just a list of stop words
                stop_words = {
                    "the", "a", "an", "and", "or", "but", "in", "on", "at",
                    "to", "for", "of", "with", "by", "as", "from", "into",
                    "during", "including", "until", "against", "among",
                    "throughout", "despite", "towards", "upon", "concerning",
                    "about", "like", "through", "over", "before", "between",
                    "after", "since", "without", "under", "within", "along",
                    "following", "across", "behind", "beyond", "plus",
                    "except", "but", "up", "out", "around", "down", "off",
                    "above", "below"
                }
                content_words = [w for w in words if w.lower() not in stop_words]
                if not content_words:
                    continue
                
                segments.append(phrase)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_phrases = []
    for phrase in segments:
        lower_phrase = phrase.lower()
        if lower_phrase not in seen:
            seen.add(lower_phrase)
            unique_phrases.append(phrase)
    
    return unique_phrases

def generate_questions(context: str, num_questions: int = 3) -> List[str]:
    # Clean up context
    context = re.sub(r'[.,:;]+\s*', '. ', context)  # Normalize punctuation
    context = re.sub(r'\s+', ' ', context)  # Normalize whitespace
    context = context.strip()
    
    print("\nContext:", context)
    
    # Extract key phrases
    phrases = extract_key_phrases(context)
    print("\nKey phrases:", phrases)
    
    # Question templates for different types
    templates = {
        'concept': [
            "What is {}?",
            "Can you explain what {} means?",
            "What does {} refer to?"
        ],
        'process': [
            "How does {} work?",
            "Can you explain how {} operates?",
            "What is the process of {}?"
        ],
        'purpose': [
            "What is the purpose of {}?",
            "Why is {} important?",
            "What are the benefits of {}?"
        ],
        'process': [
            "How does {} work?",
            "Explain the process of {}.",
            "How is {} implemented?"
        ],
        'components': [
            "What are the key components of {}?",
            "What elements make up {}?",
            "What are the main parts of {}?"
        ],
        'application': [
            "How is {} used in practice?",
            "What are the applications of {}?",
            "Give examples of how {} is applied."
        ],
        'comparison': [
            "How does {} differ from other approaches?",
            "Compare and contrast {} with alternatives.",
            "What makes {} unique?"
        ],
        'example': [
            "Can you give an example of {}?",
            "What is a practical application of {}?",
            "How is {} used in the real world?"
        ],
        'impact': [
            "What impact does {} have?",
            "How does {} affect the industry?",
            "What role does {} play?"
        ]
    }
    
    # Keywords that suggest question types
    type_indicators = {
        'concept': ['is', 'refers to', 'means', 'defined as', 'called', 'known as'],
        'process': ['works by', 'functions', 'operates', 'processes', 'steps', 'how'],
        'purpose': ['purpose', 'goal', 'benefit', 'important', 'helps', 'enables', 'allows', 'used for'],
        'components': ['consists of', 'contains', 'includes', 'parts', 'elements', 'components', 'features'],
        'application': ['used in', 'applied to', 'implemented', 'examples', 'applications', 'practices'],
        'comparison': ['compared to', 'versus', 'unlike', 'different from', 'better than', 'advantages'],
        'example': ['example', 'instance', 'case', 'scenario', 'such as', 'like'],
        'impact': ['impact', 'effect', 'influence', 'role', 'significance', 'importance']
    }
    
    # Technical terms that should get special treatment
    tech_terms = {
        'ai': ['purpose', 'application', 'impact'],
        'artificial intelligence': ['purpose', 'application', 'impact'],
        'machine learning': ['process', 'application', 'example'],
        'deep learning': ['process', 'application', 'example'],
        'neural network': ['components', 'process', 'example'],
        'computer vision': ['application', 'example', 'impact'],
        'nlp': ['application', 'example', 'impact'],
        'reinforcement learning': ['process', 'application', 'example']
    }
    
    questions = []
    seen_phrases = set()
    
    for phrase in phrases:
        # Skip if too short or too long
        words = phrase.split()
        if len(words) < 2 or len(words) > 10:
            continue
        
        # Skip if we've seen this phrase
        phrase_lower = phrase.lower()
        if phrase_lower in seen_phrases:
            continue
        seen_phrases.add(phrase_lower)
        
        # Skip if phrase starts with question words
        if any(phrase_lower.startswith(word) for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            continue
        
        # Determine appropriate question types
        question_types = set()
        
        # Check if it's a technical term
        for term, types in tech_terms.items():
            if term in phrase_lower:
                question_types.update(types)
        
        # Check context for type indicators
        context_lower = context.lower()
        phrase_pos = context_lower.find(phrase_lower)
        if phrase_pos >= 0:
            # Look at text around the phrase
            start = max(0, phrase_pos - 50)
            end = min(len(context_lower), phrase_pos + len(phrase) + 50)
            surrounding = context_lower[start:end]
            
            for qtype, indicators in type_indicators.items():
                if any(indicator in surrounding for indicator in indicators):
                    question_types.add(qtype)
        
        # Always include concept questions for new terms
        if not any(term in phrase_lower for term in tech_terms):
            question_types.add('concept')
        
        # Generate questions for each appropriate type
        successful_questions = []
        for qtype in question_types:
            for template in templates[qtype]:
                question = template.format(phrase)
                
                # Try to generate an answer
                answer = generate_answer(question, context)
                if answer:
                    # Check answer quality
                    answer_words = answer.split()
                    if len(answer_words) >= 5 and not answer.lower().startswith(phrase_lower):
                        successful_questions.append((question, len(answer_words)))
                        
                if len(successful_questions) >= 3:  # Limit questions per phrase
                    break
            
            if len(successful_questions) >= 3:
                break
        
        # Sort by answer length (longer answers often better)
        successful_questions.sort(key=lambda x: x[1], reverse=True)
        questions.extend(q[0] for q in successful_questions[:2])  # Take top 2
        
        if len(questions) >= num_questions * 2:  # Generate extra for filtering
            break
    
    print(f"Generated {len(questions)} raw questions")
    
    # Score and rank questions
    scored_questions = []
    for question in questions:
        score = 0
        
        # Prefer questions about technical terms
        if any(term in question.lower() for term in tech_terms):
            score += 2
        
        # Prefer questions with good answers
        answer = generate_answer(question, context)
        if answer:
            # Longer answers usually better
            score += min(len(answer.split()) / 5, 3)
            
            # Boost score if answer contains technical terms
            if any(term in answer.lower() for term in tech_terms):
                score += 1
        
        scored_questions.append((question, score))
    
    # Sort by score and remove duplicates
    scored_questions.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    unique_questions = []
    for question, _ in scored_questions:
        lower_q = question.lower()
        if lower_q not in seen:
            seen.add(lower_q)
            unique_questions.append(question)
            if len(unique_questions) >= num_questions:
                break
    
    print(f"After filtering: {len(unique_questions)} questions\n")
    return unique_questions


def generate_answer(question: str, context: str) -> Optional[str]:
    # Clean up context
    context = re.sub(r'[.,:;]+\s*', '. ', context)  # Normalize punctuation
    context = re.sub(r'\s+', ' ', context)  # Normalize whitespace
    context = context.strip()
    
    # Extract key concept and question type
    question = question.lower()
    
    # Determine question type and extract concept
    if any(phrase in question for phrase in ['what is', 'what does', 'what do we mean', 'can you explain what']):
        question_type = 'concept'
        concept = re.sub(r'^.*?(?:is|does|mean by|explain what)\s+(?:the|a|an)?\s*', '', question)
    elif any(phrase in question for phrase in ['how does', 'how is', 'explain how']):
        question_type = 'process'
        concept = re.sub(r'^.*?(?:does|is|how)\s+(?:the|a|an)?\s*', '', question)
    elif any(phrase in question for phrase in ['why is', 'what is the purpose', 'what are the benefits']):
        question_type = 'purpose'
        concept = re.sub(r'^.*?(?:is|the purpose of|benefits of)\s+(?:the|a|an)?\s*', '', question)
    elif any(phrase in question for phrase in ['what are the key', 'what elements', 'what are the main parts']):
        question_type = 'components'
        concept = re.sub(r'^.*?(?:of|make up|in)\s+(?:the|a|an)?\s*', '', question)
    elif any(phrase in question for phrase in ['how is', 'what are the applications', 'how can']):
        question_type = 'application'
        concept = re.sub(r'^.*?(?:is|are|can)\s+(?:the|a|an)?\s*', '', question)
    else:
        question_type = 'concept'
        concept = re.sub(r'^.*?(?:is|does|mean by|explain)\s+(?:the|a|an)?\s*', '', question)
    
    # Remove question mark and clean up concept
    concept = re.sub(r'\?+$', '', concept).strip()
    
    # Look for relevant sentences in context
    relevant_sentences = []
    for sentence in re.split(r'[.!?]+', context):
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Score sentence relevance
        score = 0
        sentence_lower = sentence.lower()
        concept_lower = concept.lower()
        
        # Direct mention of concept
        if concept_lower in sentence_lower:
            score += 3
        
        # Contains key words from concept
        concept_words = set(concept_lower.split())
        sentence_words = set(sentence_lower.split())
        common_words = concept_words & sentence_words
        score += len(common_words)
        
        # Contains question type indicators
        if question_type == 'concept' and any(word in sentence_lower for word in ['is', 'refers to', 'means', 'defined as']):
            score += 2
        elif question_type == 'process' and any(word in sentence_lower for word in ['works by', 'functions', 'operates', 'steps']):
            score += 2
        elif question_type == 'purpose' and any(word in sentence_lower for word in ['purpose', 'goal', 'benefit', 'importance']):
            score += 2
        elif question_type == 'components' and any(word in sentence_lower for word in ['consists of', 'contains', 'includes', 'parts']):
            score += 2
        elif question_type == 'application' and any(word in sentence_lower for word in ['used in', 'applied to', 'implemented', 'example']):
            score += 2
        
        if score > 0:
            relevant_sentences.append((sentence, score))
    
    if not relevant_sentences:
        return None
    
    # Sort by relevance score
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    # Combine top sentences into answer
    answer_parts = [s[0] for s in relevant_sentences[:2]]
    answer = ' '.join(answer_parts)
    
    # Clean up answer
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    # Remove leading articles and numbers
    answer = re.sub(r'^(the|a|an|\d+\.?)\s+', '', answer, flags=re.IGNORECASE)
    
    # Ensure answer is reasonable length
    if len(answer.split()) < 3 or len(answer.split()) > 50:
        return None
    
    # Check for generic/useless answers
    lower_answer = answer.lower()
    if any(phrase in lower_answer for phrase in [
        "this is", "that is", "these are", "those are",
        "example", "following", "above", "below", "next", "previous"
    ]):
        return None
    
    # Format answer nicely
    # Capitalize first letter
    answer = answer[0].upper() + answer[1:]
    
    # Ensure proper punctuation
    if not answer[-1] in ['.', '!', '?']:
        answer += '.'
    
    return answer
