"""
Direct Answer Generation Pipeline
Forces the agent to actually ANSWER questions directly with specifics
"""

import re
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class DirectAnswerCore:
    """Mandatory: Always answer the question directly first"""
    
    def __init__(self):
        self.answer_required_patterns = {
            "what is": ("definition", "Provide definition with specifics"),
            "what are": ("enumeration", "List items with details"),
            "who is": ("identity", "Name the person/entity with details"),
            "who are": ("identity_plural", "Name the people/entities with details"),
            "when did": ("temporal", "Give date/time with context"),
            "when was": ("temporal", "Give date/time with context"),
            "where is": ("location", "Provide location with details"),
            "where are": ("location_plural", "Provide locations with details"),
            "how many": ("quantity", "Give number with unit"),
            "how much": ("quantity", "Give amount with unit"),
            "how do": ("procedural", "Explain process step by step"),
            "how does": ("mechanism", "Explain how it works"),
            "which": ("selection", "Identify with specifics"),
            "why does": ("causal", "Explain cause with evidence"),
            "why do": ("causal", "Explain cause with evidence"),
            "why is": ("causal", "Explain reason with evidence"),
            "what causes": ("causal", "Explain cause with evidence"),
            "best": ("recommendation", "Provide ranked recommendations"),
            "top": ("ranking", "Provide ranked list with details"),
            "largest": ("superlative", "Identify the largest with specifics"),
            "biggest": ("superlative", "Identify the largest with specifics"),
            "smallest": ("superlative", "Identify the smallest with specifics"),
            "most": ("superlative", "Identify the most with specifics")
        }
        
        # Templates for forcing direct answers
        self.answer_templates = {
            "definition": "**{subject}** is {definition}.",
            "enumeration": "The main {subject} are:\n{list}",
            "identity": "**{name}** is {description}.",
            "temporal": "This occurred on **{date}**{context}.",
            "location": "**{subject}** is located in {location}.",
            "quantity": "There are **{number} {unit}**{context}.",
            "procedural": "To {action}:\n{steps}",
            "mechanism": "{subject} works by {mechanism}.",
            "selection": "The {criteria} is **{selection}**.",
            "causal": "{effect} occurs because {cause}.",
            "recommendation": "Based on {criteria}, the best {subject} are:\n{recommendations}",
            "ranking": "Based on {data_source}, the top {subject} are:\n{ranking}",
            "superlative": "The {superlative} {subject} is **{answer}**{details}."
        }
    
    def generate_direct_answer(self, question: str, research_data: Dict, 
                                facts: Dict = None) -> str:
        """Generate an answer that actually answers the question"""
        
        # 1. Classify question type
        q_type, instruction = self._classify_question(question)
        
        # 2. Extract the subject/topic from question
        subject = self._extract_subject(question, q_type)
        
        # 3. Get answer template
        template = self.answer_templates.get(q_type, "{answer}")
        
        # 4. Extract or construct facts
        if not facts:
            facts = self._extract_relevant_facts(research_data, question, q_type)
        
        # 5. If no facts available, acknowledge but still structure properly
        if not facts.get("primary_answer"):
            return self._construct_no_info_response(question, subject, q_type)
        
        # 6. Fill template with facts
        answer = self._fill_template(template, facts, subject, q_type)
        
        # 7. Ensure directness - remove any vague language
        answer = self._ensure_directness(answer, question)
        
        return answer
    
    def _classify_question(self, question: str) -> Tuple[str, str]:
        """Classify the question type and get answer instruction"""
        question_lower = question.lower().strip()
        
        for pattern, (q_type, instruction) in self.answer_required_patterns.items():
            if pattern in question_lower:
                return q_type, instruction
        
        # Check for question mark - general question
        if "?" in question:
            return "general", "Provide a direct, informative answer"
        
        return "statement", "Respond appropriately"
    
    def _extract_subject(self, question: str, q_type: str) -> str:
        """Extract the main subject of the question"""
        question_lower = question.lower()
        
        # Remove question words and markers
        patterns_to_remove = [
            r"^what is the\s+", r"^what is\s+", r"^what are the\s+", r"^what are\s+",
            r"^who is the\s+", r"^who is\s+", r"^who are the\s+", r"^who are\s+",
            r"^where is the\s+", r"^where is\s+", r"^where are the\s+",
            r"^when did the\s+", r"^when did\s+", r"^when was the\s+",
            r"^how many\s+", r"^how much\s+", r"^how do\s+", r"^how does\s+",
            r"^why does\s+", r"^why do\s+", r"^why is\s+",
            r"^which\s+", r"^the\s+", r"\?$"
        ]
        
        subject = question_lower
        for pattern in patterns_to_remove:
            subject = re.sub(pattern, "", subject, flags=re.IGNORECASE)
        
        # Clean up
        subject = subject.strip().rstrip("?").strip()
        
        # Capitalize for display
        if subject:
            subject = subject[0].upper() + subject[1:]
        
        return subject if subject else "this topic"
    
    def _extract_relevant_facts(self, research_data: Dict, question: str, 
                                  q_type: str) -> Dict:
        """Extract facts relevant to the question from research data"""
        facts = {
            "primary_answer": "",
            "supporting_details": [],
            "numbers": [],
            "names": [],
            "dates": [],
            "locations": [],
            "sources": [],
            "definition": "",
            "list": "",
            "ranking": "",
            "mechanism": "",
            "cause": "",
            "effect": ""
        }
        
        # Get summary from research
        summary = research_data.get("summary", "")
        if not summary:
            return facts
        
        # Extract entities from summary
        facts["names"] = self._extract_proper_nouns(summary)
        facts["numbers"] = self._extract_numbers(summary)
        facts["dates"] = self._extract_dates(summary)
        
        # Extract primary answer based on question type
        if q_type in ["definition", "mechanism"]:
            facts["primary_answer"] = self._extract_definition(summary)
            facts["definition"] = facts["primary_answer"]
        
        elif q_type in ["superlative", "ranking"]:
            if facts["names"]:
                facts["primary_answer"] = facts["names"][0]
                if facts["numbers"]:
                    facts["details"] = f" with {facts['numbers'][0]}"
                else:
                    facts["details"] = ""
            if len(facts["names"]) > 1:
                facts["ranking"] = "\n".join([
                    f"{i+1}. **{name}**" for i, name in enumerate(facts["names"][:10])
                ])
        
        elif q_type == "enumeration":
            if facts["names"]:
                facts["list"] = "\n".join([f"• {name}" for name in facts["names"][:7]])
                facts["primary_answer"] = facts["names"][0]
        
        elif q_type == "quantity":
            if facts["numbers"]:
                facts["primary_answer"] = facts["numbers"][0]
                facts["number"] = facts["numbers"][0]
        
        elif q_type == "temporal":
            if facts["dates"]:
                facts["primary_answer"] = facts["dates"][0]
                facts["date"] = facts["dates"][0]
        
        else:
            # General - take first sentence as answer
            sentences = re.split(r'(?<=[.!?])\s+', summary)
            if sentences:
                facts["primary_answer"] = sentences[0]
        
        # Get sources
        citations = research_data.get("citations", [])
        for citation in citations[:3]:
            if isinstance(citation, dict):
                facts["sources"].append(citation.get("source", "research"))
            else:
                facts["sources"].append(str(citation))
        
        # Supporting details - remaining sentences
        if summary:
            sentences = re.split(r'(?<=[.!?])\s+', summary)
            facts["supporting_details"] = sentences[1:5]
        
        return facts
    
    def _fill_template(self, template: str, facts: Dict, subject: str, 
                        q_type: str) -> str:
        """Fill the answer template with extracted facts"""
        
        # Prepare all possible template variables
        template_vars = {
            "subject": subject,
            "answer": facts.get("primary_answer", ""),
            "definition": facts.get("definition", facts.get("primary_answer", "")),
            "name": facts.get("names", [""])[0] if facts.get("names") else subject,
            "description": facts.get("primary_answer", "a notable entity"),
            "date": facts.get("date", facts.get("dates", [""])[0] if facts.get("dates") else ""),
            "context": "",
            "location": facts.get("locations", [""])[0] if facts.get("locations") else "",
            "number": facts.get("number", facts.get("numbers", [""])[0] if facts.get("numbers") else ""),
            "unit": "units",
            "action": subject.lower() if subject else "complete this",
            "steps": facts.get("steps", "1. Begin the process\n2. Follow standard procedure"),
            "mechanism": facts.get("mechanism", facts.get("primary_answer", "")),
            "criteria": "current data",
            "selection": facts.get("primary_answer", ""),
            "cause": facts.get("cause", facts.get("primary_answer", "")),
            "effect": subject,
            "recommendations": facts.get("list", facts.get("ranking", "")),
            "ranking": facts.get("ranking", facts.get("list", "")),
            "list": facts.get("list", ""),
            "data_source": facts.get("sources", ["available data"])[0] if facts.get("sources") else "available data",
            "superlative": self._get_superlative_word(subject),
            "details": facts.get("details", "")
        }
        
        try:
            answer = template.format(**template_vars)
        except KeyError:
            # Fallback if template can't be filled
            answer = facts.get("primary_answer", f"Regarding {subject}...")
        
        return answer
    
    def _get_superlative_word(self, text: str) -> str:
        """Extract superlative from text"""
        superlatives = ["largest", "biggest", "smallest", "most", "best", "worst", 
                       "fastest", "slowest", "highest", "lowest"]
        text_lower = text.lower()
        for sup in superlatives:
            if sup in text_lower:
                return sup
        return "top"
    
    def _construct_no_info_response(self, question: str, subject: str, 
                                     q_type: str) -> str:
        """Construct response when no facts are available"""
        return f"I don't have specific current information about {subject}. Could you provide more context or rephrase your question?"
    
    def _ensure_directness(self, answer: str, question: str) -> str:
        """Force answer to be direct and specific"""
        
        # Remove vague openings
        vague_beginnings = [
            "Current understanding suggests",
            "Many experts believe",
            "It appears that",
            "Generally speaking",
            "In many cases",
            "Sources suggest that",
            "According to various",
            "From what I found,",
            "hmm...",
            "oh,",
            "well,",
            "so,"
        ]
        
        for vague in vague_beginnings:
            if answer.lower().startswith(vague.lower()):
                answer = answer[len(vague):].strip()
                # Capitalize first letter
                if answer:
                    answer = answer[0].upper() + answer[1:]
        
        # Remove filler phrases
        filler_phrases = [
            "I think that", "In my opinion,", "It seems like",
            "You could say that", "One might say"
        ]
        
        for filler in filler_phrases:
            answer = re.sub(re.escape(filler), "", answer, flags=re.IGNORECASE)
        
        # Clean up any double spaces or leading spaces
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        return answer
    
    def _extract_proper_nouns(self, text: str) -> List[str]:
        """Extract proper nouns (names, organizations)"""
        # Pattern for proper nouns (capitalized words)
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, text)
        
        # Filter out common words that might be capitalized at sentence start
        common_words = {"The", "This", "That", "These", "Those", "It", "They", 
                       "We", "He", "She", "Based", "According", "Details", "Note"}
        
        filtered = [m for m in matches if m not in common_words]
        
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for name in filtered:
            if name not in seen:
                seen.add(name)
                unique.append(name)
        
        return unique
    
    def _extract_numbers(self, text: str) -> List[str]:
        """Extract numbers with their context"""
        # Match numbers with optional currency, units, percentages
        patterns = [
            r'\$[\d,]+\.?\d*\s*(?:trillion|billion|million)?',  # Currency
            r'[\d,]+\.?\d*\s*(?:trillion|billion|million|thousand)',  # Large numbers
            r'\d+(?:\.\d+)?%',  # Percentages
            r'\b\d{4}\b',  # Years
            r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'  # General numbers
        ]
        
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            numbers.extend(matches)
        
        return list(dict.fromkeys(numbers))  # Deduplicate
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text"""
        patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{4}\b',  # Just years
            r'\b(?:in|on|since)\s+\d{4}\b'
        ]
        
        dates = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        
        return list(dict.fromkeys(dates))
    
    def _extract_definition(self, text: str) -> str:
        """Extract a definition from text"""
        # Look for "X is Y" pattern
        is_pattern = r'^([^.]+?)\s+(?:is|are|was|were|refers to|means)\s+([^.]+)'
        match = re.search(is_pattern, text)
        
        if match:
            return f"{match.group(1)} is {match.group(2)}"
        
        # Fallback: first sentence
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return sentences[0] if sentences else text


class EmergencyResponseFix:
    """Quick fixes to immediately make responses DeepSeek-like"""
    
    def __init__(self):
        self.direct_answer_core = DirectAnswerCore()
        
        # Vague phrases to remove/replace
        self.vague_fixes = {
            r"\bcurrent understanding suggests\b": "",
            r"\bmany experts believe\b": "Research indicates",
            r"\bit appears that\b": "",
            r"\bgenerally speaking,?\s*": "",
            r"\bin many cases,?\s*": "",
            r"\bsources suggest that?\b": "According to data,",
            r"\bfrom what i found,?\s*": "",
            r"\bfrom my research,?\s*": "Based on research,",
            r"\bit seems\s*": "",
            r"\bhmm\.\.\.?\s*": "",
            r"\boh,?\s*": "",
            r"\bwell,?\s*": "",
            r"\bso,?\s*": "",
            r"\bi think\s*": "",
            r"\bi believe\s*": "",
            r"^\s*\.\.\.\s*": ""
        }
    
    def fix_response(self, user_input: str, current_response: str, 
                     research_data: Dict = None) -> str:
        """Apply critical fixes to make responses DeepSeek-like"""
        
        fixed = current_response
        
        # 1. If it's a direct question, ensure we ANSWER IT
        if self._is_direct_question(user_input):
            fixed = self._ensure_direct_answer(fixed, user_input, research_data)
        
        # 2. Remove vague beginnings and phrases
        fixed = self._remove_vague_language(fixed)
        
        # 3. Add specifics if missing
        if not self._has_specifics(fixed) and research_data:
            fixed = self._add_specifics(fixed, user_input, research_data)
        
        # 4. Add structure if long
        if len(fixed.split()) > 50:
            fixed = self._add_structure(fixed)
        
        # 5. Ensure proper formatting and ending
        fixed = self._format_properly(fixed)
        
        return fixed
    
    def _is_direct_question(self, text: str) -> bool:
        """Check if this requires a direct answer"""
        direct_indicators = [
            "what is", "what are", "who is", "who are",
            "when did", "when was", "where is", "where are",
            "how many", "how much", "which", "why does", "why is",
            "largest", "biggest", "best", "top", "most"
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in direct_indicators)
    
    def _ensure_direct_answer(self, response: str, question: str, 
                               research_data: Dict = None) -> str:
        """Ensure response actually answers the question"""
        
        # Check if response starts with an answer
        question_lower = question.lower()
        response_lower = response.lower()[:100]
        
        # If asking "what is X" and response doesn't define X
        if "what is" in question_lower:
            subject = self.direct_answer_core._extract_subject(question, "definition")
            if subject.lower() not in response_lower and "is" not in response_lower[:30]:
                # Prepend a direct answer structure
                if research_data and research_data.get("summary"):
                    first_sentence = research_data["summary"].split(".")[0]
                    response = f"**{subject}** {first_sentence}.\n\n{response}"
                else:
                    response = f"Regarding {subject}: {response}"
        
        # If asking for superlative and no specific entity mentioned early
        superlatives = ["largest", "biggest", "smallest", "best", "most", "top"]
        for sup in superlatives:
            if sup in question_lower:
                # Check if answer starts with a specific entity
                has_entity = bool(re.match(r'^(?:The\s+)?[A-Z][a-z]+', response))
                if not has_entity and research_data:
                    names = self.direct_answer_core._extract_proper_nouns(
                        research_data.get("summary", "")
                    )
                    if names:
                        response = f"The {sup} is **{names[0]}**.\n\n{response}"
                break
        
        return response
    
    def _remove_vague_language(self, response: str) -> str:
        """Remove vague phrases"""
        fixed = response
        
        for pattern, replacement in self.vague_fixes.items():
            fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)
        
        # Fix capitalization after removals
        fixed = re.sub(r'^\s*([a-z])', lambda m: m.group(1).upper(), fixed)
        fixed = re.sub(r'\.\s+([a-z])', lambda m: '. ' + m.group(1).upper(), fixed)
        
        # Clean up extra spaces
        fixed = re.sub(r'\s+', ' ', fixed)
        fixed = re.sub(r'\s+\.', '.', fixed)
        
        return fixed.strip()
    
    def _has_specifics(self, response: str) -> bool:
        """Check if response has specific information"""
        # Has numbers
        has_numbers = bool(re.search(r'\d+', response))
        
        # Has proper nouns beyond common words
        proper_nouns = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', response)
        has_proper_nouns = len(proper_nouns) > 0
        
        # Has bold text (structured)
        has_bold = '**' in response
        
        return has_numbers or has_proper_nouns or has_bold
    
    def _add_specifics(self, response: str, question: str, 
                       research_data: Dict) -> str:
        """Add specific details from research"""
        if not research_data:
            return response
        
        summary = research_data.get("summary", "")
        if not summary:
            return response
        
        # Extract specifics from research
        numbers = self.direct_answer_core._extract_numbers(summary)
        names = self.direct_answer_core._extract_proper_nouns(summary)
        
        # If response lacks specifics, add them
        additions = []
        if names and not any(name in response for name in names[:3]):
            additions.append(f"Key entities: {', '.join(names[:3])}")
        if numbers and not any(num in response for num in numbers[:2]):
            additions.append(f"Key figures: {', '.join(numbers[:2])}")
        
        if additions:
            response = response.rstrip('.') + ".\n\n**Additional specifics:**\n• " + "\n• ".join(additions)
        
        return response
    
    def _add_structure(self, response: str) -> str:
        """Add structure to long responses"""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', response)
        
        if len(sentences) <= 2:
            return response
        
        # First sentence is the main answer
        main_answer = sentences[0]
        
        # Rest become details
        details = sentences[1:]
        
        # Build structured response
        structured = main_answer + "\n\n**Details:**\n"
        
        # Format details as bullet points if short enough
        if len(details) <= 5:
            structured += "\n".join([f"• {s}" for s in details])
        else:
            structured += " ".join(details[:4])
        
        return structured
    
    def _format_properly(self, response: str) -> str:
        """Apply proper formatting"""
        # Remove multiple blank lines
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        # Ensure proper punctuation at end
        if response and not response.rstrip().endswith(('.', '!', '?', '*')):
            response = response.rstrip() + '.'
        
        # Clean up whitespace
        response = response.strip()
        
        return response


class FactSynthesizer:
    """Extract and organize facts from research for response generation"""
    
    def __init__(self):
        self.direct_answer_core = DirectAnswerCore()
    
    def synthesize_facts(self, research_data: Dict, question: str) -> Dict:
        """Convert research into organized, usable facts"""
        
        facts = {
            "primary_answer": "",
            "direct_response": "",
            "supporting_details": [],
            "key_numbers": [],
            "key_entities": [],
            "key_dates": [],
            "sources": [],
            "confidence": 0.5,
            "question_type": "",
            "subject": ""
        }
        
        # Classify the question
        q_type, _ = self.direct_answer_core._classify_question(question)
        facts["question_type"] = q_type
        facts["subject"] = self.direct_answer_core._extract_subject(question, q_type)
        
        # Get summary
        summary = research_data.get("summary", "")
        if not summary:
            return facts
        
        # Extract entities
        facts["key_entities"] = self.direct_answer_core._extract_proper_nouns(summary)
        facts["key_numbers"] = self.direct_answer_core._extract_numbers(summary)
        facts["key_dates"] = self.direct_answer_core._extract_dates(summary)
        
        # Get primary answer
        facts["primary_answer"] = self._get_primary_answer(summary, q_type, facts)
        
        # Get supporting details
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        facts["supporting_details"] = sentences[1:6]
        
        # Get sources
        citations = research_data.get("citations", [])
        for citation in citations[:3]:
            if isinstance(citation, dict):
                source = citation.get("source", "")
                if source == "duckduckgo":
                    # Try to extract domain from URL
                    url = citation.get("url", "")
                    domain_match = re.search(r'//(?:www\.)?([^/]+)', url)
                    if domain_match:
                        source = domain_match.group(1)
                facts["sources"].append({
                    "name": source,
                    "url": citation.get("url", ""),
                    "type": citation.get("source", "web")
                })
            else:
                facts["sources"].append({"name": str(citation), "type": "web"})
        
        # Calculate confidence
        facts["confidence"] = research_data.get("confidence", 0.5)
        
        # Generate direct response
        facts["direct_response"] = self._generate_direct_response(facts)
        
        return facts
    
    def _get_primary_answer(self, summary: str, q_type: str, facts: Dict) -> str:
        """Extract the primary answer for the question type"""
        
        if q_type in ["superlative", "ranking"]:
            if facts["key_entities"]:
                return facts["key_entities"][0]
        
        elif q_type == "quantity":
            if facts["key_numbers"]:
                return facts["key_numbers"][0]
        
        elif q_type == "temporal":
            if facts["key_dates"]:
                return facts["key_dates"][0]
        
        elif q_type == "definition":
            # Look for "X is Y" pattern
            is_match = re.search(r'^([^.]+?)\s+(?:is|are|refers to)\s+([^.]+)', summary)
            if is_match:
                return f"{is_match.group(1)} is {is_match.group(2)}"
        
        # Default: first sentence
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        return sentences[0] if sentences else summary[:200]
    
    def _generate_direct_response(self, facts: Dict) -> str:
        """Generate a direct response from facts"""
        q_type = facts["question_type"]
        subject = facts["subject"]
        primary = facts["primary_answer"]
        
        if not primary:
            return f"I couldn't find specific information about {subject}."
        
        if q_type == "superlative":
            sup_word = "largest"  # Could be extracted from question
            details = ""
            if facts["key_numbers"]:
                details = f" ({facts['key_numbers'][0]})"
            return f"The {sup_word} {subject.lower()} is **{primary}**{details}."
        
        elif q_type == "quantity":
            return f"There are **{primary}** {subject.lower()}."
        
        elif q_type == "definition":
            return primary if "is" in primary.lower() else f"**{subject}** is {primary}."
        
        elif q_type == "temporal":
            return f"{subject} occurred in **{primary}**."
        
        else:
            return primary


if __name__ == "__main__":
    print("=== Direct Answer Core Test ===\n")
    
    core = DirectAnswerCore()
    fixer = EmergencyResponseFix()
    
    # Test with banking question
    question = "What is the largest bank in the world?"
    research = {
        "summary": "The Industrial and Commercial Bank of China (ICBC) is the largest bank in the world by total assets, with approximately $5.5 trillion USD. Other major banks include China Construction Bank and Bank of America.",
        "confidence": 0.85,
        "citations": [{"source": "wikipedia", "url": "https://en.wikipedia.org/wiki/ICBC"}]
    }
    
    print(f"Question: {question}")
    print()
    
    # Test direct answer generation
    direct_answer = core.generate_direct_answer(question, research)
    print("Direct Answer:")
    print(direct_answer)
    print()
    
    # Test emergency fix on vague response
    vague_response = "Current understanding suggests many of the largest banks in the world are part of larger bank holding companies. It seems they offer various services."
    
    print("Original (Vague):")
    print(vague_response)
    print()
    
    fixed = fixer.fix_response(question, vague_response, research)
    print("After Emergency Fix:")
    print(fixed)
