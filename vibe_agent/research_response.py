"""
Research Response Builder - DeepSeek Quality Upgrade
Converts research results into high-quality, specific, well-structured responses
"""

import re
import random
from deepseek_quality import ImmediateDeepSeekUpgrade, AnswerEnhancer, AnswerQualityEvaluator
from thinking_integration import ConfidenceCalibrator, SourceIntegrator
from research_detail_builder import detailed_response_builder


class ResearchResponseBuilder:
    """Convert research results into DeepSeek-quality responses."""
    
    def __init__(self):
        self.upgrader = ImmediateDeepSeekUpgrade()
        self.enhancer = AnswerEnhancer()
        self.evaluator = AnswerQualityEvaluator()
        self.calibrator = ConfidenceCalibrator()
        self.source_integrator = SourceIntegrator()
    
    # DeepSeek-style templates with structure
    RESPONSE_TEMPLATES = {
        "factual": "{summary}\n\n*{citation}*",
        "structured": """{direct_answer}

**Details:**
{details}

*{citation}*""",
        "comparative": """{intro}

{ranked_list}

**Key factors:** {factors}

*{citation}*""",
        "exploratory": """{summary}

**Additional context:**
{context}

*{citation}*"""
    }
    
    # Old templates kept for vibe-mode compatibility
    CONVERSATIONAL_TEMPLATES = {
        "factual": "{summary}. *{citation}*",
        "exploratory": "{summary}. what's your take? *{citation}*",
        "uncertain": "Research suggests {summary}. *{citation}*",
        "deep_dive": "After exploring this, here's what I found: {summary} *{citation}*"
    }
    
    DIRECT_TEMPLATES = {
        "factual": "{summary}",
        "confident": "{summary}",
        "minimal": "{summary}"
    }
    
    def build_response(self, research_result, vibe="REFLECTION", direct_mode=False, 
                       question="", deepseek_mode=True):
        """
        Create high-quality response with research.
        
        Args:
            research_result: Dict with summary, confidence, citations
            vibe: Current vibe state
            direct_mode: If True, use minimal formatting
            question: Original question for context
            deepseek_mode: If True, use DeepSeek quality enhancements
        """
        
        if not research_result or not research_result.get("summary"):
            return "I couldn't find a clear answer to that. Could you rephrase?"
        
        # Direct mode: concise but still quality
        if direct_mode:
            return self._build_direct_response(research_result, question, deepseek_mode)
        
        # Full quality mode
        if deepseek_mode:
            # Check for detailed/technical query
            if self._requires_detailed_breakdown(question, research_result.get("summary", "")):
                try:
                    detail_builder = detailed_response_builder()
                    return detail_builder.build_detailed_response(research_result, question)
                except Exception as e:
                    print(f"Detail builder failed, falling back: {e}")
                    # Fallthrough to standard DeepSeek response
            
            return self._build_deepseek_response(research_result, question, vibe)
        else:
            return self._build_conversational_response(research_result, vibe)

    def _requires_detailed_breakdown(self, question, summary):
        """Check if query warrants a full detailed breakdown"""
        q_lower = question.lower()
        triggers = ["features", "benefits", "changes", "new in", "changelog", "vs", "versus", "performance", "optimization", "difference"]
        
        # Explicit trigger words
        if any(t in q_lower for t in triggers):
            return True
            
        # Or if summary contains list-like data
        if summary.count("•") > 3 or summary.count("- ") > 3:
            return True
            
        return False
    
    def _build_deepseek_response(self, research_result, question, vibe):
        """Build a DeepSeek-quality structured response."""
        
        summary = research_result.get("summary", "")
        confidence = research_result.get("confidence", 0.5)
        citations = research_result.get("citations", [])
        
        # 1. Determine response structure based on question type
        structure_type = self._classify_response_structure(question, summary)
        
        # 2. Extract components
        components = self._extract_response_components(summary, question, structure_type)
        
        # 3. Apply template
        template = self.RESPONSE_TEMPLATES.get(structure_type, self.RESPONSE_TEMPLATES["factual"])
        
        # 4. Format citation
        components["citation"] = self._format_citation_deepseek(citations)
        
        try:
            response = template.format(**components)
        except KeyError:
            # Fallback to simple format
            response = f"{summary}\n\n*{components['citation']}*"
        
        # 5. Apply quality enhancements
        response = self.upgrader.enhance_response(response, {"citations": citations})
        
        # 6. Apply confidence calibration
        response = self.calibrator.calibrate_response(response, confidence)
        
        # 7. Final quality check
        eval_result = self.evaluator.evaluate_answer(response, question)
        
        # If quality is still low, enhance more aggressively
        if eval_result["total_score"] < 60:
            response = self.enhancer.enhance_answer(response, question, research_result)
        
        return response.strip()
    
    def _classify_response_structure(self, question, summary):
        """Determine the best response structure for this question."""
        question_lower = question.lower() if question else ""
        
        # List/ranking questions
        if re.search(r'\b(top \d+|best|largest|biggest|list|rank)', question_lower):
            return "comparative"
        
        # Detail-heavy questions
        if len(summary.split()) > 80:
            return "structured"
        
        # Explanatory questions
        if re.search(r'\b(how|why|explain|what is)\b', question_lower):
            if len(summary.split()) > 50:
                return "structured"
            return "exploratory"
        
        return "factual"
    
    def _extract_response_components(self, summary, question, structure_type):
        """Extract components needed for the response template."""
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        
        components = {
            "summary": summary,
            "direct_answer": sentences[0] if sentences else summary,
            "details": " ".join(sentences[1:4]) if len(sentences) > 1 else "Additional information available upon request.",
            "context": " ".join(sentences[2:]) if len(sentences) > 2 else "",
            "intro": "",
            "ranked_list": "",
            "factors": "Reliability, scale, and current data"
        }
        
        # For comparative structure, try to extract a list
        if structure_type == "comparative":
            # Look for numbered or bulleted items
            list_items = re.findall(r'(?:\d+[.)]\s*|\•\s*|[-*]\s*)([^.]+)', summary)
            if list_items:
                # Format as numbered list
                formatted_items = [f"{i+1}. **{item.strip()}**" for i, item in enumerate(list_items[:10])]
                components["ranked_list"] = "\n".join(formatted_items)
                components["intro"] = f"Based on available data, here are the key results"
            else:
                # Extract proper nouns as potential list items
                proper_nouns = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', summary)
                unique_nouns = list(dict.fromkeys(proper_nouns))[:5]
                if unique_nouns:
                    formatted_items = [f"• **{noun}**" for noun in unique_nouns]
                    components["ranked_list"] = "\n".join(formatted_items)
                    components["intro"] = "Key entities mentioned"
                else:
                    components["ranked_list"] = components["direct_answer"]
                    components["intro"] = "Research findings"
        
        return components
    
    def _build_direct_response(self, research_result, question="", deepseek_mode=True):
        """Build a direct, concise answer."""
        summary = research_result.get("summary", "")
        confidence = research_result.get("confidence", 0)
        citations = research_result.get("citations", [])
        
        if not summary:
            return "I couldn't find a clear answer to that."
        
        # Extract first meaningful sentence
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        direct_answer = sentences[0] if sentences else summary
        
        # If first sentence is too long, find a better break point
        if len(direct_answer) > 200:
            # Try to break at a comma
            if ", " in direct_answer[:150]:
                direct_answer = direct_answer[:direct_answer.index(", ", 100)] + "."
            else:
                direct_answer = direct_answer[:150] + "..."
        
        if deepseek_mode:
            # Apply DeepSeek upgrades even to direct responses
            direct_answer = self.upgrader.enhance_response(direct_answer)
            
            # Add structured details if we have more info
            if len(sentences) > 1 and len(direct_answer) < 100:
                direct_answer += f"\n\n**Details:** {sentences[1]}"
        
        # Add citation for high confidence
        if confidence > 0.6 and citations:
            citation = self._format_citation_deepseek(citations)
            direct_answer += f"\n\n*{citation}*"
        
        # Calibrate confidence in language
        direct_answer = self.calibrator.calibrate_response(direct_answer, confidence)
        
        return direct_answer.strip()
    
    def _build_conversational_response(self, research_result, vibe):
        """Build a conversational response (legacy mode)."""
        
        # Choose template based on confidence
        if research_result["confidence"] > 0.8:
            template_type = "factual"
        elif research_result["confidence"] > 0.4:
            template_type = "exploratory"
        else:
            template_type = "uncertain"
            
        template = self.CONVERSATIONAL_TEMPLATES[template_type]
        
        # Conversationalize summary (still clean it up)
        summary = self._conversationalize(research_result["summary"])
        summary = self.upgrader.enhance_response(summary)
        
        # Format citation
        citation = self._format_citation_deepseek(research_result.get("citations", []))
        
        response = template.format(summary=summary, citation=citation)
        
        # Confidence warning
        if research_result["confidence"] < 0.3:
            response += " (I'd recommend verifying this)"
            
        return response
    
    def _conversationalize(self, summary):
        """Make summary conversational while keeping specificity."""
        # Remove overly formal phrases but keep substance
        replacements = {
            "According to": "Based on",
            "Studies show that": "Research indicates",
            "It is known that": "Evidence shows",
            "The data indicates": "Data shows",
            "It has been found": "Research found"
        }
        
        result = summary
        for k, v in replacements.items():
            result = result.replace(k, v)
        
        # Don't lowercase - keeps proper nouns intact
        return result
    
    def _format_citation_deepseek(self, citations):
        """Format citations in DeepSeek style."""
        if not citations:
            return "Based on available research data."
        
        # Extract source names
        source_names = []
        for citation in citations[:3]:
            if isinstance(citation, dict):
                source = citation.get("source", "")
                url = citation.get("url", "")
                
                # Clean up source name
                if source == "duckduckgo" and url:
                    # Extract domain from URL
                    domain_match = re.search(r'//([^/]+)', url)
                    if domain_match:
                        source = domain_match.group(1).replace("www.", "")
                elif source == "wikipedia":
                    source = "Wikipedia"
                elif source == "arxiv":
                    source = "arXiv research"
                elif source == "news":
                    source = "news sources"
                
                if source:
                    source_names.append(source)
            elif isinstance(citation, str):
                source_names.append(citation)
        
        if not source_names:
            return "Based on research."
        
        # Format based on count
        if len(source_names) == 1:
            return f"Source: {source_names[0]}"
        else:
            return f"Sources: {', '.join(source_names)}"
    
    def _format_citation(self, citations):
        """Legacy citation formatting."""
        return self._format_citation_deepseek(citations)
    
    def evaluate_response(self, response, question=""):
        """Evaluate the quality of a response."""
        return self.evaluator.evaluate_answer(response, question)


if __name__ == "__main__":
    print("=== Research Response Builder Test ===\n")
    
    builder = ResearchResponseBuilder()
    
    # Test with a sample research result
    research_result = {
        "summary": "Many of the largest banks are part of larger holding companies. The Industrial and Commercial Bank of China (ICBC) is the largest bank in the world by total assets at approximately $5.5 trillion. Other major banks include China Construction Bank and Bank of America.",
        "confidence": 0.85,
        "citations": [
            {"source": "wikipedia", "url": "https://en.wikipedia.org/wiki/Bank"},
            {"source": "duckduckgo", "url": "https://www.thebanker.com/top-1000"}
        ]
    }
    
    print("BEFORE (Old Style):")
    old_response = builder._build_conversational_response(research_result, "REFLECTION")
    print(old_response)
    print()
    
    print("AFTER (DeepSeek Style):")
    new_response = builder.build_response(
        research_result,
        question="What is the largest bank in the world?",
        deepseek_mode=True
    )
    print(new_response)
    print()
    
    # Evaluate
    eval_result = builder.evaluate_response(new_response, "What is the largest bank?")
    print(f"Quality Score: {eval_result['total_score']}/100")
    print(f"Breakdown: {eval_result['breakdown']}")

