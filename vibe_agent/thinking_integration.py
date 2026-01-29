"""
Thinking Integration & Confidence Calibration
Connects the agent's thinking process to final response quality
"""

import re
import random
from typing import Dict, List, Any, Optional


class ThinkingIntegration:
    """Connect thinking process to final response - ensure coherence"""
    
    def __init__(self):
        self.thinking_history = []
    
    def integrate_thinking(self, thinking_data: Dict, response: str) -> Dict[str, Any]:
        """
        Ensure thinking process informs the response
        Returns: {"final_response": "", "thinking_explanation": "", "insight_coverage": 0-100}
        """
        
        # Extract key insights from thinking process
        insights = self._extract_insights(thinking_data)
        
        # Check if response addresses insights
        coverage = self._check_insight_coverage(response, insights)
        
        # If missing coverage, enhance response
        if coverage["missing_insights"]:
            response = self._enhance_with_insights(response, coverage["missing_insights"])
        
        # Generate thinking explanation for user (optional transparency)
        thinking_explanation = self._generate_thinking_explanation(thinking_data, coverage)
        
        # Store for learning
        self.thinking_history.append({
            "thinking": thinking_data,
            "response": response,
            "coverage": coverage["coverage_percentage"]
        })
        
        return {
            "final_response": response,
            "thinking_explanation": thinking_explanation,
            "insight_coverage": coverage["coverage_percentage"],
            "insights_used": len(insights) - len(coverage["missing_insights"]),
            "insights_total": len(insights)
        }
    
    def _extract_insights(self, thinking_data: Dict) -> List[Dict]:
        """Extract key insights from thinking process"""
        insights = []
        
        # Handle different thinking data formats
        if isinstance(thinking_data, dict):
            # From reasoning engine
            thought_process = thinking_data.get("thought_process", {})
            
            # Extract from question analysis
            qa = thought_process.get("question_analysis", {})
            if qa:
                insights.append({
                    "type": "question_understanding",
                    "content": f"Question type: {qa.get('question_type', 'unknown')}",
                    "confidence": 0.8,
                    "addressed": False
                })
            
            # Extract key insights
            key_insights = thought_process.get("key_insights", [])
            for insight in key_insights:
                insights.append({
                    "type": "key_insight",
                    "content": insight,
                    "confidence": thinking_data.get("confidence", 0.5),
                    "addressed": False
                })
            
            # Extract from reasoning type
            reasoning_type = thought_process.get("reasoning_applied", "")
            if reasoning_type:
                insights.append({
                    "type": "reasoning_approach",
                    "content": f"Using {reasoning_type} reasoning",
                    "confidence": 0.7,
                    "addressed": False
                })
        
        elif isinstance(thinking_data, str):
            # Parse NEURAL DRIFT format from thinking logs
            drift_pattern = r"Understanding:\s*([^\n]+)"
            matches = re.findall(drift_pattern, thinking_data)
            for match in matches:
                insights.append({
                    "type": "understanding",
                    "content": match.strip(),
                    "confidence": 0.6,
                    "addressed": False
                })
            
            # Parse key insights
            insight_pattern = r"Key insight:\s*([^\n]+)"
            matches = re.findall(insight_pattern, thinking_data)
            for match in matches:
                insights.append({
                    "type": "key_insight",
                    "content": match.strip(),
                    "confidence": 0.7,
                    "addressed": False
                })
        
        return insights
    
    def _check_insight_coverage(self, response: str, insights: List[Dict]) -> Dict:
        """Check which insights are addressed in response"""
        covered = 0
        missing = []
        
        response_lower = response.lower()
        
        for insight in insights:
            # Extract key terms from insight
            key_terms = self._extract_key_terms(insight["content"])
            
            # Check if any key terms appear in response
            addressed = any(term.lower() in response_lower for term in key_terms if len(term) > 3)
            
            if addressed:
                covered += 1
                insight["addressed"] = True
            else:
                missing.append(insight)
        
        coverage_percentage = (covered / len(insights)) * 100 if insights else 100
        
        return {
            "coverage_percentage": coverage_percentage,
            "missing_insights": missing,
            "covered_insights": [i for i in insights if i["addressed"]],
            "total_insights": len(insights)
        }
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract meaningful terms from text"""
        # Remove common words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", 
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "must", "to",
                      "of", "in", "for", "on", "with", "at", "by", "from", "as",
                      "into", "through", "that", "this", "these", "those", "and",
                      "or", "but", "if", "then", "than", "so", "because", "about",
                      "using", "type", "question"}
        
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in stop_words and len(w) > 3]
    
    def _enhance_with_insights(self, response: str, missing_insights: List[Dict]) -> str:
        """Enhance response with missing insights"""
        # Only add truly important missing insights
        important_missing = [i for i in missing_insights if i["confidence"] > 0.6]
        
        if not important_missing:
            return response
        
        # Add a supplementary section
        additions = []
        for insight in important_missing[:2]:  # Limit to top 2
            if insight["type"] == "key_insight":
                additions.append(insight["content"])
        
        if additions:
            if "\n\n" in response:
                # Insert before final section
                parts = response.rsplit("\n\n", 1)
                response = f"{parts[0]}\n\n**Additional context:** {' '.join(additions)}\n\n{parts[1]}"
            else:
                response += f"\n\n**Note:** {' '.join(additions)}"
        
        return response
    
    def _generate_thinking_explanation(self, thinking_data: Dict, coverage: Dict) -> str:
        """Generate a brief explanation of the thinking process"""
        parts = []
        
        # Get thinking metadata
        if isinstance(thinking_data, dict):
            confidence = thinking_data.get("confidence", 0.5)
            thought_process = thinking_data.get("thought_process", {})
            
            # Question type
            qa = thought_process.get("question_analysis", {})
            if qa.get("question_type"):
                parts.append(f"Identified as {qa['question_type']} question")
            
            # Reasoning approach
            reasoning = thought_process.get("reasoning_applied", "")
            if reasoning:
                parts.append(f"Applied {reasoning}")
            
            # Confidence level
            if confidence > 0.7:
                parts.append("High confidence in response")
            elif confidence < 0.4:
                parts.append("Some uncertainty remains")
        
        # Coverage note
        if coverage["coverage_percentage"] < 50:
            parts.append("Some insights may need further exploration")
        
        return " • ".join(parts) if parts else "Standard response generation"


class ConfidenceCalibrator:
    """Match DeepSeek's appropriate confidence levels"""
    
    def __init__(self):
        # Track confidence accuracy over time
        self.calibration_history = []
        self.overconfidence_count = 0
        self.underconfidence_count = 0
    
    def calibrate_response(self, response: str, research_confidence: float, 
                           thinking_confidence: float = 0.5) -> str:
        """
        Adjust response language based on confidence levels
        High confidence = direct statements
        Low confidence = qualified statements
        """
        overall_confidence = (research_confidence * 0.6 + thinking_confidence * 0.4)
        
        if overall_confidence >= 0.8:
            # High confidence: Direct, assertive
            return self._make_assertive(response)
        elif overall_confidence >= 0.6:
            # Medium confidence: Slight qualification
            return self._add_qualifiers(response, level="mild")
        elif overall_confidence >= 0.4:
            # Low confidence: Clear qualification
            return self._add_qualifiers(response, level="moderate")
        else:
            # Very low confidence: Strong qualification
            return self._add_qualifiers(response, level="strong")
    
    def _make_assertive(self, response: str) -> str:
        """Make response more direct and confident"""
        # Remove unnecessary hedging
        hedging_patterns = [
            (r"\bi think\s+", ""),
            (r"\bi believe\s+", ""),
            (r"\bperhaps\s+", ""),
            (r"\bmaybe\s+", ""),
            (r"\bpossibly\s+", ""),
            (r"\bit seems like\s+", ""),
            (r"\bit appears that\s+", ""),
            (r"\bfrom what i understand,?\s*", ""),
        ]
        
        result = response
        for pattern, replacement in hedging_patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        # Capitalize after removals
        result = re.sub(r'^\s*([a-z])', lambda m: m.group(1).upper(), result)
        
        return result
    
    def _add_qualifiers(self, response: str, level: str = "mild") -> str:
        """Add appropriate qualifiers based on confidence"""
        qualifiers = {
            "mild": [
                "Based on available information,",
                "According to current data,",
                "Research indicates that"
            ],
            "moderate": [
                "It appears that",
                "Current understanding suggests",
                "Available evidence indicates"
            ],
            "strong": [
                "There's some indication that",
                "Preliminary information suggests",
                "It's possible that"
            ]
        }
        
        # Check if already qualified
        already_qualified = any(
            q.lower() in response[:150].lower() 
            for q in ["based on", "according to", "research", "it appears", 
                     "evidence suggests", "information suggests"]
        )
        
        if already_qualified:
            return response
        
        # Add qualifier to beginning
        qualifier = random.choice(qualifiers[level])
        
        # Handle capitalization
        if response and response[0].isupper():
            response = response[0].lower() + response[1:]
        
        return f"{qualifier} {response}"
    
    def get_confidence_language(self, confidence: float) -> Dict[str, str]:
        """Get appropriate language markers for a confidence level"""
        if confidence >= 0.8:
            return {
                "opener": "",
                "certainty_word": "is",
                "qualifier": "",
                "closing": ""
            }
        elif confidence >= 0.6:
            return {
                "opener": "Generally speaking, ",
                "certainty_word": "is typically",
                "qualifier": "in most cases",
                "closing": ""
            }
        elif confidence >= 0.4:
            return {
                "opener": "Based on available information, ",
                "certainty_word": "appears to be",
                "qualifier": "though this may vary",
                "closing": " Further verification is recommended."
            }
        else:
            return {
                "opener": "There are indications that ",
                "certainty_word": "might be",
                "qualifier": "but with limited certainty",
                "closing": " This should be verified with additional sources."
            }
    
    def record_feedback(self, was_accurate: bool, stated_confidence: float):
        """Record feedback to improve calibration over time"""
        self.calibration_history.append({
            "stated_confidence": stated_confidence,
            "was_accurate": was_accurate
        })
        
        if was_accurate and stated_confidence < 0.5:
            self.underconfidence_count += 1
        elif not was_accurate and stated_confidence > 0.7:
            self.overconfidence_count += 1
    
    def get_calibration_adjustment(self) -> float:
        """Get adjustment factor based on historical accuracy"""
        if len(self.calibration_history) < 10:
            return 0.0  # Not enough data
        
        # If we're often wrong when confident, lower confidence
        if self.overconfidence_count > len(self.calibration_history) * 0.2:
            return -0.1
        
        # If we're often right when uncertain, raise confidence
        if self.underconfidence_count > len(self.calibration_history) * 0.2:
            return 0.1
        
        return 0.0


class ResponsePipeline:
    """Full pipeline from thinking to DeepSeek-quality response"""
    
    def __init__(self, research_engine=None):
        self.thinking_integrator = ThinkingIntegration()
        self.confidence_calibrator = ConfidenceCalibrator()
        self.research_engine = research_engine
    
    def process(self, user_input: str, raw_response: str, 
                thinking_data: Dict = None, research_data: Dict = None) -> Dict[str, Any]:
        """
        Full processing pipeline:
        1. Research integration
        2. Thinking integration
        3. Confidence calibration
        4. Quality enhancement
        """
        result = {
            "original_response": raw_response,
            "enhanced_response": raw_response,
            "thinking_explanation": "",
            "confidence": 0.5,
            "quality_metrics": {}
        }
        
        # 1. Determine confidence
        research_confidence = research_data.get("confidence", 0.5) if research_data else 0.5
        thinking_confidence = thinking_data.get("confidence", 0.5) if thinking_data else 0.5
        overall_confidence = (research_confidence * 0.6 + thinking_confidence * 0.4)
        result["confidence"] = overall_confidence
        
        # 2. Integrate thinking
        if thinking_data:
            integration = self.thinking_integrator.integrate_thinking(thinking_data, raw_response)
            result["enhanced_response"] = integration["final_response"]
            result["thinking_explanation"] = integration["thinking_explanation"]
            result["insight_coverage"] = integration["insight_coverage"]
        
        # 3. Calibrate confidence in language
        result["enhanced_response"] = self.confidence_calibrator.calibrate_response(
            result["enhanced_response"],
            research_confidence,
            thinking_confidence
        )
        
        return result


class SourceIntegrator:
    """Properly integrate and cite sources in responses"""
    
    def __init__(self):
        self.source_reliability = {
            "wikipedia": 0.7,
            "scientific_journal": 0.95,
            "news_outlet": 0.75,
            "government": 0.9,
            "corporate": 0.6,
            "blog": 0.4,
            "social_media": 0.3,
            "unknown": 0.5
        }
    
    def integrate_sources(self, response: str, sources: List[Dict]) -> str:
        """Add proper source citations to response"""
        if not sources:
            return response
        
        # Sort by reliability
        sorted_sources = sorted(
            sources, 
            key=lambda s: self.source_reliability.get(s.get("type", "unknown"), 0.5),
            reverse=True
        )
        
        # Format citations
        citation_text = self._format_citations(sorted_sources[:3])
        
        # Add to response
        if "*Source" not in response and "*Based on" not in response:
            if "\n\n" in response:
                response = response.rstrip()
                response += f"\n\n*{citation_text}*"
            else:
                response += f"\n\n*{citation_text}*"
        
        return response
    
    def _format_citations(self, sources: List[Dict]) -> str:
        """Format source list for display"""
        if not sources:
            return "Based on available research."
        
        source_names = []
        for source in sources:
            name = source.get("source") or source.get("url") or source.get("name", "research")
            # Clean up URL to just domain
            if "http" in name:
                match = re.search(r'//([^/]+)', name)
                if match:
                    name = match.group(1).replace("www.", "")
            source_names.append(name)
        
        if len(source_names) == 1:
            return f"Source: {source_names[0]}"
        else:
            return f"Sources: {', '.join(source_names)}"
    
    def calculate_source_score(self, sources: List[Dict]) -> float:
        """Calculate overall reliability score for sources"""
        if not sources:
            return 0.5
        
        scores = [
            self.source_reliability.get(s.get("type", "unknown"), 0.5)
            for s in sources
        ]
        
        return sum(scores) / len(scores)


if __name__ == "__main__":
    print("=== Thinking Integration Test ===\n")
    
    # Test thinking integration
    integrator = ThinkingIntegration()
    
    thinking_data = {
        "thought_process": {
            "question_analysis": {
                "question_type": "factual",
                "difficulty_level": "medium"
            },
            "key_insights": [
                "User wants specific information about banks",
                "Need to provide ranking with criteria"
            ],
            "reasoning_applied": "factual_retrieval"
        },
        "confidence": 0.75
    }
    
    response = "Many banks are large and handle lots of money."
    
    result = integrator.integrate_thinking(thinking_data, response)
    
    print(f"Original: {response}")
    print(f"Enhanced: {result['final_response']}")
    print(f"Insight Coverage: {result['insight_coverage']:.0f}%")
    print(f"Thinking Explanation: {result['thinking_explanation']}")
    
    print("\n=== Confidence Calibration Test ===\n")
    
    calibrator = ConfidenceCalibrator()
    
    test_response = "The largest bank in the world is ICBC with $5.5 trillion in assets."
    
    for conf in [0.9, 0.7, 0.5, 0.3]:
        calibrated = calibrator.calibrate_response(test_response, conf)
        print(f"Confidence {conf:.0%}: {calibrated[:80]}...")
