"""
Enhanced Reasoning Engine v3 - Advanced Cognitive Processing
Implements multi-stage reasoning with uncertainty quantification and meta-reasoning
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math
from datetime import datetime

class ReasoningType(Enum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    COUNTERFACTUAL = "counterfactual"

class ConfidenceLevel(Enum):
    VERY_LOW = (0.0, 0.2, "highly uncertain")
    LOW = (0.2, 0.4, "uncertain")
    MODERATE = (0.4, 0.6, "moderately confident")
    HIGH = (0.6, 0.8, "confident")
    VERY_HIGH = (0.8, 1.0, "highly confident")

@dataclass
class ReasoningStep:
    step_id: int
    reasoning_type: ReasoningType
    premise: str
    inference: str
    conclusion: str
    confidence: float
    evidence: List[str]
    assumptions: List[str]
    potential_flaws: List[str]

@dataclass
class ReasoningChain:
    steps: List[ReasoningStep]
    final_conclusion: str
    overall_confidence: float
    reasoning_path: str
    alternative_conclusions: List[str]
    uncertainty_factors: List[str]

class EnhancedReasoningEngine:
    """
    Advanced reasoning engine with multi-stage processing and uncertainty quantification
    """
    
    def __init__(self):
        self.reasoning_patterns = self._load_reasoning_patterns()
        self.domain_knowledge = self._load_domain_knowledge()
        self.fallacy_detectors = self._init_fallacy_detectors()
        
    def _load_reasoning_patterns(self) -> Dict[str, Any]:
        """Load common reasoning patterns and templates"""
        return {
            "causal_indicators": [
                "because", "since", "due to", "caused by", "results in",
                "leads to", "triggers", "brings about", "stems from"
            ],
            "evidence_indicators": [
                "studies show", "research indicates", "data suggests",
                "evidence points to", "statistics reveal", "experiments demonstrate"
            ],
            "uncertainty_indicators": [
                "might", "could", "possibly", "perhaps", "likely",
                "probably", "seems", "appears", "suggests"
            ],
            "contradiction_indicators": [
                "however", "but", "although", "despite", "nevertheless",
                "on the other hand", "conversely", "in contrast"
            ]
        }
    
    def _load_domain_knowledge(self) -> Dict[str, Any]:
        """Load domain-specific reasoning rules"""
        return {
            "science": {
                "requires_evidence": True,
                "falsifiability": True,
                "peer_review": True,
                "confidence_threshold": 0.7
            },
            "philosophy": {
                "logical_consistency": True,
                "premise_validity": True,
                "argument_soundness": True,
                "confidence_threshold": 0.6
            },
            "mathematics": {
                "logical_proof": True,
                "axiom_based": True,
                "deductive_certainty": True,
                "confidence_threshold": 0.9
            },
            "everyday": {
                "common_sense": True,
                "practical_experience": True,
                "heuristic_based": True,
                "confidence_threshold": 0.5
            }
        }
    
    def _init_fallacy_detectors(self) -> Dict[str, Any]:
        """Initialize logical fallacy detection patterns"""
        return {
            "ad_hominem": {
                "patterns": ["you're wrong because you", "can't trust them because"],
                "description": "Attacking the person rather than the argument"
            },
            "straw_man": {
                "patterns": ["so you're saying", "your position is that"],
                "description": "Misrepresenting someone's argument"
            },
            "false_dichotomy": {
                "patterns": ["either", "only two options", "must choose"],
                "description": "Presenting only two options when more exist"
            },
            "slippery_slope": {
                "patterns": ["if we allow", "this will lead to", "next thing"],
                "description": "Assuming one event will lead to extreme consequences"
            }
        }
    
    def analyze_reasoning_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the logical structure of reasoning in text"""
        
        # Identify reasoning indicators
        causal_matches = self._find_patterns(text, self.reasoning_patterns["causal_indicators"])
        evidence_matches = self._find_patterns(text, self.reasoning_patterns["evidence_indicators"])
        uncertainty_matches = self._find_patterns(text, self.reasoning_patterns["uncertainty_indicators"])
        contradiction_matches = self._find_patterns(text, self.reasoning_patterns["contradiction_indicators"])
        
        # Determine primary reasoning type
        reasoning_type = self._classify_reasoning_type(text, {
            "causal": causal_matches,
            "evidence": evidence_matches,
            "uncertainty": uncertainty_matches,
            "contradiction": contradiction_matches
        })
        
        # Extract premises and conclusions
        premises, conclusions = self._extract_premises_conclusions(text)
        
        # Assess logical structure
        structure_quality = self._assess_logical_structure(premises, conclusions)
        
        return {
            "reasoning_type": reasoning_type,
            "premises": premises,
            "conclusions": conclusions,
            "structure_quality": structure_quality,
            "causal_chains": causal_matches,
            "evidence_strength": len(evidence_matches),
            "uncertainty_level": len(uncertainty_matches) / max(1, len(text.split())),
            "contradictions": contradiction_matches
        }
    
    def generate_reasoning_chain(self, query: str, context: str = "") -> ReasoningChain:
        """Generate a complete reasoning chain for a query"""
        
        # Step 1: Analyze the query
        query_analysis = self._analyze_query_type(query)
        domain = self._identify_domain(query + " " + context)
        
        # Step 2: Generate reasoning steps
        steps = []
        step_id = 1
        
        # Initial premise identification
        premises = self._extract_key_premises(query, context)
        for premise in premises:
            step = ReasoningStep(
                step_id=step_id,
                reasoning_type=ReasoningType.DEDUCTIVE,
                premise=premise,
                inference="Given premise",
                conclusion=f"We accept: {premise}",
                confidence=0.8,
                evidence=[context] if context else [],
                assumptions=[],
                potential_flaws=[]
            )
            steps.append(step)
            step_id += 1
        
        # Generate inference steps
        inference_steps = self._generate_inference_steps(query, premises, domain)
        steps.extend(inference_steps)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_chain_confidence(steps)
        
        # Generate alternative conclusions
        alternatives = self._generate_alternative_conclusions(query, steps)
        
        # Identify uncertainty factors
        uncertainty_factors = self._identify_uncertainty_factors(steps, domain)
        
        # Create final reasoning chain
        final_conclusion = steps[-1].conclusion if steps else "No conclusion reached"
        reasoning_path = " → ".join([step.inference for step in steps])
        
        return ReasoningChain(
            steps=steps,
            final_conclusion=final_conclusion,
            overall_confidence=overall_confidence,
            reasoning_path=reasoning_path,
            alternative_conclusions=alternatives,
            uncertainty_factors=uncertainty_factors
        )
    
    def meta_reason_about_reasoning(self, reasoning_chain: ReasoningChain) -> Dict[str, Any]:
        """Perform meta-reasoning about the quality of reasoning"""
        
        # Assess reasoning quality
        quality_metrics = {
            "logical_consistency": self._check_logical_consistency(reasoning_chain.steps),
            "evidence_strength": self._assess_evidence_strength(reasoning_chain.steps),
            "assumption_validity": self._check_assumptions(reasoning_chain.steps),
            "potential_biases": self._identify_potential_biases(reasoning_chain.steps),
            "completeness": self._assess_completeness(reasoning_chain.steps)
        }
        
        # Generate improvement suggestions
        improvements = self._suggest_improvements(reasoning_chain, quality_metrics)
        
        # Calculate meta-confidence
        meta_confidence = self._calculate_meta_confidence(quality_metrics)
        
        return {
            "quality_metrics": quality_metrics,
            "improvements": improvements,
            "meta_confidence": meta_confidence,
            "reasoning_strengths": self._identify_strengths(quality_metrics),
            "reasoning_weaknesses": self._identify_weaknesses(quality_metrics)
        }
    
    def _find_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """Find pattern matches in text"""
        matches = []
        text_lower = text.lower()
        for pattern in patterns:
            if pattern.lower() in text_lower:
                matches.append(pattern)
        return matches
    
    def _classify_reasoning_type(self, text: str, indicators: Dict[str, List[str]]) -> ReasoningType:
        """Classify the primary type of reasoning used"""
        scores = {}
        for reasoning_type, matches in indicators.items():
            scores[reasoning_type] = len(matches)
        
        if scores.get("causal", 0) > 0:
            return ReasoningType.CAUSAL
        elif scores.get("evidence", 0) > 0:
            return ReasoningType.INDUCTIVE
        elif "if" in text.lower() and "then" in text.lower():
            return ReasoningType.DEDUCTIVE
        elif "like" in text.lower() or "similar" in text.lower():
            return ReasoningType.ANALOGICAL
        else:
            return ReasoningType.ABDUCTIVE
    
    def _extract_premises_conclusions(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract premises and conclusions from text"""
        sentences = re.split(r'[.!?]+', text)
        premises = []
        conclusions = []
        
        conclusion_indicators = ["therefore", "thus", "hence", "so", "consequently"]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            is_conclusion = any(indicator in sentence.lower() for indicator in conclusion_indicators)
            
            if is_conclusion:
                conclusions.append(sentence)
            else:
                premises.append(sentence)
        
        return premises, conclusions
    
    def _assess_logical_structure(self, premises: List[str], conclusions: List[str]) -> float:
        """Assess the quality of logical structure"""
        if not premises or not conclusions:
            return 0.3
        
        # Simple heuristic: ratio of conclusions to premises should be reasonable
        ratio = len(conclusions) / len(premises)
        if 0.2 <= ratio <= 0.8:
            return 0.8
        elif 0.1 <= ratio <= 1.0:
            return 0.6
        else:
            return 0.4
    
    def _analyze_query_type(self, query: str) -> Dict[str, Any]:
        """Analyze what type of query this is"""
        query_lower = query.lower()
        
        question_words = ["what", "why", "how", "when", "where", "who", "which"]
        is_question = any(word in query_lower for word in question_words) or query.endswith("?")
        
        causal_words = ["why", "because", "cause", "reason"]
        is_causal = any(word in query_lower for word in causal_words)
        
        comparison_words = ["better", "worse", "compare", "versus", "vs"]
        is_comparison = any(word in query_lower for word in comparison_words)
        
        return {
            "is_question": is_question,
            "is_causal": is_causal,
            "is_comparison": is_comparison,
            "complexity": len(query.split()),
            "requires_evidence": "evidence" in query_lower or "proof" in query_lower
        }
    
    def _identify_domain(self, text: str) -> str:
        """Identify the domain of discourse"""
        text_lower = text.lower()
        
        science_words = ["study", "research", "experiment", "data", "hypothesis", "theory"]
        math_words = ["equation", "formula", "proof", "theorem", "calculate", "number"]
        philosophy_words = ["ethics", "morality", "consciousness", "existence", "meaning", "truth"]
        
        if any(word in text_lower for word in science_words):
            return "science"
        elif any(word in text_lower for word in math_words):
            return "mathematics"
        elif any(word in text_lower for word in philosophy_words):
            return "philosophy"
        else:
            return "everyday"
    
    def _extract_key_premises(self, query: str, context: str) -> List[str]:
        """Extract key premises from query and context"""
        premises = []
        
        # Extract from context
        if context:
            context_sentences = re.split(r'[.!?]+', context)
            for sentence in context_sentences[:3]:  # Take first 3 sentences
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:
                    premises.append(sentence)
        
        # Extract implicit premises from query
        if "because" in query.lower():
            parts = query.lower().split("because")
            if len(parts) > 1:
                premises.append(f"Given that {parts[1].strip()}")
        
        return premises[:5]  # Limit to 5 premises
    
    def _generate_inference_steps(self, query: str, premises: List[str], domain: str) -> List[ReasoningStep]:
        """Generate inference steps based on premises"""
        steps = []
        step_id = len(premises) + 1
        
        # Generate a simple inference chain
        if premises:
            # Combine premises
            combined_premise = " and ".join(premises[:2])
            
            step = ReasoningStep(
                step_id=step_id,
                reasoning_type=ReasoningType.DEDUCTIVE,
                premise=combined_premise,
                inference="Logical combination of premises",
                conclusion=f"From the given information, we can infer relevant patterns",
                confidence=0.7,
                evidence=premises,
                assumptions=["Premises are accurate", "Logic is valid"],
                potential_flaws=["Premises might be incomplete", "Hidden assumptions"]
            )
            steps.append(step)
            step_id += 1
        
        # Generate domain-specific inference
        domain_rules = self.domain_knowledge.get(domain, self.domain_knowledge["everyday"])
        confidence_threshold = domain_rules["confidence_threshold"]
        
        step = ReasoningStep(
            step_id=step_id,
            reasoning_type=ReasoningType.INDUCTIVE,
            premise="Domain knowledge and inference patterns",
            inference=f"Applying {domain} reasoning principles",
            conclusion="Reaching conclusion based on available evidence and reasoning",
            confidence=confidence_threshold,
            evidence=[f"{domain} domain knowledge"],
            assumptions=[f"Domain principles apply", "Evidence is sufficient"],
            potential_flaws=["Domain knowledge might be incomplete", "Context might be missing"]
        )
        steps.append(step)
        
        return steps
    
    def _calculate_chain_confidence(self, steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence in reasoning chain"""
        if not steps:
            return 0.0
        
        # Use geometric mean to account for weakest link
        confidences = [step.confidence for step in steps]
        product = 1.0
        for conf in confidences:
            product *= conf
        
        return product ** (1.0 / len(confidences))
    
    def _generate_alternative_conclusions(self, query: str, steps: List[ReasoningStep]) -> List[str]:
        """Generate alternative conclusions"""
        alternatives = []
        
        if steps:
            main_conclusion = steps[-1].conclusion
            
            # Generate opposite conclusion
            if "is" in main_conclusion:
                alternatives.append(main_conclusion.replace("is", "is not"))
            
            # Generate uncertain version
            alternatives.append(f"It's possible that {main_conclusion.lower()}")
            
            # Generate conditional version
            alternatives.append(f"If our assumptions are correct, then {main_conclusion.lower()}")
        
        return alternatives[:3]
    
    def _identify_uncertainty_factors(self, steps: List[ReasoningStep], domain: str) -> List[str]:
        """Identify factors that contribute to uncertainty"""
        factors = []
        
        # Check for assumption-heavy steps
        for step in steps:
            if len(step.assumptions) > 2:
                factors.append(f"Step {step.step_id} relies on multiple assumptions")
            
            if step.confidence < 0.6:
                factors.append(f"Low confidence in step {step.step_id}")
            
            if step.potential_flaws:
                factors.append(f"Potential flaws identified in step {step.step_id}")
        
        # Domain-specific uncertainty factors
        if domain == "science":
            factors.append("Scientific claims require empirical validation")
        elif domain == "philosophy":
            factors.append("Philosophical conclusions are often debatable")
        
        return factors
    
    def _check_logical_consistency(self, steps: List[ReasoningStep]) -> float:
        """Check logical consistency across steps"""
        # Simple heuristic: look for contradictions
        conclusions = [step.conclusion.lower() for step in steps]
        
        contradiction_count = 0
        for i, conclusion1 in enumerate(conclusions):
            for j, conclusion2 in enumerate(conclusions[i+1:], i+1):
                if "not" in conclusion1 and conclusion1.replace("not", "").strip() in conclusion2:
                    contradiction_count += 1
                elif "not" in conclusion2 and conclusion2.replace("not", "").strip() in conclusion1:
                    contradiction_count += 1
        
        consistency_score = max(0.0, 1.0 - (contradiction_count * 0.3))
        return consistency_score
    
    def _assess_evidence_strength(self, steps: List[ReasoningStep]) -> float:
        """Assess the strength of evidence across steps"""
        if not steps:
            return 0.0
        
        total_evidence = sum(len(step.evidence) for step in steps)
        avg_evidence = total_evidence / len(steps)
        
        # Normalize to 0-1 scale
        return min(1.0, avg_evidence / 3.0)
    
    def _check_assumptions(self, steps: List[ReasoningStep]) -> float:
        """Check the validity of assumptions"""
        if not steps:
            return 0.0
        
        total_assumptions = sum(len(step.assumptions) for step in steps)
        avg_assumptions = total_assumptions / len(steps)
        
        # Fewer assumptions generally better
        return max(0.0, 1.0 - (avg_assumptions / 5.0))
    
    def _identify_potential_biases(self, steps: List[ReasoningStep]) -> List[str]:
        """Identify potential cognitive biases"""
        biases = []
        
        # Check for confirmation bias
        evidence_types = []
        for step in steps:
            for evidence in step.evidence:
                evidence_types.append(evidence.lower())
        
        if len(set(evidence_types)) < len(evidence_types) * 0.7:
            biases.append("Potential confirmation bias - similar evidence sources")
        
        # Check for availability heuristic
        recent_words = ["recent", "lately", "now", "today", "current"]
        for step in steps:
            if any(word in step.premise.lower() for word in recent_words):
                biases.append("Potential availability heuristic - overweighting recent information")
                break
        
        return biases
    
    def _assess_completeness(self, steps: List[ReasoningStep]) -> float:
        """Assess completeness of reasoning"""
        # Simple heuristic based on number of steps and evidence
        if len(steps) < 2:
            return 0.3
        elif len(steps) < 4:
            return 0.6
        else:
            return 0.8
    
    def _suggest_improvements(self, reasoning_chain: ReasoningChain, quality_metrics: Dict[str, Any]) -> List[str]:
        """Suggest improvements to reasoning"""
        suggestions = []
        
        if quality_metrics["logical_consistency"] < 0.7:
            suggestions.append("Review for logical contradictions")
        
        if quality_metrics["evidence_strength"] < 0.5:
            suggestions.append("Gather more supporting evidence")
        
        if quality_metrics["assumption_validity"] < 0.6:
            suggestions.append("Examine and validate key assumptions")
        
        if reasoning_chain.overall_confidence < 0.5:
            suggestions.append("Consider alternative explanations")
        
        return suggestions
    
    def _calculate_meta_confidence(self, quality_metrics: Dict[str, Any]) -> float:
        """Calculate confidence in the reasoning process itself"""
        scores = []
        for metric, value in quality_metrics.items():
            if isinstance(value, (int, float)):
                scores.append(value)
        
        if not scores:
            return 0.5
        
        return sum(scores) / len(scores)
    
    def _identify_strengths(self, quality_metrics: Dict[str, Any]) -> List[str]:
        """Identify strengths in reasoning"""
        strengths = []
        
        if quality_metrics.get("logical_consistency", 0) > 0.8:
            strengths.append("Strong logical consistency")
        
        if quality_metrics.get("evidence_strength", 0) > 0.7:
            strengths.append("Well-supported with evidence")
        
        if quality_metrics.get("completeness", 0) > 0.7:
            strengths.append("Comprehensive reasoning process")
        
        return strengths
    
    def _identify_weaknesses(self, quality_metrics: Dict[str, Any]) -> List[str]:
        """Identify weaknesses in reasoning"""
        weaknesses = []
        
        if quality_metrics.get("logical_consistency", 1) < 0.6:
            weaknesses.append("Logical inconsistencies present")
        
        if quality_metrics.get("evidence_strength", 1) < 0.5:
            weaknesses.append("Insufficient evidence")
        
        if quality_metrics.get("assumption_validity", 1) < 0.5:
            weaknesses.append("Questionable assumptions")
        
        return weaknesses

# Example usage and testing
if __name__ == "__main__":
    engine = EnhancedReasoningEngine()
    
    # Test reasoning chain generation
    query = "Why do people procrastinate on important tasks?"
    context = "Research shows that procrastination is often linked to fear of failure and perfectionism."
    
    reasoning_chain = engine.generate_reasoning_chain(query, context)
    meta_analysis = engine.meta_reason_about_reasoning(reasoning_chain)
    
    print("=== Reasoning Chain ===")
    print(f"Query: {query}")
    print(f"Final Conclusion: {reasoning_chain.final_conclusion}")
    print(f"Overall Confidence: {reasoning_chain.overall_confidence:.2f}")
    print(f"Reasoning Path: {reasoning_chain.reasoning_path}")
    
    print("\n=== Meta-Analysis ===")
    print(f"Meta-Confidence: {meta_analysis['meta_confidence']:.2f}")
    print(f"Strengths: {meta_analysis['reasoning_strengths']}")
    print(f"Weaknesses: {meta_analysis['reasoning_weaknesses']}")
    print(f"Improvements: {meta_analysis['improvements']}")