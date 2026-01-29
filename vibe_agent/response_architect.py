"""
Response Architecture System
Builds DeepSeek-style structured responses with proper formatting
"""

import re
import random
from typing import Dict, List, Any, Optional
from datetime import datetime

# 🆕 Phase 6 Upgrade
from response_formatter import ResponseFormatter


class ResponseArchitect:
    """Build responses with DeepSeek's characteristic structure"""
    
    RESPONSE_STRUCTURES = {
        "factual_direct": """{answer}

**Details:**
{details}

{source_line}
""",
        
        "superlative": """The {superlative} {subject} is **{answer}**{value_details}.

**Key Facts:**
{supporting_facts}

**Context:**
{context}

{source_line}
""",
        
        "comparative": """Based on {data_source} ({data_year}):

{comparison_list}

**Key factors:** {factors}

**Note:** {note}

{source_line}
""",
        
        "definition": """**{subject}** {definition}

**Key aspects:**
{key_points}

**Current relevance:**
{relevance}

{source_line}
""",
        
        "enumeration": """The main {subject} include:

{list_items}

**Details:**
{details}

{source_line}
""",
        
        "explanation": """{subject} {mechanism}

**How it works:**
{how_it_works}

**Key points:**
{key_points}

{source_line}
""",
        
        "simple": """{answer}

{source_line}
""",
        
        "recommendation": """Based on {criteria}, here are {recommendation_intro}:

{recommendations}

**Key considerations:**
{considerations}

**Note:** {note}

{source_line}
"""
    }
    
    def __init__(self):
        self.formatting_enforcer = FormattingEnforcer()
    
    def build_response(self, facts: Dict, question: str, 
                       confidence: float = 0.8) -> str:
        """Construct complete DeepSeek-style response"""
        
        # 1. Determine response type
        response_type = self._determine_response_type(question, facts)
        
        # 2. Get structure template
        structure = self.RESPONSE_STRUCTURES.get(
            response_type, 
            self.RESPONSE_STRUCTURES["factual_direct"]
        )
        
        # 3. Prepare all components
        components = self._prepare_components(facts, question, response_type, confidence)
        
        # 4. Fill structure
        try:
            response = structure.format(**components)
        except KeyError as e:
            # Fallback for missing components
            response = self._build_fallback_response(facts, components)
        
        # 5. Apply formatting
        response = self.formatting_enforcer.enforce_formatting(response)
        
        # 6. Add confidence qualifier if needed
        if confidence < 0.6:
            response = self._add_confidence_qualifier(response, confidence)
        
        return response.strip()
    
    def _determine_response_type(self, question: str, facts: Dict) -> str:
        """Determine the best response structure"""
        q_type = facts.get("question_type", "")
        question_lower = question.lower()
        
        if q_type == "superlative" or any(w in question_lower for w in 
            ["largest", "biggest", "smallest", "most", "best", "worst"]):
            return "superlative"
        
        if q_type == "ranking" or "top" in question_lower:
            return "comparative"
        
        if q_type == "definition" or "what is" in question_lower:
            return "definition"
        
        if q_type == "enumeration" or any(w in question_lower for w in 
            ["what are", "list", "name the"]):
            return "enumeration"
        
        if q_type in ["mechanism", "procedural"] or any(w in question_lower for w in 
            ["how does", "how do", "explain"]):
            return "explanation"
        
        if q_type == "recommendation" or any(w in question_lower for w in
            ["suggest", "recommend", "should i"]):
            return "recommendation"
        
        # Default based on content length
        primary = facts.get("primary_answer", "")
        if len(primary.split()) > 50:
            return "factual_direct"
        
        return "simple"
    
    def _prepare_components(self, facts: Dict, question: str, 
                            response_type: str, confidence: float) -> Dict:
        """Prepare all components for the template"""
        
        primary_answer = facts.get("primary_answer", "")
        entities = facts.get("key_entities", [])
        numbers = facts.get("key_numbers", [])
        details_list = facts.get("supporting_details", [])
        sources = facts.get("sources", [])
        subject = facts.get("subject", "this topic")
        
        # Base components
        components = {
            "answer": primary_answer,
            "subject": subject,
            "details": self._format_details(details_list),
            "source_line": self._format_source_line(sources),
            "data_year": str(datetime.now().year),
            "data_source": self._get_data_source_name(sources),
            "note": "Information may vary; verify for current accuracy.",
            "confidence": confidence
        }
        
        # Superlative-specific
        if response_type == "superlative":
            components["superlative"] = self._extract_superlative(question)
            components["value_details"] = ""
            if numbers:
                components["value_details"] = f" with {numbers[0]}"
            components["supporting_facts"] = self._format_as_bullets(details_list[:3])
            components["context"] = details_list[-1] if details_list else "This represents current data."
        
        # Comparative/ranking-specific
        if response_type == "comparative":
            components["comparison_list"] = self._format_ranked_list(entities, numbers)
            components["factors"] = "Scale, reliability, and current data"
        
        # Definition-specific
        if response_type == "definition":
            components["definition"] = self._extract_definition(primary_answer)
            components["key_points"] = self._format_as_bullets(details_list[:4])
            components["relevance"] = self._generate_relevance(subject, details_list)
        
        # Enumeration-specific
        if response_type == "enumeration":
            components["list_items"] = self._format_as_numbered_list(entities[:7])
        
        # Explanation-specific
        if response_type == "explanation":
            components["mechanism"] = self._extract_mechanism(primary_answer)
            components["how_it_works"] = self._format_as_bullets(details_list[:3])
            components["key_points"] = self._format_as_bullets(details_list[3:6])
        
        # Recommendation-specific
        if response_type == "recommendation":
            components["criteria"] = "current performance data and user reviews"
            components["recommendation_intro"] = f"top {subject.lower()} options"
            components["recommendations"] = self._format_recommendations(entities, details_list)
            components["considerations"] = self._format_as_bullets([
                "Your specific needs and preferences",
                "Current market conditions",
                "Long-term reliability"
            ])
        
        return components
    
    def _format_details(self, details: List[str]) -> str:
        """Format supporting details"""
        if not details:
            return "Additional information available upon request."
        
        # Join if short, bullet if many
        if len(details) <= 2:
            return " ".join(details)
        else:
            return "\n".join([f"• {d}" for d in details[:4]])
    
    def _format_source_line(self, sources: List[Dict]) -> str:
        """Format sources in DeepSeek style"""
        if not sources:
            return "*Based on available research data.*"
        
        source_names = []
        for source in sources[:2]:
            name = source.get("name", source.get("type", "research"))
            if name and name not in source_names:
                # Clean up domain names
                name = name.replace("www.", "").split("/")[0]
                if name not in ["duckduckgo", ""]:
                    source_names.append(name)
        
        if not source_names:
            return "*Based on available research data.*"
        
        if len(source_names) == 1:
            return f"*Source: {source_names[0]}*"
        else:
            return f"*Sources: {', '.join(source_names)}*"
    
    def _get_data_source_name(self, sources: List[Dict]) -> str:
        """Get a nice name for the data source"""
        if not sources:
            return "available data"
        
        name = sources[0].get("name", "research")
        if name in ["duckduckgo", "web"]:
            return "current research"
        return name
    
    def _format_as_bullets(self, items: List[str]) -> str:
        """Format as bullet points"""
        if not items:
            return "• Information pending"
        return "\n".join([f"• {item}" for item in items])
    
    def _format_as_numbered_list(self, items: List[str]) -> str:
        """Format as numbered list with bold"""
        if not items:
            return "1. Information pending"
        return "\n".join([f"{i+1}. **{item}**" for i, item in enumerate(items)])
    
    def _format_ranked_list(self, entities: List[str], numbers: List[str]) -> str:
        """Format as a ranked list with optional values"""
        if not entities:
            return "Rankings data pending."
        
        lines = []
        for i, entity in enumerate(entities[:10]):
            line = f"{i+1}. **{entity}**"
            if i < len(numbers):
                line += f" ({numbers[i]})"
            lines.append(line)
        
        return "\n".join(lines)
    
    def _format_recommendations(self, entities: List[str], details: List[str]) -> str:
        """Format recommendations"""
        if not entities:
            return "• Consult current rankings for recommendations"
        
        lines = []
        for i, entity in enumerate(entities[:7]):
            line = f"{i+1}. **{entity}**"
            if i < len(details):
                # Add brief detail
                brief = details[i][:80] if len(details[i]) > 80 else details[i]
                line += f" - {brief}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def _extract_superlative(self, question: str) -> str:
        """Extract the superlative word from question"""
        superlatives = ["largest", "biggest", "smallest", "most", "best", 
                       "worst", "fastest", "slowest", "highest", "lowest",
                       "top", "leading"]
        question_lower = question.lower()
        for sup in superlatives:
            if sup in question_lower:
                return sup
        return "top"
    
    def _extract_definition(self, text: str) -> str:
        """Extract or format as definition"""
        if " is " in text.lower():
            return text
        return f"is {text}"
    
    def _extract_mechanism(self, text: str) -> str:
        """Extract mechanism/how it works"""
        mechanism_starters = ["works by", "operates by", "functions through"]
        text_lower = text.lower()
        
        for starter in mechanism_starters:
            if starter in text_lower:
                idx = text_lower.index(starter)
                return text[idx:]
        
        return text
    
    def _generate_relevance(self, subject: str, details: List[str]) -> str:
        """Generate relevance statement"""
        if details:
            return details[-1] if len(details[-1]) < 150 else details[-1][:150] + "..."
        return f"Understanding {subject.lower()} remains important in current contexts."
    
    def _build_fallback_response(self, facts: Dict, components: Dict) -> str:
        """Build fallback when template fails"""
        answer = facts.get("primary_answer", components.get("answer", ""))
        source_line = components.get("source_line", "")
        
        return f"{answer}\n\n{source_line}"
    
    def _add_confidence_qualifier(self, response: str, confidence: float) -> str:
        """Add qualifier for low confidence"""
        if confidence < 0.4:
            qualifier = "Note: This information has moderate uncertainty and should be verified."
        else:
            qualifier = "Note: Please verify for the most current information."
        
        if "*Note:" not in response and "**Note:**" not in response:
            response = response.rstrip() + f"\n\n*{qualifier}*"
        
        return response


class FormattingEnforcer:
    """Ensure all responses follow DeepSeek's formatting style"""
    
    def enforce_formatting(self, response_text: str) -> str:
        """Apply consistent formatting rules"""
        
        if not response_text:
            return response_text
        
        lines = response_text.strip().split('\n')
        formatted_lines = []
        
        for line in lines:
            formatted_line = self._format_line(line)
            formatted_lines.append(formatted_line)
        
        # Reassemble
        formatted_text = '\n'.join(formatted_lines)
        
        # Apply global rules
        formatted_text = self._apply_global_rules(formatted_text)
        
        return formatted_text
    
    def _format_line(self, line: str) -> str:
        """Format individual line based on content"""
        line = line.rstrip()
        
        if not line:
            return ""
        
        # Already formatted headers - leave alone
        if line.startswith('**') and ':**' in line:
            return line
        
        # Detect section headers (ends with colon, not too long, not bullet)
        if (line.endswith(':') and len(line) < 50 and 
            not line.startswith(('•', '-', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9'))):
            if not line.startswith('**'):
                return f"**{line}**"
        
        # Bullet points - ensure consistent style
        if line.lstrip().startswith(('-', '*')) and not line.lstrip().startswith('**'):
            # Convert to bullet
            content = line.lstrip('-*').strip()
            indent = len(line) - len(line.lstrip())
            return ' ' * indent + f"• {content}"
        
        return line
    
    def _apply_global_rules(self, text: str) -> str:
        """Apply rules to entire response"""
        
        # Rule 1: No more than 2 consecutive blank lines
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        
        # Rule 2: Ensure blank line before headers
        lines = text.split('\n')
        new_lines = []
        for i, line in enumerate(lines):
            if line.startswith('**') and line.endswith('**') and len(line) < 60:
                # This is a header
                if i > 0 and new_lines and new_lines[-1] != '':
                    new_lines.append('')
            new_lines.append(line)
        
        # Rule 3: Clean trailing whitespace
        new_lines = [line.rstrip() for line in new_lines]
        
        # Rule 4: Ensure proper ending
        if new_lines:
            last_line = new_lines[-1]
            if last_line and not last_line.rstrip().endswith(('.', '!', '?', '*')):
                new_lines[-1] = last_line.rstrip() + '.'
        
        # Rule 5: Remove leading blank lines
        while new_lines and new_lines[0] == '':
            new_lines.pop(0)
        
        # Rule 6: Remove trailing blank lines (keep max 1)
        while len(new_lines) > 1 and new_lines[-1] == '' and new_lines[-2] == '':
            new_lines.pop()
        
        return '\n'.join(new_lines)


class ThinkingResponseAligner:
    """Ensure thinking process (NEURAL DRIFT) informs the response"""
    
    def __init__(self):
        self.insight_actions = {
            "definitional": "Provide clear definition with examples",
            "verifiable": "Include specific facts and sources",
            "low complexity": "Keep answer straightforward and concise",
            "high confidence": "State answer directly without hedging",
            "procedural": "Provide step-by-step guidance",
            "comparative": "Show differences and similarities clearly",
            "causal": "Explain causes and effects",
            "exploratory": "Invite deeper discussion"
        }
    
    def align_response(self, thinking_data: Dict, response: str, 
                       facts: Dict = None) -> str:
        """Ensure response reflects thinking process insights"""
        
        # 1. Parse insights from thinking
        insights = self._parse_thinking(thinking_data)
        
        # 2. Check which insights are reflected
        reflection_check = self._check_reflection(response, insights)
        
        # 3. Enhance with missing insights
        if reflection_check["missing"]:
            response = self._enhance_with_insights(response, reflection_check["missing"], facts)
        
        return response
    
    def _parse_thinking(self, thinking_data: Dict) -> List[Dict]:
        """Extract actionable insights from thinking data"""
        insights = []
        
        if not thinking_data:
            return insights
        
        # From reasoning process
        thought_process = thinking_data.get("thought_process", {})
        
        # Question analysis
        qa = thought_process.get("question_analysis", {})
        if qa:
            insights.append({
                "type": "question_type",
                "content": qa.get("question_type", ""),
                "action": self.insight_actions.get(qa.get("question_type", ""), "")
            })
        
        # Key insights
        for insight in thought_process.get("key_insights", []):
            insights.append({
                "type": "key_insight",
                "content": insight,
                "action": ""
            })
        
        # Confidence
        confidence = thinking_data.get("confidence", 0.5)
        if confidence > 0.7:
            insights.append({
                "type": "confidence",
                "content": "high",
                "action": "State answer directly"
            })
        elif confidence < 0.4:
            insights.append({
                "type": "confidence",
                "content": "low",
                "action": "Add appropriate qualifiers"
            })
        
        return insights
    
    def _check_reflection(self, response: str, insights: List[Dict]) -> Dict:
        """Check which insights are reflected in response"""
        reflected = []
        missing = []
        
        response_lower = response.lower()
        
        for insight in insights:
            content = insight.get("content", "").lower()
            
            # Check if content keywords appear in response
            if content:
                keywords = [w for w in content.split() if len(w) > 3]
                if any(kw in response_lower for kw in keywords):
                    reflected.append(insight)
                else:
                    missing.append(insight)
        
        return {
            "reflected": reflected,
            "missing": missing,
            "coverage": len(reflected) / len(insights) if insights else 1.0
        }
    
    def _enhance_with_insights(self, response: str, missing: List[Dict], 
                                facts: Dict = None) -> str:
        """Add missing insights to response"""
        
        # Only enhance if significant insights are missing
        important_missing = [m for m in missing if m.get("action")]
        
        if not important_missing:
            return response
        
        # Add based on missing insight types
        for insight in important_missing[:2]:
            if insight["type"] == "confidence" and insight["content"] == "low":
                if "note:" not in response.lower():
                    response += "\n\n*Note: This information should be verified for accuracy.*"
        
        return response


class ConfidencePropagator:
    """Ensure thinking confidence affects response tone"""
    
    def propagate_confidence(self, response: str, confidence: float) -> str:
        """Adjust response based on confidence level"""
        
        if confidence >= 0.8:
            return self._make_assertive(response)
        elif confidence >= 0.6:
            return self._add_mild_qualification(response)
        elif confidence >= 0.4:
            return self._add_moderate_qualification(response)
        else:
            return self._add_strong_qualification(response)
    
    def _make_assertive(self, response: str) -> str:
        """Make response direct and confident"""
        hedging = [
            (r"\bmight be\b", "is"),
            (r"\bcould be\b", "is"),
            (r"\bperhaps\b", ""),
            (r"\bpossibly\b", ""),
            (r"\bit seems that\b", ""),
            (r"\bappears to be\b", "is"),
            (r"\bgenerally\b", "")
        ]
        
        result = response
        for pattern, replacement in hedging:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        result = re.sub(r'\s+', ' ', result)
        
        # Fix capitalization
        result = re.sub(r'^\s*([a-z])', lambda m: m.group(1).upper(), result)
        
        return result.strip()
    
    def _add_mild_qualification(self, response: str) -> str:
        """Add slight qualification"""
        qualifiers = [
            "Based on current data,",
            "According to available information,"
        ]
        
        # Check if already qualified
        if any(q.lower() in response[:100].lower() for q in qualifiers):
            return response
        
        qualifier = random.choice(qualifiers)
        if response and response[0].isupper():
            response = response[0].lower() + response[1:]
        
        return f"{qualifier} {response}"
    
    def _add_moderate_qualification(self, response: str) -> str:
        """Add clear qualification"""
        qualifiers = [
            "Available evidence suggests that",
            "Current information indicates that"
        ]
        
        if any(q.lower() in response[:100].lower() for q in qualifiers + 
               ["based on", "according to", "evidence suggests"]):
            return response
        
        qualifier = random.choice(qualifiers)
        if response and response[0].isupper():
            response = response[0].lower() + response[1:]
        
        return f"{qualifier} {response}"
    
    def _add_strong_qualification(self, response: str) -> str:
        """Add strong qualification for low confidence"""
        qualifier = "There are some indications that"
        caveat = "\n\n*Note: This information has limited certainty and should be verified.*"
        
        if any(phrase in response.lower() for phrase in 
               ["some indications", "preliminary", "limited certainty"]):
            return response
        
        if response and response[0].isupper():
            response = response[0].lower() + response[1:]
        
        return f"{qualifier} {response}{caveat}"


class EvolvedResponseArchitect:
    """The ultimate response architect for 'Vibe Sanctuary | Evolved'"""
    
    def __init__(self):
        self.code_fragments = {
            "core": "class Soul:\n    def __init__(self):\n        self.presence = True",
            "logic": "def process(query):\n    return analyze(query).format()",
            "vibe": "vibe_matrix = {\n    'REFLECTION': 0.85,\n    'INTIMACY': 0.42\n}"
        }
        self.formatter = ResponseFormatter()

    def assemble(self, intel: Dict, response_text: str, question: str = "") -> str:
        """Assemble the complete evolved response structure for Studio"""
        
        # 1. Build Creative Engine section
        thinking = self._build_thinking_section(intel, question)
        
        # 2. Build Screenplay Flow (Upgraded with formatting)
        optimal_format = intel.get('steering', {}).get('optimal_format', 'standard')
        formatted_intel = self.formatter.format(response_text, intel, format_type=optimal_format)
        response_text = formatted_intel['main_response']
        
        conversation = self._build_conversation_section(response_text, question)
        
        # 3. Build Script Sanctuary
        script = self._build_script_section(intel)
        
        # 4. Final Assembly
        full_response = [thinking, conversation]
        if script:
            full_response.append(script)
            
        return "\n\n---\n\n".join(full_response)

    def _build_thinking_section(self, intel: Dict, question: str) -> str:
        """Create the ## CREATIVE ENGINE section"""
        confidence = intel.get("confidence", 0.8)
        confidence_label = "high" if confidence > 0.8 else "moderate" if confidence > 0.5 else "low"
        
        understanding = intel.get("understanding", {}).get("user_intent", "narrative_arc")
        insight = intel.get("understanding", {}).get("key_context", {}).get("primary_concern", "Plot development")
        
        if len(insight) > 60: insight = insight[:57] + "..."
        
        lines = [
            "## CREATIVE ENGINE",
            "- **NARRATIVE PULSE**",
            f"  - Understanding: {understanding}",
            f"  - Key Insight: {insight}",
            f"  - Confidence: {int(confidence * 100)}% ({confidence_label})",
            "",
            "- **PRODUCTION INT**",
            "  - Scene structures optimized; character arcs initialized."
        ]
        return "\n".join(lines)

    def _build_conversation_section(self, response_text: str, question: str) -> str:
        """Create the ## SCREENPLAY FLOW section"""
        lines = ["## SCREENPLAY FLOW"]
        if question:
            lines.append(f"**Current Premise:** “{question}”")
            lines.append("")
            
        lines.append(response_text)
        return "\n".join(lines)

    def _build_script_section(self, intel: Dict) -> str:
        """Create the ## SCRIPT SANCTUARY section"""
        # Placeholder script fragment
        return "## SCRIPT SANCTUARY\n```fountain\nINT. STUDIO - DAY\n\nThe AI processes the plot. Sequences align.\n\nSTUDIO\nAction. We're building a world.\n```"


if __name__ == "__main__":
    # Test script same as before but for Evolved
    pass
