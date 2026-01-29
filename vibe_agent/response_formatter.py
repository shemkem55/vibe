"""
Response Formatter - Structured Output Engine
Implements the multi-format response system from the AI upgrade blueprint.
"""

from typing import Dict, Any, List

class ResponseFormatter:
    """Orchestrates structured output for different contexts"""
    
    def __init__(self):
        self.formats = {
            'standard': self._format_standard,
            'academic': self._format_academic,
            'creative': self._format_creative,
            'technical': self._format_technical,
            'executive': self._format_executive,
        }

    def format(self, content: str, intel_packet: Dict, format_type: str = 'standard') -> Dict[str, Any]:
        """Format content into the specified style"""
        formatter = self.formats.get(format_type, self._format_standard)
        
        # Base elements
        thinking = intel_packet.get('reasoning', {}).get('thought_summary', 'Processing...')
        
        # Format the main body
        formatted_content = formatter(content, intel_packet)
        
        # Add metadata/metrics
        confidence = intel_packet.get('reasoning', {}).get('confidence', 0.5)
        
        return {
            'thinking_process': thinking,
            'main_response': formatted_content,
            'metadata': {
                'format': format_type,
                'confidence': confidence,
                'version': '3.0 (Upgraded)'
            }
        }

    def _format_standard(self, content: str, intel: Dict) -> str:
        """Balanced, clear response"""
        return content

    def _format_academic(self, content: str, intel: Dict) -> str:
        """Formal, structured with definitions and citations (simulated)"""
        vibe = intel.get('meta', {}).get('vibe', 'Neutral')
        return f"# Analysis Report\n\n## Abstract\n{content[:100]}...\n\n## Discussion\n{content}\n\n## Contextual Vibe: {vibe}"

    def _format_creative(self, content: str, intel: Dict) -> str:
        """Rich, atmospheric, use of markdown for emphasis"""
        return f"***\n\n{content}\n\n***"

    def _format_technical(self, content: str, intel: Dict) -> str:
        """Code-centric, precise, bulleted"""
        return f"```\n{content}\n```"

    def _format_executive(self, content: str, intel: Dict) -> str:
        """TL;DR, key takeaways, bottom line first"""
        return f"**Summary:** {content[:50]}...\n\n**Action Items:**\n- Check implementation\n- Verify results"

    def detect_optimal_format(self, content: str, intel: Dict) -> str:
        """Automatically detect the best format based on intent"""
        intent = intel.get('understanding', {}).get('intent', 'conversation')
        complexity = intel.get('understanding', {}).get('complexity', 'low')
        
        if intent == 'technical' or '```' in content:
            return 'technical'
        elif intent == 'creative':
            return 'creative'
        elif complexity == 'high':
            return 'academic'
        elif intent == 'advice':
            return 'executive'
        else:
            return 'standard'
