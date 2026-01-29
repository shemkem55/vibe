"""
Detailed Research Response Builder
Builds comprehensive, structured, and deep-dive responses from research data.
Focuses on technical depth, structured facts, and proper sourcing.
"""

import re
import datetime

class detailed_response_builder:
    """Build detailed, research-backed responses"""
    
    def __init__(self):
        self.quality_checker = ResponseQualityChecker()
        
    def build_detailed_response(self, research_data, query):
        """Build comprehensive response from research"""
        
        # 1. Organize Data
        organized_data = self._organize_research_data(research_data, query)
        
        # 2. Build Sections
        sections = self._build_sections(organized_data, query)
        
        # 3. Assemble
        response = self._assemble_response(sections, query, research_data)
        
        # 4. Quality Check & Enhance if needed
        if not self.quality_checker.is_sufficiently_detailed(response):
             response += "\n\n*Note: Information may be limited by available search snippets.*"
        
        return response

    def _organize_research_data(self, research_data, query):
        """Extract facts and categorize them"""
        summary_text = research_data.get("summary", "")
        
        data = {
            "summary": summary_text[:500] if len(summary_text) > 500 else summary_text,
            "features": [],
            "performance": [],
            "changes": [],
            "timeline": [],
            "examples": [],
            "sources": research_data.get("citations", [])
        }
        
        # Simple extraction heuristics based on keywords in lines/sentences
        sentences = re.split(r'(?<=[.!?])\s+', summary_text)
        
        for sent in sentences:
            sent_l = sent.lower()
            if any(x in sent_l for x in ["new feature", "introduce", "add", "support for"]):
                data["features"].append(sent)
            elif any(x in sent_l for x in ["faster", "speed", "performance", "optimiz"]):
                data["performance"].append(sent)
            elif any(x in sent_l for x in ["deprecate", "remove", "change", "breaking"]):
                data["changes"].append(sent)
            elif any(x in sent_l for x in ["release", "date", "timeline", "schedule"]):
                data["timeline"].append(sent)
            elif any(x in sent_l for x in ["example", "code", "syntax"]):
                data["examples"].append(sent)
        
        # If no categorized features but long summary, treat middle details as features
        if not data["features"] and len(sentences) > 5:
            data["features"] = sentences[1:4]
            
        return data

    def _build_sections(self, data, query):
        sections = {}
        
        # Features Section
        if data["features"]:
            lines = ["## 🚀 **Key Features**", ""]
            for i, feat in enumerate(data["features"][:8], 1):
                clean_feat = feat.strip()
                if len(clean_feat) > 10:
                    lines.append(f"### {i}. **{self._extract_title(clean_feat)}**")
                    lines.append(f"{clean_feat}")
                    lines.append("")
            sections["features"] = "\n".join(lines)
            
        # Performance Section
        if data["performance"]:
            lines = ["## ⚡ **Performance & Optimization**", ""]
            for perf in data["performance"][:5]:
                lines.append(f"• {perf.strip()}")
            sections["performance"] = "\n".join(lines)
            
        # Changes/Deprecations
        if data["changes"]:
            lines = ["## ⚠️ **Changes & Deprecations**", ""]
            for change in data["changes"][:5]:
                lines.append(f"• {change.strip()}")
            sections["changes"] = "\n".join(lines)
            
        return sections
        
    def _extract_title(self, text):
        """Extract a short title from a sentence"""
        words = text.split()
        if len(words) < 5:
            return text
        return " ".join(words[:5]) + "..."
    
    def _assemble_response(self, sections, query, data):
        response = []
        
        # Header
        response.append(f"# 🔍 **Research: {query}**")
        response.append(f"*Compiled from {len(data['sources'])} sources*")
        response.append("")
        
        # Executive Summary
        response.append("## 📋 **Executive Summary**")
        response.append(data["summary"])
        response.append("")
        
        # Sections
        for key in ["features", "performance", "changes"]:
            if key in sections:
                response.append(sections[key])
                response.append("")
        
        # Sources
        response.append("---")
        response.append(f"**Sources**:")
        seen_urls = set()
        for i, src in enumerate(data["sources"][:5], 1):
            url = src.get("url", "")
            if url and url not in seen_urls:
                response.append(f"{i}. [{src.get('title', 'Source')}]({url}) - *{src.get('source')}*")
                seen_urls.add(url)
                
        return "\n".join(response)

class ResponseQualityChecker:
    def is_sufficiently_detailed(self, response):
        return len(response.split()) > 100
