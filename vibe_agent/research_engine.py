import random
import requests
import feedparser
import arxiv
import wikipedia
import re
from bs4 import BeautifulSoup, Comment
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from cachetools import LRUCache
import urllib.parse
from typing import Dict, List, Any

class ResearchTrigger:
    """Determine when to research vs. converse"""
    
    RESEARCH_KEYWORDS = {
        "factual": ["what is", "who invented", "when did", "how many", "population of"],
        "technical": ["explain", "define", "how does", "why does", "concept of"],
        "current": ["latest", "recent", "new", "update", "today", "news"],
        "statistical": ["percentage", "statistics", "data", "study shows"]
    }
    
    def needs_research(self, query):
        """Should we research this?"""
        query_lower = query.lower()
        
        # Quick checks
        if "?" not in query and not any(kw in query_lower for kw in ["tell me about", "research", "find out"]):
            return False 
        
        # Check for research keywords
        for category, keywords in self.RESEARCH_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return {"type": category, "depth": "quick"}
        
        # Check if user explicitly asks for info
        if any(phrase in query_lower for phrase in ["find out", "look up", "research", "information about"]):
            return {"type": "explicit", "depth": "deep"}
        
        return False

class ResearchEngine:
    """Advanced Research Engine with Deep Web Reading Capabilities"""
    
    def __init__(self):
        self.cache = LRUCache(maxsize=100)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        })
    
    def research(self, query: str, depth: str = "quick") -> Dict[str, Any]:
        """Main research method with multi-stage processing"""
        cache_key = f"{query}_{depth}_v2"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        print(f"🕵️  Researching: {query} (Depth: {depth})")
        
        # 1. Source Selection
        sources_to_query = self._select_sources(query, depth)
        
        # 2. Initial Gathering
        initial_results = self._gather_initial_results(sources_to_query, query)
        
        # 3. Deep Reading (Fetch content from URLs found)
        deep_content = self._deep_read_urls(initial_results, max_pages=2 if depth == "deep" else 1)
        
        # 4. Integrate Deep Content back into results
        combined_results = self._integrate_results(initial_results, deep_content)
        
        # 5. Synthesize
        synthesized = self._synthesize_results(combined_results, query)
        self.cache[cache_key] = synthesized
        
        return synthesized
    
    def _gather_initial_results(self, sources, query):
        """Gather initial search snippets in parallel"""
        results = {}
        with ThreadPoolExecutor(max_workers=len(sources)) as executor:
            future_to_source = {
                executor.submit(self._get_source_func(source), query): source 
                for source in sources
            }
            
            for future in concurrent.futures.as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    results[source] = future.result()
                except Exception as e:
                    print(f"Source error ({source}): {e}")
                    results[source] = {"error": str(e), "content": ""}
        return results
    
    def _deep_read_urls(self, initial_results: Dict, max_pages: int = 1) -> List[Dict]:
        """Visit URLs found in search results to get full content"""
        urls_to_visit = []
        
        # Extract URLs from snippets
        for source, data in initial_results.items():
            if data.get("url") and "wikipedia.org" not in data.get("url", ""): # Wikipedia already gets content
                 urls_to_visit.append(data["url"])
        
        urls_to_visit = list(set(urls_to_visit))[:max_pages]
        deep_data = []
        
        if not urls_to_visit:
            return deep_data
            
        print(f"📖 Deep reading: {urls_to_visit}")
        
        with ThreadPoolExecutor(max_workers=max_pages) as executor:
            future_to_url = {executor.submit(self._fetch_page_content, url): url for url in urls_to_visit}
            
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    content = future.result()
                    if content:
                        deep_data.append(content)
                except Exception as e:
                    print(f"Deep read failed for {url}: {e}")
        
        return deep_data
    
    def _fetch_page_content(self, url: str) -> Dict[str, str]:
        """Fetch and extract meaningful text from a web page"""
        try:
            response = self.session.get(url, timeout=5)
            if response.status_code != 200:
                return None
                
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Remove junk
            for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'iframe', 'ads', 'noscript']):
                element.decompose()
                
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # Extract paragraphs that look like content
            paragraphs = []
            for p in soup.find_all('p'):
                text = p.get_text().strip()
                if len(text) > 50:  # Skip short snippets
                    paragraphs.append(text)
            
            # Combine reasonable amount of text
            full_text = " ".join(paragraphs[:10]) # First 10 paragraphs usually contain the meat
            title = soup.title.string if soup.title else url
            
            if len(full_text) < 100:
                return None
                
            return {
                "content": full_text,
                "url": url,
                "title": title.strip(),
                "source": "web_deep_read"
            }
        except Exception:
            return None

    def _integrate_results(self, initial: Dict, deep: List[Dict]) -> Dict:
        """Combine specific source results with deep read general web results"""
        for i, item in enumerate(deep):
            initial[f"deep_read_{i}"] = item
        return initial

    def _get_source_func(self, source):
        mapping = {
            "wikipedia": self._query_wikipedia,
            "duckduckgo": self._search_duckduckgo,
            "arxiv": self._search_arxiv,
            "news": self._search_news_rss
        }
        # Fallback to DDG if unknown
        return mapping.get(source, self._search_duckduckgo)

    def _select_sources(self, query, depth):
        sources = ["duckduckgo"]
        query_l = query.lower()
        if any(word in query_l for word in ["theory", "paper", "study", "research", "science", "quantum", "physics"]):
            sources.append("arxiv")
        if any(word in query_l for word in ["news", "recent", "today", "latest", "update"]):
            sources.append("news")
        if depth == "deep" or len(query.split()) < 5 or "history" in query_l or "who is" in query_l:
            sources.append("wikipedia")
        return sources

    def _query_wikipedia(self, query):
        try:
            # Clean query for wikipedia explicitly
            clean_q = query.replace("what is", "").replace("who is", "").strip()
            search_results = wikipedia.search(clean_q, results=1)
            
            if not search_results: 
                return {"content": "", "url": "", "source": "wikipedia"}
                
            try:
                page = wikipedia.page(search_results[0], auto_suggest=False)
            except wikipedia.DisambiguationError as e:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                
            summary = page.summary
            # Limit to reasonable length
            if len(summary) > 1000:
                summary = summary[:1000] + "..."
                
            return {
                "content": summary, 
                "url": page.url, 
                "title": page.title, 
                "source": "wikipedia"
            }
        except Exception as e:
            # Fallback to search if page load fails
            try:
                return {
                    "content": wikipedia.summary(query, sentences=2),
                    "url": "https://wikipedia.org",
                    "title": "Wikipedia Result",
                    "source": "wikipedia"
                }
            except:
                return {"content": "", "url": "", "source": "wikipedia"}

    def _search_duckduckgo(self, query):
        try:
            # Use html version which is easier to scrape without API
            url = f"https://html.duckduckgo.com/html/"
            data = {'q': query}
            response = self.session.post(url, data=data, timeout=5)
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Extract multiple results
            results = []
            for result in soup.find_all('div', class_='result'):
                link = result.find('a', class_='result__a')
                snippet = result.find('a', class_='result__snippet')
                if link and snippet:
                    results.append(f"{link.get_text()}: {snippet.get_text()}")
                    # Just capture the first valid URL for deep reading later
                    first_url = link['href']
                    break # Just take the top result structure for now
            
            content = " ".join(results)
            # Find the actual URL if possible
            first_url_val = ""
            if soup.find('a', class_='result__a'):
                first_url_val = soup.find('a', class_='result__a')['href']
            
            return {
                "content": content[:800], 
                "url": first_url_val, 
                "title": "Search Result",
                "source": "duckduckgo"
            }
        except Exception as e:
            return {"content": "", "source": "duckduckgo", "error": str(e)}

    def _search_arxiv(self, query):
        try:
            # Clean query for ArXiv
            clean_q = re.sub(r'(what is|explain|theory of)', '', query).strip()
            search = arxiv.Search(query=clean_q, max_results=2, sort_by=arxiv.SortCriterion.Relevance)
            
            content_list = []
            primary_url = ""
            primary_title = ""
            
            for res in search.results():
                content_list.append(f"Paper '{res.title}': {res.summary[:300]}...")
                if not primary_url:
                    primary_url = res.entry_id
                    primary_title = res.title
            
            return {
                "content": "\n".join(content_list), 
                "url": primary_url, 
                "title": primary_title, 
                "source": "arxiv"
            }
        except:
            return {"content": "", "source": "arxiv"}

    def _search_news_rss(self, query):
        try:
            # Google news RSS
            rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            
            if not feed.entries: return {"content": "", "source": "news"}
            
            news_items = []
            primary_url = ""
            primary_title = ""
            
            for entry in feed.entries[:3]:
                news_items.append(f"{entry.title} ({entry.published})")
                if not primary_url:
                    primary_url = entry.link
                    primary_title = entry.title
            
            return {
                "content": "Top Headlines: " + "; ".join(news_items), 
                "url": primary_url, 
                "title": primary_title, 
                "source": "news"
            }
        except:
            return {"content": "", "source": "news"}

    def _synthesize_results(self, results: Dict, query: str) -> Dict:
        """Combine all results into a coherent summary"""
        all_text_segments = []
        citations = []
        
        # Priority ordering
        priority_order = ["wikipedia", "arxiv", "web_deep_read", "duckduckgo", "news"]
        
        # Sort results by priority
        sorted_keys = sorted(results.keys(), key=lambda k: self._get_priority(k, results, priority_order))
        
        for src in sorted_keys:
            data = results[src]
            if data.get("content"):
                # Clean content
                content = data["content"].strip()
                if content:
                    all_text_segments.append(content)
                
                # Add citation
                if data.get("url"):
                    citations.append({
                        "source": data.get("source", src),
                        "url": data["url"],
                        "title": data.get("title", "Reference")
                    })
        
        # Construct summary
        if not all_text_segments:
            return {"summary": "", "citations": [], "confidence": 0.0}
            
        full_text = "\n\n".join(all_text_segments)
        
        # Basic extractive summarization logic (take first few sentences of highest priority, then fill details)
        # DeepSeek builder will handle the formatting, we just need rich raw text.
        
        # Confidence metric
        confidence = 0.5
        if "wikipedia" in results: confidence += 0.3
        if "deep_read" in str(results.keys()): confidence += 0.2
        confidence = min(confidence, 0.95)
        
        return {
            "summary": full_text,  # Return rich text, let the builder summarize/format
            "citations": citations[:5],
            "confidence": confidence
        }

    def _get_priority(self, key, results, order):
        source_type = results[key].get("source", "other")
        if source_type in order:
            return order.index(source_type)
        return 99
