"""
Knowledge Graph - Phase 4.1 Intelligence Upgrade
Connect information for deeper understanding and cross-domain insights
"""

import sqlite3
import json
import re
from collections import defaultdict
from datetime import datetime


class KnowledgeGraph:
    """Connect concepts for deeper understanding"""
    
    def __init__(self, db_path='agent_memory.db'):
        self.db_path = db_path
        self._init_graph_tables()
        
        # In-memory graph representation
        self.concepts = {}  # concept_id -> {name, properties, domain}
        self.edges = defaultdict(list)  # concept_id -> [(target_id, relationship, weight)]
        
        # Pre-defined domain knowledge for cross-domain connections
        self._init_domain_knowledge()
        
        # Load existing graph
        self._load_graph()
    
    def _init_graph_tables(self):
        """Initialize graph-specific tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Concepts (nodes)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS kg_concepts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    domain TEXT,
                    properties_json TEXT,
                    mention_count INTEGER DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Relationships (edges)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS kg_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER,
                    target_id INTEGER,
                    relationship_type TEXT,
                    weight REAL DEFAULT 0.5,
                    evidence_count INTEGER DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES kg_concepts(id),
                    FOREIGN KEY (target_id) REFERENCES kg_concepts(id)
                )
            ''')
            
            # Inferences (derived knowledge)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS kg_inferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    statement TEXT,
                    derived_from TEXT,
                    confidence REAL,
                    validated INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def _init_domain_knowledge(self):
        """Initialize cross-domain knowledge structures"""
        # Analogical mappings between domains
        self.domain_analogies = {
            ("biology", "technology"): {
                "brain": "computer",
                "neuron": "processor",
                "dna": "code",
                "evolution": "optimization",
                "immune_system": "firewall",
                "cell": "module",
                "virus": "malware",
                "ecosystem": "network"
            },
            ("physics", "philosophy"): {
                "entropy": "chaos",
                "equilibrium": "balance",
                "force": "influence",
                "energy": "potential",
                "momentum": "motivation",
                "gravity": "attraction",
                "light": "truth",
                "wave": "change"
            },
            ("nature", "emotion"): {
                "storm": "anger",
                "calm": "peace",
                "sunrise": "hope",
                "night": "sadness",
                "spring": "joy",
                "winter": "melancholy",
                "ocean": "depth",
                "river": "flow"
            },
            ("art", "science"): {
                "composition": "structure",
                "harmony": "balance",
                "color": "spectrum",
                "rhythm": "pattern",
                "canvas": "medium",
                "perspective": "framework",
                "style": "methodology"
            },
            ("business", "biology"): {
                "growth": "growth",
                "competition": "survival",
                "market": "ecosystem",
                "startup": "organism",
                "merger": "symbiosis",
                "bankruptcy": "extinction",
                "innovation": "mutation"
            }
        }
        
        # 🆕 Deep Metaphor Bridges (Expansion)
        self.metaphorical_bridges = {
            "journey": ["life", "career", "project", "learning", "discovery"],
            "building": ["knowledge", "society", "character", "argument", "relationship"],
            "nature": ["economy", "emotions", "growth", "networks"],
            "war": ["debate", "competition", "strategy", "disease"],
            "machine": ["society", "mind", "organization", "universe"]
        }
        
        # Common relationship types
        self.relationship_types = {
            "is_a": 0.9,  # inheritance
            "part_of": 0.8,  # composition
            "related_to": 0.5,  # general relation
            "causes": 0.7,  # causality
            "similar_to": 0.6,  # similarity
            "opposite_of": 0.4,  # antonymy
            "used_for": 0.6,  # purpose
            "found_in": 0.5,  # location/context
            "analogous_to": 0.7  # cross-domain
        }
        
        # Domain keywords for classification
        self.domain_keywords = {
            "science": ["quantum", "physics", "chemistry", "biology", "atom", "molecule", "experiment"],
            "technology": ["computer", "software", "algorithm", "ai", "code", "programming", "digital"],
            "philosophy": ["consciousness", "existence", "meaning", "ethics", "truth", "reality", "mind"],
            "nature": ["ocean", "forest", "mountain", "star", "earth", "weather", "animal", "plant"],
            "emotion": ["love", "fear", "joy", "sadness", "anger", "peace", "happiness", "anxiety"],
            "art": ["music", "painting", "poetry", "creative", "beauty", "expression", "design"],
            "business": ["market", "company", "startup", "growth", "strategy", "investment"]
        }
    
    def _load_graph(self):
        """Load existing graph from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Load concepts
            cursor.execute('SELECT * FROM kg_concepts')
            for row in cursor.fetchall():
                self.concepts[row['id']] = {
                    "name": row['name'],
                    "domain": row['domain'],
                    "properties": json.loads(row['properties_json'] or '{}')
                }
            
            # Load relationships
            cursor.execute('SELECT * FROM kg_relationships')
            for row in cursor.fetchall():
                self.edges[row['source_id']].append(
                    (row['target_id'], row['relationship_type'], row['weight'])
                )
    
    def add_concept(self, name, domain=None, properties=None):
        """Add a concept to the knowledge graph"""
        name = name.lower().strip()
        domain = domain or self._detect_domain(name)
        properties = properties or {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if exists
            cursor.execute('SELECT id, mention_count FROM kg_concepts WHERE name = ?', (name,))
            existing = cursor.fetchone()
            
            if existing:
                # Increment mention count
                conn.execute(
                    'UPDATE kg_concepts SET mention_count = mention_count + 1 WHERE id = ?',
                    (existing[0],)
                )
                return existing[0]
            else:
                # Insert new
                cursor.execute('''
                    INSERT INTO kg_concepts (name, domain, properties_json)
                    VALUES (?, ?, ?)
                ''', (name, domain, json.dumps(properties)))
                concept_id = cursor.lastrowid
                
                # Update in-memory graph
                self.concepts[concept_id] = {
                    "name": name,
                    "domain": domain,
                    "properties": properties
                }
                
                return concept_id
    
    def add_relationship(self, source_name, target_name, relationship_type="related_to"):
        """Add a relationship between two concepts"""
        source_id = self.add_concept(source_name)
        target_id = self.add_concept(target_name)
        weight = self.relationship_types.get(relationship_type, 0.5)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if relationship exists
            cursor.execute('''
                SELECT id, evidence_count FROM kg_relationships 
                WHERE source_id = ? AND target_id = ? AND relationship_type = ?
            ''', (source_id, target_id, relationship_type))
            existing = cursor.fetchone()
            
            if existing:
                # Strengthen relationship
                new_weight = min(1.0, weight + 0.1)
                conn.execute('''
                    UPDATE kg_relationships 
                    SET weight = ?, evidence_count = evidence_count + 1 
                    WHERE id = ?
                ''', (new_weight, existing[0]))
            else:
                # Create new relationship
                conn.execute('''
                    INSERT INTO kg_relationships (source_id, target_id, relationship_type, weight)
                    VALUES (?, ?, ?, ?)
                ''', (source_id, target_id, relationship_type, weight))
            
            # Update in-memory graph
            self.edges[source_id].append((target_id, relationship_type, weight))
        
        return (source_id, target_id)
    
    def _detect_domain(self, text):
        """Detect which domain a concept belongs to"""
        text_lower = text.lower()
        
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return domain
        
        return "general"
    
    def extract_concepts_and_relations(self, text):
        """Extract concepts and relationships from text"""
        text_lower = text.lower()
        concepts = []
        relationships = []
        
        # Extract nouns/concepts (simple approach)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)
        
        # Common meaningful words to track
        meaningful = set()
        for domain, keywords in self.domain_keywords.items():
            meaningful.update(keywords)
        
        for word in words:
            if word in meaningful:
                concepts.append(word)
        
        # Detect relationships from patterns
        relationship_patterns = {
            r"(\w+) is (?:a|an) (\w+)": "is_a",
            r"(\w+) (?:is part of|belongs to) (\w+)": "part_of",
            r"(\w+) causes (\w+)": "causes",
            r"(\w+) and (\w+)": "related_to",
            r"(\w+) like (\w+)": "similar_to",
            r"(\w+) versus (\w+)": "opposite_of",
            r"(\w+) for (\w+)": "used_for"
        }
        
        for pattern, rel_type in relationship_patterns.items():
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if len(match) == 2:
                    relationships.append((match[0], match[1], rel_type))
        
        return concepts, relationships
    
    def learn_from_text(self, text):
        """Extract knowledge from text and update graph"""
        concepts, relationships = self.extract_concepts_and_relations(text)
        
        # Add concepts
        for concept in concepts:
            self.add_concept(concept)
        
        # Add relationships
        for source, target, rel_type in relationships:
            self.add_relationship(source, target, rel_type)
        
        # Try to infer new connections
        self._infer_connections()
        
        return {"concepts_added": len(concepts), "relationships_added": len(relationships)}
    
    def _infer_connections(self):
        """Make intelligent inferences from existing knowledge"""
        # Transitive inference: If A is_a B, and B is_a C, then A is_a C
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Find transitive is_a chains
            cursor.execute('''
                SELECT r1.source_id, r2.target_id, r1.weight * r2.weight
                FROM kg_relationships r1
                JOIN kg_relationships r2 ON r1.target_id = r2.source_id
                WHERE r1.relationship_type = 'is_a' AND r2.relationship_type = 'is_a'
                AND r1.source_id != r2.target_id
            ''')
            
            for source_id, target_id, combined_weight in cursor.fetchall():
                if combined_weight > 0.4:
                    # Check if this inference already exists
                    cursor.execute('''
                        SELECT id FROM kg_relationships 
                        WHERE source_id = ? AND target_id = ? AND relationship_type = 'is_a'
                    ''', (source_id, target_id))
                    
                    if not cursor.fetchone():
                        # Add inferred relationship
                        conn.execute('''
                            INSERT INTO kg_relationships 
                            (source_id, target_id, relationship_type, weight)
                            VALUES (?, ?, ?, ?)
                        ''', (source_id, target_id, 'is_a', combined_weight * 0.8))
                        
                        # Log inference
                        source_name = self.concepts.get(source_id, {}).get("name", "?")
                        target_name = self.concepts.get(target_id, {}).get("name", "?")
                        conn.execute('''
                            INSERT INTO kg_inferences (statement, derived_from, confidence)
                            VALUES (?, ?, ?)
                        ''', (
                            f"{source_name} is_a {target_name}",
                            "transitive_inference",
                            combined_weight * 0.8
                        ))
    
    def find_analogies(self, concept, target_domain=None):
        """Find analogous concepts in different domains"""
        concept = concept.lower()
        concept_domain = self._detect_domain(concept)
        
        analogies = []
        
        # Search in domain analogy maps
        for (domain1, domain2), mapping in self.domain_analogies.items():
            if concept_domain == domain1:
                if concept in mapping:
                    if target_domain is None or target_domain == domain2:
                        analogies.append({
                            "source": concept,
                            "source_domain": domain1,
                            "analogous": mapping[concept],
                            "target_domain": domain2,
                            "confidence": 0.8
                        })
            elif concept_domain == domain2:
                # Reverse lookup
                reverse_mapping = {v: k for k, v in mapping.items()}
                if concept in reverse_mapping:
                    if target_domain is None or target_domain == domain1:
                        analogies.append({
                            "source": concept,
                            "source_domain": domain2,
                            "analogous": reverse_mapping[concept],
                            "target_domain": domain1,
                            "confidence": 0.8
                        })
        
        return analogies
    
    def find_deep_metaphor(self, concept):
        """Discover deep metaphorical parallels for a concept"""
        concept = concept.lower()
        metaphors = []
        
        # 1. Direct analogies
        analogies = self.find_analogies(concept)
        for a in analogies:
            metaphors.append({
                "type": "direct_analogy",
                "target": a["analogous"],
                "reason": f"Shared functional mapping between {a['source_domain']} and {a['target_domain']}"
            })
            
        # 2. Pattern-based metaphorical bridges
        for bridge, members in self.metaphorical_bridges.items():
            if concept in members:
                # This concept is part of a common metaphor
                metaphors.append({
                    "type": "conceptual_bridge",
                    "bridge": bridge,
                    "equivalents": [m for m in members if m != concept],
                    "reason": f"Both can be viewed through the lens of a '{bridge}'"
                })
        
        # 3. Graph-based structural parallels (Simulated)
        if len(metaphors) < 2:
            # Look for nodes with similar edge structures
            pass
            
        return metaphors
    
    def query_related(self, concept, depth=2):
        """Find all concepts related to the given concept"""
        concept = concept.lower()
        
        # Find concept ID
        concept_id = None
        for cid, data in self.concepts.items():
            if data["name"] == concept:
                concept_id = cid
                break
        
        if concept_id is None:
            return {"concept": concept, "related": [], "message": "Concept not found in graph"}
        
        # BFS to find related concepts
        visited = {concept_id}
        current_level = [concept_id]
        related = []
        
        for level in range(depth):
            next_level = []
            for cid in current_level:
                for target_id, rel_type, weight in self.edges.get(cid, []):
                    if target_id not in visited:
                        visited.add(target_id)
                        next_level.append(target_id)
                        target_data = self.concepts.get(target_id, {})
                        related.append({
                            "concept": target_data.get("name", "unknown"),
                            "domain": target_data.get("domain", "general"),
                            "relationship": rel_type,
                            "distance": level + 1,
                            "weight": weight
                        })
            current_level = next_level
        
        # Sort by relevance (weight and distance)
        related.sort(key=lambda x: (-x["weight"], x["distance"]))
        
        return {
            "concept": concept,
            "domain": self.concepts.get(concept_id, {}).get("domain", "general"),
            "related": related[:10]  # Top 10
        }
    
    def explain_connection(self, concept1, concept2):
        """Find and explain the connection between two concepts"""
        c1 = concept1.lower()
        c2 = concept2.lower()
        
        # Find concept IDs
        id1 = id2 = None
        for cid, data in self.concepts.items():
            if data["name"] == c1:
                id1 = cid
            if data["name"] == c2:
                id2 = cid
        
        if id1 is None or id2 is None:
            return {
                "connected": False,
                "message": "One or both concepts not found in knowledge graph"
            }
        
        # BFS to find path
        visited = {id1}
        queue = [(id1, [])]
        
        while queue:
            current, path = queue.pop(0)
            
            for target_id, rel_type, weight in self.edges.get(current, []):
                new_path = path + [(current, target_id, rel_type)]
                
                if target_id == id2:
                    # Found path
                    explanation = self._format_path_explanation(new_path)
                    return {
                        "connected": True,
                        "path_length": len(new_path),
                        "explanation": explanation,
                        "path": new_path
                    }
                
                if target_id not in visited:
                    visited.add(target_id)
                    queue.append((target_id, new_path))
        
        # Check for cross-domain analogy
        analogies = self.find_analogies(c1)
        for analogy in analogies:
            if analogy["analogous"] == c2:
                return {
                    "connected": True,
                    "connection_type": "analogy",
                    "explanation": f"{c1} in {analogy['source_domain']} is analogous to {c2} in {analogy['target_domain']}"
                }
        
        return {
            "connected": False,
            "message": f"No direct connection found between {c1} and {c2}"
        }
    
    def _format_path_explanation(self, path):
        """Format a path into human-readable explanation"""
        if not path:
            return "No path found"
        
        parts = []
        for source_id, target_id, rel_type in path:
            source_name = self.concepts.get(source_id, {}).get("name", "?")
            target_name = self.concepts.get(target_id, {}).get("name", "?")
            rel_readable = rel_type.replace("_", " ")
            parts.append(f"{source_name} {rel_readable} {target_name}")
        
        return " → ".join(parts)
    
    def get_graph_stats(self):
        """Get statistics about the knowledge graph"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM kg_concepts')
            concept_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM kg_relationships')
            relationship_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM kg_inferences')
            inference_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT domain, COUNT(*) FROM kg_concepts GROUP BY domain')
            domains = dict(cursor.fetchall())
        
        return {
            "total_concepts": concept_count,
            "total_relationships": relationship_count,
            "total_inferences": inference_count,
            "domains": domains
        }


if __name__ == "__main__":
    # Test knowledge graph
    kg = KnowledgeGraph()
    
    print("=== Knowledge Graph Test ===\n")
    
    # Add some concepts and relationships
    kg.add_relationship("human", "mammal", "is_a")
    kg.add_relationship("mammal", "animal", "is_a")
    kg.add_relationship("brain", "human", "part_of")
    kg.add_relationship("neuron", "brain", "part_of")
    kg.add_relationship("computer", "technology", "is_a")
    kg.add_relationship("algorithm", "computer", "used_for")
    
    # Learn from text
    print("Learning from text...")
    result = kg.learn_from_text("The brain is like a computer, with neurons similar to processors.")
    print(f"  Extracted: {result}")
    
    # Test analogies
    print("\n--- Finding Analogies ---")
    for concept in ["brain", "virus", "evolution"]:
        analogies = kg.find_analogies(concept)
        if analogies:
            for a in analogies:
                print(f"  {a['source']} ({a['source_domain']}) ≈ {a['analogous']} ({a['target_domain']})")
    
    # Test related concepts
    print("\n--- Related Concepts ---")
    related = kg.query_related("human", depth=3)
    print(f"Concepts related to '{related['concept']}':")
    for r in related["related"]:
        print(f"  - {r['concept']} ({r['relationship']}, distance: {r['distance']})")
    
    # Test connection explanation
    print("\n--- Connection Explanation ---")
    connection = kg.explain_connection("neuron", "animal")
    if connection["connected"]:
        print(f"  {connection['explanation']}")
    else:
        print(f"  {connection['message']}")
    
    # Stats
    print("\n--- Graph Stats ---")
    stats = kg.get_graph_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
