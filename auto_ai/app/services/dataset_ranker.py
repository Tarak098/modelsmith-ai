import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DatasetRanker:
    @staticmethod
    def score(query: str, dataset: Dict[str, Any], expected_task: str = "classification") -> float:
        name = dataset.get("name", "").lower()
        desc = dataset.get("description", "").lower()
        query_lower = query.lower()
        
        # 1. Semantic Similarity (40%)
        query_words = set(query_lower.split())
        target_text = f"{name} {desc}"
        matches = sum(1 for w in query_words if w in target_text)
        sim_score = matches / max(1, len(query_words))
        
        # 2. Popularity (20%)
        popularity = dataset.get("popularity", 0.5)
        
        # 3. Quality Score (15%)
        quality = dataset.get("quality_score", 0.8)
        
        # 4. Missing values (10%)
        missing_penalty = 1.0
        
        # 5. Size appropriateness (5%)
        size_score = 1.0
        rows = dataset.get("rows")
        if rows:
            if rows < 100:
                size_score = 0.5
            elif rows > 100000:
                size_score = 0.8
                
        # 6. Target availability (5%)
        target_score = 1.0 if dataset.get("target") else 0.5
        
        # 7. Documentation quality (5%)
        doc_score = 1.0 if len(desc) > 100 else 0.5
        
        final_score = (
            0.40 * sim_score +
            0.20 * popularity +
            0.15 * quality +
            0.10 * missing_penalty +
            0.05 * size_score +
            0.05 * target_score +
            0.05 * doc_score
        )
        logger.info(f"Dataset '{dataset['name']}' scored: {final_score:.3f} (sim={sim_score:.2f}, pop={popularity:.2f})")
        return final_score
