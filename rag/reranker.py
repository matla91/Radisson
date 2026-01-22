"""
Re-Ranker avec Cross-Encoder

Ce module implémente un re-ranker basé sur un Cross-Encoder pour améliorer
la précision de la récupération après la phase initiale (BM25/Dense).

Pipeline recommandé :
1. BM25 + Dense → top-20 candidats (Retrieval)
2. Cross-Encoder → Re-rank top-20 → top-5 final (Re-ranking)

Amélioration attendue :
- Precision@1 : +10-20% vs. retrieval seul
- Precision@5 : +15-25% vs. retrieval seul

Modèle recommandé : cross-encoder/ms-marco-MiniLM-L-6-v2
- Taille : ~90 MB
- Vitesse : ~100-200 queries/sec sur CPU
- Qualité : State-of-the-art sur MS-MARCO
"""

import logging
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder
import numpy as np

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Reranker:
    """
    Re-ranker basé sur un Cross-Encoder.
    
    Usage :
        reranker = Reranker()
        results = reranker.rerank(query, candidates, top_k=5)
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32
    ):
        """
        Initialise le re-ranker.
        
        Args:
            model_name: Nom du modèle Cross-Encoder
            batch_size: Taille de batch pour traitement par lots
        """
        logger.info(f"🔧 Chargement du Cross-Encoder: {model_name}")
        
        self.model = CrossEncoder(model_name, max_length=512)
        self.batch_size = batch_size
        
        logger.info(f"✅ Re-ranker prêt (batch_size: {batch_size})")
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
        return_scores: bool = False
    ) -> List[Dict]:
        """
        Re-rank les candidats avec le Cross-Encoder.
        
        Args:
            query: Question de l'utilisateur
            candidates: Liste de chunks candidats (de la phase retrieval)
            top_k: Nombre de résultats finaux
            return_scores: Inclut les scores dans les résultats
        
        Returns:
            Liste de chunks re-rankés (top_k meilleurs)
        """
        if not candidates:
            logger.warning("⚠️  Aucun candidat à re-ranker")
            return []
        
        logger.info(f"🔄 Re-ranking de {len(candidates)} candidats...")
        
        # Prépare les paires (query, passage)
        pairs = [
            (query, candidate["text"]) 
            for candidate in candidates
        ]
        
        # Score avec le Cross-Encoder
        scores = self.model.predict(
            pairs, 
            batch_size=self.batch_size,
            show_progress_bar=False
        )
        
        # Trie par score décroissant
        scored_candidates = [
            (candidate, float(score)) 
            for candidate, score in zip(candidates, scores)
        ]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Top-k résultats
        top_candidates = scored_candidates[:top_k]
        
        logger.info(f"✅ Top-{top_k} re-rankés (scores: {[f'{s:.3f}' for _, s in top_candidates[:3]]})")
        
        # Construction du résultat
        results = []
        for candidate, score in top_candidates:
            if return_scores:
                candidate = candidate.copy()
                candidate["rerank_score"] = score
            results.append(candidate)
        
        return results
    
    def rerank_with_stats(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5
    ) -> Tuple[List[Dict], Dict]:
        """
        Re-rank avec statistiques détaillées.
        
        Returns:
            (résultats, statistiques)
        """
        if not candidates:
            return [], {}
        
        # Prépare les paires
        pairs = [(query, c["text"]) for c in candidates]
        
        # Score
        scores = self.model.predict(
            pairs, 
            batch_size=self.batch_size,
            show_progress_bar=False
        )
        
        # Statistiques
        stats = {
            "total_candidates": len(candidates),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "top_k": top_k
        }
        
        # Trie
        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Top-k
        results = []
        for candidate, score in scored_candidates[:top_k]:
            candidate = candidate.copy()
            candidate["rerank_score"] = float(score)
            results.append(candidate)
        
        return results, stats


class HybridRetrieverWithReranking:
    """
    Retriever hybride avec re-ranking intégré.
    
    Pipeline complet :
    1. BM25 + Dense → RRF → top-20
    2. Cross-Encoder → Re-rank → top-5
    """
    
    def __init__(
        self,
        data_dir: str = "../data",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_reranking: bool = True
    ):
        """
        Initialise le retriever avec re-ranking.
        
        Args:
            data_dir: Répertoire des données
            reranker_model: Modèle de re-ranking
            use_reranking: Active/désactive le re-ranking
        """
        from retriever_hybrid import HybridRetriever
        
        logger.info("🚀 Initialisation HybridRetriever + Reranker...")
        
        # Retriever hybride
        self.retriever = HybridRetriever(data_dir=data_dir)
        
        # Re-ranker (optionnel)
        self.use_reranking = use_reranking
        if use_reranking:
            self.reranker = Reranker(model_name=reranker_model)
        else:
            self.reranker = None
        
        logger.info(f"✅ Retriever prêt (re-ranking: {use_reranking})")
    
    def search(
        self,
        query: str,
        k: int = 5,
        k_retrieval: int = None,
        filter_metadata: Dict = None
    ) -> List[Dict]:
        """
        Recherche hybride avec re-ranking.
        
        Args:
            query: Question de l'utilisateur
            k: Nombre final de résultats
            k_retrieval: Nombre de candidats pour re-ranking (défaut: k*4)
            filter_metadata: Filtres optionnels
        
        Returns:
            Liste de chunks (re-rankés si activé)
        """
        # Calcul du nombre de candidats à récupérer
        if k_retrieval is None:
            k_retrieval = max(k * 4, 20)
        
        logger.info(f"📊 Pipeline: Retrieval (top-{k_retrieval}) → Reranking (top-{k})")
        
        # Phase 1 : Retrieval
        candidates = self.retriever.search(
            query=query,
            k=k_retrieval,
            method="hybrid",
            filter_metadata=filter_metadata
        )
        
        logger.info(f"✅ Phase 1 (Retrieval): {len(candidates)} candidats")
        
        # Phase 2 : Re-ranking (si activé)
        if self.use_reranking and self.reranker and len(candidates) > 0:
            results = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=k,
                return_scores=True
            )
            logger.info(f"✅ Phase 2 (Re-ranking): {len(results)} résultats finaux")
        else:
            # Pas de re-ranking → retourne juste les top-k
            results = candidates[:k]
            logger.info(f"⏭️  Phase 2 ignorée (re-ranking désactivé)")
        
        return results
    
    def compare_with_without_reranking(
        self,
        query: str,
        k: int = 5
    ) -> Dict:
        """
        Compare les résultats avec et sans re-ranking.
        
        Utile pour évaluation.
        """
        k_retrieval = max(k * 4, 20)
        
        # Récupère les candidats
        candidates = self.retriever.search(
            query=query,
            k=k_retrieval,
            method="hybrid"
        )
        
        # Sans re-ranking (juste top-k du retrieval)
        without_reranking = candidates[:k]
        
        # Avec re-ranking
        with_reranking = []
        if self.reranker:
            with_reranking = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=k,
                return_scores=True
            )
        
        return {
            "query": query,
            "without_reranking": without_reranking,
            "with_reranking": with_reranking,
            "candidates_count": len(candidates)
        }


# ===== EXEMPLE D'UTILISATION =====

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎯 COMPARAISON: AVEC vs. SANS RE-RANKING")
    print("="*80 + "\n")
    
    # Initialisation
    retriever_with_reranking = HybridRetrieverWithReranking(
        data_dir="../data",
        use_reranking=True
    )
    
    # Exemples de requêtes
    queries = [
        "Quelle est la politique de remboursement des frais de voyage?",
        "Formulaire T4A pour déclaration fiscale",
        "Article 3.2.1 du règlement sur les absences"
    ]
    
    for query in queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)
        
        # Comparaison
        comparison = retriever_with_reranking.compare_with_without_reranking(query, k=3)
        
        print("\n❌ SANS Re-ranking (Retrieval seul):")
        for i, chunk in enumerate(comparison["without_reranking"], 1):
            title = chunk["metadata"]["title"][:60]
            preview = chunk["text"][:80].replace('\n', ' ')
            print(f"  {i}. {title}")
            print(f"     {preview}...")
        
        print("\n✅ AVEC Re-ranking (Cross-Encoder):")
        for i, chunk in enumerate(comparison["with_reranking"], 1):
            title = chunk["metadata"]["title"][:60]
            score = chunk.get("rerank_score", 0)
            preview = chunk["text"][:80].replace('\n', ' ')
            print(f"  {i}. [{score:.3f}] {title}")
            print(f"     {preview}...")
        
        # Analyse de l'ordre
        print("\n📊 Analyse:")
        without_ids = [c["id"] for c in comparison["without_reranking"]]
        with_ids = [c["id"] for c in comparison["with_reranking"]]
        
        changes = sum(1 for i, j in zip(without_ids, with_ids) if i != j)
        print(f"  - Candidats récupérés: {comparison['candidates_count']}")
        print(f"  - Changements d'ordre: {changes}/{len(without_ids)}")
        
        # Nouveaux documents dans top-3 après re-ranking
        new_in_top3 = set(with_ids) - set(without_ids)
        if new_in_top3:
            print(f"  - Nouveaux docs dans top-3: {len(new_in_top3)}")