"""
Tests et Benchmarks pour le système RAG UQAC

Ce script permet de :
1. Tester chaque composant individuellement
2. Comparer les performances des différentes méthodes
3. Mesurer les latences et la qualité
4. Générer des rapports détaillés

Usage :
    python test_benchmark.py --test all
    python test_benchmark.py --test retrieval
    python test_benchmark.py --test reranking
    python test_benchmark.py --benchmark
"""

import argparse
import time
import json
from typing import List, Dict
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGBenchmark:
    """
    Suite de tests et benchmarks pour le système RAG.
    """
    
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = data_dir
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "benchmarks": {}
        }
    
    def test_retriever_loading(self):
        """Test du chargement du retriever."""
        logger.info("\n" + "="*80)
        logger.info("TEST 1: Chargement du Retriever")
        logger.info("="*80)
        
        try:
            from retriever_hybrid import HybridRetriever
            
            start = time.time()
            retriever = HybridRetriever(data_dir=self.data_dir)
            load_time = time.time() - start
            
            logger.info(f"✅ Retriever chargé en {load_time:.3f}s")
            
            self.results["tests"]["retriever_loading"] = {
                "status": "PASS",
                "load_time_seconds": load_time
            }
            return True
        
        except Exception as e:
            logger.error(f"❌ Échec: {e}")
            self.results["tests"]["retriever_loading"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return False
    
    def test_search_methods(self):
        """Test des 3 méthodes de recherche."""
        logger.info("\n" + "="*80)
        logger.info("TEST 2: Méthodes de Recherche (BM25, Dense, Hybrid)")
        logger.info("="*80)
        
        try:
            from retriever_hybrid import HybridRetriever
            
            retriever = HybridRetriever(data_dir=self.data_dir)
            test_query = "Quelle est la politique de remboursement des frais de voyage?"
            
            methods = ["bm25", "dense", "hybrid"]
            results = {}
            
            for method in methods:
                logger.info(f"\n🔍 Test méthode: {method}")
                
                start = time.time()
                docs = retriever.search(test_query, k=5, method=method)
                latency = time.time() - start
                
                logger.info(f"  ✅ {len(docs)} documents en {latency:.3f}s")
                
                # Vérifie que les résultats sont non vides
                assert len(docs) > 0, f"Aucun résultat pour {method}"
                assert all("text" in d for d in docs), "Chunks malformés"
                assert all("metadata" in d for d in docs), "Métadonnées manquantes"
                
                results[method] = {
                    "status": "PASS",
                    "num_results": len(docs),
                    "latency_seconds": latency
                }
            
            self.results["tests"]["search_methods"] = results
            logger.info("\n✅ Toutes les méthodes fonctionnent")
            return True
        
        except Exception as e:
            logger.error(f"❌ Échec: {e}")
            self.results["tests"]["search_methods"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return False
    
    def test_reranking(self):
        """Test du re-ranking."""
        logger.info("\n" + "="*80)
        logger.info("TEST 3: Re-Ranking avec Cross-Encoder")
        logger.info("="*80)
        
        try:
            from reranker import HybridRetrieverWithReranking
            
            retriever = HybridRetrieverWithReranking(
                data_dir=self.data_dir,
                use_reranking=True
            )
            
            test_query = "Formulaire de demande de remboursement"
            
            start = time.time()
            results = retriever.search(test_query, k=5)
            latency = time.time() - start
            
            logger.info(f"✅ {len(results)} résultats re-rankés en {latency:.3f}s")
            
            # Vérifie que les scores de re-ranking sont présents
            has_scores = all("rerank_score" in r for r in results)
            logger.info(f"  Scores présents: {has_scores}")
            
            if has_scores:
                scores = [r["rerank_score"] for r in results]
                logger.info(f"  Scores: {[f'{s:.3f}' for s in scores]}")
            
            self.results["tests"]["reranking"] = {
                "status": "PASS",
                "num_results": len(results),
                "latency_seconds": latency,
                "has_rerank_scores": has_scores
            }
            return True
        
        except Exception as e:
            logger.error(f"❌ Échec: {e}")
            self.results["tests"]["reranking"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return False
    
    def test_rag_engine(self):
        """Test du RAG Engine complet."""
        logger.info("\n" + "="*80)
        logger.info("TEST 4: RAG Engine v2 (avec historique)")
        logger.info("="*80)
        
        try:
            from rag_engine_v2 import RAGEngineV2
            
            engine = RAGEngineV2(
                data_dir=self.data_dir,
                use_ollama=False  # Désactivé pour tests unitaires
            )
            
            # Test 1 : Question simple
            logger.info("\n📝 Test question simple...")
            answer1, docs1 = engine.ask("Quelle est la politique de voyage?")
            assert len(docs1) > 0, "Aucun document récupéré"
            logger.info(f"  ✅ {len(docs1)} documents récupérés")
            
            # Test 2 : Question contextuelle
            logger.info("\n📝 Test question contextuelle...")
            answer2, docs2 = engine.ask("Et pour les voyages internationaux?")
            assert len(docs2) > 0, "Aucun document récupéré"
            logger.info(f"  ✅ {len(docs2)} documents récupérés")
            
            # Test 3 : Historique
            history = engine.get_history()
            assert len(history) == 2, "Historique incorrect"
            logger.info(f"  ✅ Historique: {len(history)} échanges")
            
            self.results["tests"]["rag_engine"] = {
                "status": "PASS",
                "test_cases": 3,
                "history_length": len(history)
            }
            return True
        
        except Exception as e:
            logger.error(f"❌ Échec: {e}")
            self.results["tests"]["rag_engine"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return False
    
    def benchmark_latency(self, num_queries: int = 20):
        """Benchmark de latence sur plusieurs requêtes."""
        logger.info("\n" + "="*80)
        logger.info(f"BENCHMARK: Latence ({num_queries} requêtes)")
        logger.info("="*80)
        
        try:
            from retriever_hybrid import HybridRetriever
            from reranker import HybridRetrieverWithReranking
            
            # Requêtes de test variées
            test_queries = [
                "Quelle est la politique de remboursement?",
                "Formulaire T4A",
                "Article 3.2.1",
                "Procédure de demande de congé",
                "Règlement sur les absences",
                "Guide de l'employé",
                "Politique de voyages internationaux",
                "Code de conduite",
                "Formulaire de réclamation",
                "Chapitre 5 section 2"
            ] * 2  # Répète pour avoir 20 requêtes
            
            test_queries = test_queries[:num_queries]
            
            # Test des différentes configurations
            configs = {
                "dense_only": ("dense", False),
                "bm25_only": ("bm25", False),
                "hybrid_no_rerank": ("hybrid", False),
                "hybrid_with_rerank": ("hybrid", True)
            }
            
            results = {}
            
            for config_name, (method, use_rerank) in configs.items():
                logger.info(f"\n🔍 Configuration: {config_name}")
                
                if use_rerank:
                    retriever = HybridRetrieverWithReranking(
                        data_dir=self.data_dir,
                        use_reranking=True
                    )
                else:
                    retriever = HybridRetriever(data_dir=self.data_dir)
                
                latencies = []
                
                for i, query in enumerate(test_queries, 1):
                    start = time.time()
                    
                    if use_rerank:
                        docs = retriever.search(query, k=5)
                    else:
                        docs = retriever.search(query, k=5, method=method)
                    
                    latency = time.time() - start
                    latencies.append(latency)
                    
                    if i % 5 == 0:
                        logger.info(f"  Requête {i}/{num_queries}: {latency:.3f}s")
                
                # Statistiques
                latencies.sort()
                p50 = latencies[len(latencies)//2]
                p95 = latencies[int(len(latencies)*0.95)]
                p99 = latencies[int(len(latencies)*0.99)]
                avg = sum(latencies) / len(latencies)
                
                results[config_name] = {
                    "num_queries": num_queries,
                    "avg_latency_ms": avg * 1000,
                    "p50_latency_ms": p50 * 1000,
                    "p95_latency_ms": p95 * 1000,
                    "p99_latency_ms": p99 * 1000,
                    "min_latency_ms": min(latencies) * 1000,
                    "max_latency_ms": max(latencies) * 1000
                }
                
                logger.info(f"  ✅ Moyenne: {avg*1000:.1f}ms, P95: {p95*1000:.1f}ms")
            
            self.results["benchmarks"]["latency"] = results
            
            # Affichage comparatif
            logger.info("\n" + "="*80)
            logger.info("📊 COMPARAISON DES LATENCES")
            logger.info("="*80)
            logger.info(f"\n{'Configuration':<25} {'Avg (ms)':<12} {'P50 (ms)':<12} {'P95 (ms)':<12}")
            logger.info("-" * 65)
            for config, stats in results.items():
                logger.info(
                    f"{config:<25} "
                    f"{stats['avg_latency_ms']:>10.1f}  "
                    f"{stats['p50_latency_ms']:>10.1f}  "
                    f"{stats['p95_latency_ms']:>10.1f}"
                )
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Échec: {e}")
            self.results["benchmarks"]["latency"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return False
    
    def benchmark_cache_hit_rate(self, num_queries: int = 50):
        """Benchmark du taux de cache hit."""
        logger.info("\n" + "="*80)
        logger.info(f"BENCHMARK: Taux de Cache Hit ({num_queries} requêtes)")
        logger.info("="*80)
        
        try:
            from retriever_hybrid import HybridRetriever
            
            retriever = HybridRetriever(
                data_dir=self.data_dir,
                use_cache=True,
                cache_size=20
            )
            
            # Simule des requêtes répétées (réaliste pour un chatbot)
            # 10 requêtes uniques, répétées aléatoirement
            unique_queries = [
                "Politique de remboursement",
                "Formulaire T4A",
                "Guide employé",
                "Règlement absences",
                "Procédure congé",
                "Code conduite",
                "Politique voyage",
                "Chapitre 5",
                "Article 3.2",
                "Formulaire réclamation"
            ]
            
            import random
            random.seed(42)
            test_queries = [random.choice(unique_queries) for _ in range(num_queries)]
            
            # Mesure avec cache
            logger.info("\n🔍 Avec cache activé...")
            start = time.time()
            for query in test_queries:
                retriever.search(query, k=5)
            time_with_cache = time.time() - start
            
            # Mesure sans cache (nouveau retriever)
            logger.info("\n🔍 Sans cache...")
            retriever_no_cache = HybridRetriever(
                data_dir=self.data_dir,
                use_cache=False
            )
            start = time.time()
            for query in test_queries:
                retriever_no_cache.search(query, k=5)
            time_without_cache = time.time() - start
            
            # Statistiques
            speedup = time_without_cache / time_with_cache
            
            results = {
                "num_queries": num_queries,
                "num_unique_queries": len(unique_queries),
                "time_with_cache_seconds": time_with_cache,
                "time_without_cache_seconds": time_without_cache,
                "speedup": speedup,
                "avg_latency_with_cache_ms": (time_with_cache / num_queries) * 1000,
                "avg_latency_without_cache_ms": (time_without_cache / num_queries) * 1000
            }
            
            self.results["benchmarks"]["cache"] = results
            
            logger.info("\n" + "="*80)
            logger.info("📊 RÉSULTATS CACHE")
            logger.info("="*80)
            logger.info(f"Temps avec cache:    {time_with_cache:.2f}s")
            logger.info(f"Temps sans cache:    {time_without_cache:.2f}s")
            logger.info(f"Speedup:             {speedup:.2f}x")
            logger.info(f"Économie:            {((1 - 1/speedup) * 100):.1f}%")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Échec: {e}")
            self.results["benchmarks"]["cache"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return False
    
    def save_results(self, output_file: str = "benchmark_results.json"):
        """Sauvegarde les résultats en JSON."""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 Résultats sauvegardés: {output_file}")
    
    def print_summary(self):
        """Affiche un résumé des résultats."""
        logger.info("\n" + "="*80)
        logger.info("📋 RÉSUMÉ DES TESTS ET BENCHMARKS")
        logger.info("="*80)
        
        # Tests
        if "tests" in self.results:
            logger.info("\n🧪 TESTS:")
            for test_name, result in self.results["tests"].items():
                status = result.get("status", "UNKNOWN")
                symbol = "✅" if status == "PASS" else "❌"
                logger.info(f"  {symbol} {test_name}: {status}")
        
        # Benchmarks
        if "benchmarks" in self.results:
            logger.info("\n⚡ BENCHMARKS:")
            
            if "latency" in self.results["benchmarks"]:
                logger.info("  Latence:")
                for config, stats in self.results["benchmarks"]["latency"].items():
                    if "avg_latency_ms" in stats:
                        logger.info(f"    - {config}: {stats['avg_latency_ms']:.1f}ms (avg)")
            
            if "cache" in self.results["benchmarks"]:
                cache_stats = self.results["benchmarks"]["cache"]
                if "speedup" in cache_stats:
                    logger.info(f"  Cache: {cache_stats['speedup']:.2f}x speedup")


# ===== MAIN =====

def main():
    parser = argparse.ArgumentParser(description="Tests et benchmarks pour le système RAG UQAC")
    parser.add_argument(
        "--test",
        choices=["all", "retrieval", "reranking", "rag_engine"],
        default="all",
        help="Tests à exécuter"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Exécute les benchmarks de performance"
    )
    parser.add_argument(
        "--data-dir",
        default="../data",
        help="Répertoire des données"
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Fichier de sortie pour les résultats"
    )
    
    args = parser.parse_args()
    
    # Initialisation
    benchmark = RAGBenchmark(data_dir=args.data_dir)
    
    # Exécute les tests
    if args.test == "all" or args.test == "retrieval":
        benchmark.test_retriever_loading()
        benchmark.test_search_methods()
    
    if args.test == "all" or args.test == "reranking":
        benchmark.test_reranking()
    
    if args.test == "all" or args.test == "rag_engine":
        benchmark.test_rag_engine()
    
    # Exécute les benchmarks
    if args.benchmark:
        benchmark.benchmark_latency(num_queries=20)
        benchmark.benchmark_cache_hit_rate(num_queries=50)
    
    # Sauvegarde et affiche le résumé
    benchmark.save_results(args.output)
    benchmark.print_summary()


if __name__ == "__main__":
    main()