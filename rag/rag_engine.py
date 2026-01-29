"""
RAG Engine v2 - Avec Historique et Reformulation Contextuelle

Améliorations par rapport à la v1 :
1. ✅ Gestion intelligente de l'historique conversationnel
2. ✅ Détection automatique du besoin de reformulation
3. ✅ Query decontextualization via LLM local (Ollama)
4. ✅ Filtrage par métadonnées (chapitre, type de document)
5. ✅ Logging détaillé pour debugging

Gère les conversations multi-tours :
  User: "Quelle est la politique de voyage?"
  Bot: "Voici la politique..."
  User: "Et pour les voyages internationaux?" ← Question contextuelle
  Bot: [Reformule automatiquement] → "Quelle est la politique de voyage pour les voyages internationaux?"
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from retriever import HybridRetriever

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGEngineV2:
    """
    Moteur RAG avec gestion d'historique et reformulation contextuelle.
    """
    
    def __init__(
        self,
        data_dir: str = "../data",
        retrieval_method: str = "hybrid",
        max_history: int = 5,
        use_ollama: bool = True,
        ollama_model: str = "llama3.2"
    ):
        """
        Initialise le RAG Engine v2.
        
        Args:
            data_dir: Répertoire contenant les données
            retrieval_method: "hybrid", "dense", ou "bm25"
            max_history: Nombre max d'échanges à conserver en historique
            use_ollama: Utilise Ollama pour reformulation (si False, regex simple)
            ollama_model: Modèle Ollama à utiliser (llama3.2, mistral, etc.)
        """
        logger.info("🚀 Initialisation du RAG Engine v2...")
        
        # Retriever hybride
        self.retriever = HybridRetriever(data_dir=data_dir)
        self.retrieval_method = retrieval_method
        
        # Historique conversationnel
        self.history = []
        self.max_history = max_history
        
        # Configuration Ollama
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        
        if use_ollama:
            self._check_ollama_available()
        
        logger.info(f"✅ RAG Engine v2 prêt (méthode: {retrieval_method}, ollama: {use_ollama})")
    
    def _check_ollama_available(self):
        """Vérifie que Ollama est installé et accessible de manière robuste."""
        try:
            import ollama
            # On tente juste de lister les modèles sans chercher de clé spécifique
            ollama.list()
            logger.info("✅ Service Ollama détecté et accessible.")
        except ImportError:
            logger.warning("⚠️ Bibliothèque 'ollama' non installée. Lancez: pip install ollama")
            self.use_ollama = False
        except Exception as e:
            logger.warning(f"⚠️ Ollama ne répond pas (vérifiez que l'app est lancée) : {e}")
            self.use_ollama = False
    
    def _needs_decontextualization(self, query: str) -> bool:
        """
        Détecte si la question nécessite une reformulation contextuelle.
        
        Indicateurs :
        - Commence par des connecteurs (et, aussi, puis, etc.)
        - Contient des pronoms déictiques (ça, celui-ci, etc.)
        - Question très courte (< 4 mots)
        - Références explicites (dans ce cas, pour ça, etc.)
        
        Returns:
            True si reformulation nécessaire
        """
        # Pas d'historique → pas besoin
        if not self.history:
            return False
        
        query_lower = query.lower().strip()
        
        # Motifs de référence contextuelle
        contextual_patterns = [
            r'^(et|aussi|de plus|puis|ensuite|également|ou)\b',  # Connecteurs
            r'\b(ça|cela|celui-ci|celle-ci|ceux-ci|ce dernier)\b',  # Pronoms déictiques
            r'^(dans ce cas|pour ça|à ce sujet|là-dessus)\b',  # Références explicites
            r'^(pour|concernant|à propos de)\s+(le|la|les|l\')\b',  # Références implicites
        ]
        
        for pattern in contextual_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"🔍 Détection contextuelle (motif: {pattern})")
                return True
        
        # Question très courte (probable référence)
        word_count = len(query_lower.split())
        if word_count < 4:
            logger.info(f"🔍 Détection contextuelle (question courte: {word_count} mots)")
            return True
        
        return False
    
    def _decontextualize_with_ollama(self, query: str) -> str:
        """
        Reformule la question avec Ollama en intégrant le contexte.
        
        Args:
            query: Question actuelle
        
        Returns:
            Question reformulée (autonome)
        """
        try:
            import ollama
            
            # Construit le contexte des 3 derniers échanges
            context_lines = []
            for q, a in self.history[-3:]:
                context_lines.append(f"Q: {q}")
                # Limite la réponse à 200 chars pour garder le prompt court
                answer_preview = a[:200] + "..." if len(a) > 200 else a
                context_lines.append(f"R: {answer_preview}")
            
            context = "\n".join(context_lines)
            
            # Prompt de reformulation
            prompt = f"""Tu es un assistant qui reformule des questions en intégrant le contexte d'une conversation.

Contexte de la conversation précédente :
{context}

Question actuelle de l'utilisateur : {query}

Tâche : Reformule la question actuelle pour qu'elle soit autonome et compréhensible sans le contexte.
Si la question est déjà autonome, retourne-la telle quelle.
Réponds UNIQUEMENT avec la question reformulée, sans explication.

Question reformulée :"""
            
            logger.info(f"🤖 Reformulation Ollama (modèle: {self.ollama_model})...")
            
            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt
            )
            
            reformulated = response['response'].strip()
            
            # Nettoyage (enlève guillemets, etc.)
            reformulated = reformulated.strip('"\'')
            
            logger.info(f"✅ Reformulation: '{query}' → '{reformulated}'")
            
            return reformulated
        
        except Exception as e:
            logger.error(f"❌ Erreur Ollama: {e}")
            logger.info("↩️  Fallback: question originale")
            return query
    
    def _decontextualize_simple(self, query: str) -> str:
        """
        Reformulation simple sans LLM (fallback).
        
        Stratégie basique :
        - Remplace "Et X" → "X"
        - Ajoute le sujet de la question précédente si manquant
        """
        if not self.history:
            return query
        
        last_question, _ = self.history[-1]
        
        # Enlève les connecteurs en début de phrase
        query_clean = re.sub(r'^(et|aussi|puis|ensuite)\s+', '', query, flags=re.I)
        
        # Si la question est très courte, ajoute le contexte de la dernière question
        if len(query_clean.split()) < 4:
            # Extraction naive du sujet de la dernière question
            # Ex: "Quelle est la politique..." → "politique"
            subject_match = re.search(r'(politique|règlement|procédure|formulaire|guide)\s+\w+', 
                                     last_question, re.I)
            if subject_match:
                subject = subject_match.group(0)
                query_clean = f"{query_clean} concernant {subject}"
        
        logger.info(f"✅ Reformulation simple: '{query}' → '{query_clean}'")
        return query_clean
    
    def ask(
        self, 
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict] = None,
        force_decontextualization: bool = False
    ) -> Tuple[str, List[Dict]]:
        """
        Traite une question utilisateur.
        
        Args:
            query: Question de l'utilisateur
            k: Nombre de documents à récupérer
            filter_metadata: Filtres optionnels (ex: {"chapitre": "Chapitre 5"})
            force_decontextualization: Force la reformulation même si non détectée
        
        Returns:
            (réponse, documents_sources)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"📝 Question: {query}")
        logger.info(f"{'='*80}")
        
        # 1. Détection du besoin de reformulation
        needs_reformulation = self._needs_decontextualization(query) or force_decontextualization
        
        # 2. Reformulation si nécessaire
        if needs_reformulation:
            logger.info("🔄 Reformulation contextuelle activée")
            if self.use_ollama:
                query_reformulated = self._decontextualize_with_ollama(query)
            else:
                query_reformulated = self._decontextualize_simple(query)
        else:
            query_reformulated = query
            logger.info("✅ Question autonome (pas de reformulation)")
        
        # 3. Récupération des documents
        logger.info(f"🔍 Recherche ({self.retrieval_method})...")
        contexts = self.retriever.search(
            query_reformulated,
            k=k,
            method=self.retrieval_method,
            filter_metadata=filter_metadata
        )
        
        logger.info(f"✅ {len(contexts)} documents récupérés")
        
        # Log des sources
        for i, ctx in enumerate(contexts, 1):
            title = ctx["metadata"]["title"][:50]
            doc_type = ctx["metadata"]["doc_type"]
            logger.info(f"  {i}. [{doc_type}] {title}...")
        
        # 4. Construction du prompt (à adapter selon votre LLM)
        prompt = self._build_prompt(query, contexts)
        
        # 5. Génération de la réponse (simulée pour l'instant)
        # TODO: Remplacer par appel à votre LLM local (Ollama)
        answer = self._generate_answer(prompt, contexts)
        
        # 6. Mise à jour de l'historique
        self._update_history(query, answer)
        
        logger.info(f"✅ Réponse générée ({len(answer)} caractères)")
        
        return answer, contexts
    
    def _build_prompt(self, query: str, contexts: List[Dict]) -> str:
        """
        Construit le prompt pour le LLM.
        
        Inclut :
        - L'historique conversationnel
        - Les documents sources
        - La question
        """
        prompt_parts = []
        
        # Instruction système
        prompt_parts.append(
            "Tu es un assistant spécialisé dans le manuel de gestion de l'UQAC. "
            "Réponds de manière précise et factuelle en te basant uniquement sur les documents fournis. "
            "Si l'information n'est pas dans les documents, dis-le clairement."
        )
        
        # Historique (si présent)
        if self.history:
            prompt_parts.append("\n## Historique de la conversation")
            for q, a in self.history[-3:]:  # 3 derniers échanges
                prompt_parts.append(f"User: {q}")
                prompt_parts.append(f"Assistant: {a[:150]}...")  # Tronque pour économiser tokens
        
        # Documents sources
        prompt_parts.append("\n## Documents sources")
        for i, ctx in enumerate(contexts, 1):
            metadata = ctx["metadata"]
            prompt_parts.append(f"\n### Source {i}")
            prompt_parts.append(f"Titre: {metadata['title']}")
            prompt_parts.append(f"Type: {metadata['doc_type']}")
            if metadata.get('chapitre'):
                prompt_parts.append(f"Chapitre: {metadata['chapitre']}")
            prompt_parts.append(f"Contenu:\n{ctx['text']}")
        
        # Question
        prompt_parts.append(f"\n## Question\n{query}")
        
        prompt_parts.append("\n## Réponse")
        
        return "\n".join(prompt_parts)
    
    def _generate_answer(self, prompt: str, contexts: List[Dict]) -> str:
        """
        Génère une réponse réelle avec Ollama au lieu de la simulation.
        """
        if not self.use_ollama:
            return "⚠️ Mode simulation : Ollama n'est pas détecté."

        try:
            import ollama
            logger.info(f"🤖 Radisson génère la réponse finale via {self.ollama_model}...")
        
            # Appel réel au modèle local
            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                options={
                    "temperature": 0.2,  # Pour rester très factuel
                    "top_p": 0.9
                }
            )
        
            return response['response'].strip()

        except Exception as e:
            logger.error(f"❌ Erreur Ollama : {e}")
            return "Je n'ai pas pu générer de réponse suite à un problème technique local."
    
    def _update_history(self, query: str, answer: str):
        """Met à jour l'historique conversationnel (FIFO)."""
        self.history.append((query, answer))
        
        # Limite la taille de l'historique
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        logger.info(f"📚 Historique mis à jour ({len(self.history)}/{self.max_history})")
    
    def clear_history(self):
        """Efface l'historique conversationnel."""
        self.history = []
        logger.info("🗑️  Historique effacé")
    
    def get_history(self) -> List[Tuple[str, str]]:
        """Retourne l'historique conversationnel."""
        return self.history.copy()


# ===== EXEMPLE D'UTILISATION =====

if __name__ == "__main__":
    # Initialisation
    engine = RAGEngineV2(
        data_dir="../data",
        retrieval_method="hybrid",
        use_ollama=True  # Mettre False si Ollama pas installé
    )
    
    # Simulation d'une conversation multi-tours
    print("\n" + "="*80)
    print("🤖 SIMULATION DE CONVERSATION MULTI-TOURS")
    print("="*80 + "\n")
    
    # Question 1 (autonome)
    answer1, docs1 = engine.ask("Quelle est la politique de remboursement des frais de voyage?")
    print(f"User: Quelle est la politique de remboursement des frais de voyage?")
    print(f"Bot: {answer1}\n")
    
    # Question 2 (contextuelle - devrait déclencher reformulation)
    answer2, docs2 = engine.ask("Et pour les voyages internationaux?")
    print(f"User: Et pour les voyages internationaux?")
    print(f"Bot: {answer2}\n")
    
    # Question 3 (avec filtre)
    answer3, docs3 = engine.ask(
        "Quels sont les formulaires nécessaires?",
        filter_metadata={"doc_type": "Formulaire"}
    )
    print(f"User: Quels sont les formulaires nécessaires?")
    print(f"Bot: {answer3}\n")
    
    # Affichage de l'historique
    print("\n" + "="*80)
    print("📚 HISTORIQUE DE LA CONVERSATION")
    print("="*80)
    for i, (q, a) in enumerate(engine.get_history(), 1):
        print(f"\n{i}. Q: {q}")
        print(f"   R: {a[:100]}...")
