"""
Service de gestion de la mémoire conversationnelle
Gère la mémoire par session utilisateur pour maintenir le contexte
"""
from langchain_community.chat_message_histories import ChatMessageHistory
from typing import Dict, List, Any, Optional
from datetime import datetime


class MemoryService:
    """Service pour gérer la mémoire conversationnelle par session"""
    
    def __init__(self):
        """Initialise le service de mémoire"""
        # Dictionnaire pour stocker les historiques de messages par session
        self.memories: Dict[str, ChatMessageHistory] = {}
        # Dictionnaire pour stocker l'historique des recherches d'emploi par session
        self.job_searches: Dict[str, List[Dict[str, Any]]] = {}
        # Dictionnaire pour stocker les informations de contexte par session
        self.session_context: Dict[str, Dict[str, Any]] = {}
    
    def get_memory(self, session_id: str) -> ChatMessageHistory:
        """
        Récupère ou crée la mémoire pour une session
        
        Args:
            session_id: Identifiant unique de la session
            
        Returns:
            Instance de ChatMessageHistory pour la session
        """
        if session_id not in self.memories:
            self.memories[session_id] = ChatMessageHistory()
        
        return self.memories[session_id]
    
    def add_message(self, session_id: str, human_message: str, ai_message: str):
        """
        Ajoute un échange de messages à la mémoire d'une session
        
        Args:
            session_id: Identifiant de la session
            human_message: Message de l'utilisateur
            ai_message: Réponse de l'assistant
        """
        memory = self.get_memory(session_id)
        memory.add_user_message(human_message)
        memory.add_ai_message(ai_message)
    
    def get_history(self, session_id: str) -> list:
        """
        Récupère l'historique de conversation d'une session
        
        Args:
            session_id: Identifiant de la session
            
        Returns:
            Liste des messages de l'historique
        """
        if session_id not in self.memories:
            return []
        
        memory = self.memories[session_id]
        return memory.messages
    
    def clear_session(self, session_id: str):
        """
        Réinitialise la mémoire d'une session
        
        Args:
            session_id: Identifiant de la session à réinitialiser
        """
        if session_id in self.memories:
            del self.memories[session_id]
    
    def clear_all(self):
        """Réinitialise toutes les sessions"""
        self.memories.clear()
    
    def get_session_count(self) -> int:
        """
        Retourne le nombre de sessions actives
        
        Returns:
            Nombre de sessions
        """
        return len(self.memories)
    
    def add_job_search(self, session_id: str, job_search_data: Dict[str, Any]):
        """
        Ajoute une recherche d'emploi à l'historique de la session
        
        Args:
            session_id: Identifiant de la session
            job_search_data: Données de la recherche d'emploi (query, country, jobs, etc.)
        """
        if session_id not in self.job_searches:
            self.job_searches[session_id] = []
        
        # Ajouter timestamp
        job_search_data['timestamp'] = datetime.now().isoformat()
        self.job_searches[session_id].append(job_search_data)
        
        # Garder seulement les 10 dernières recherches
        if len(self.job_searches[session_id]) > 10:
            self.job_searches[session_id] = self.job_searches[session_id][-10:]
    
    def get_job_search_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des recherches d'emploi d'une session
        
        Args:
            session_id: Identifiant de la session
            
        Returns:
            Liste des recherches d'emploi effectuées
        """
        return self.job_searches.get(session_id, [])
    
    def get_job_search_context(self, session_id: str) -> str:
        """
        Génère un contexte textuel détaillé des recherches d'emploi précédentes pour la session
        
        Args:
            session_id: Identifiant de la session
            
        Returns:
            Contexte formaté des recherches d'emploi avec détails complets
        """
        searches = self.get_job_search_history(session_id)
        if not searches:
            return ""
        
        context = "\n\n=== CONTEXTE DES RECHERCHES D'EMPLOI PRÉCÉDENTES DANS CETTE SESSION ===\n"
        context += "L'utilisateur peut faire référence à ces emplois dans sa conversation.\n"
        context += "Vous pouvez répondre à des questions sur ces emplois spécifiques.\n\n"
        
        # Prendre la dernière recherche (la plus récente)
        latest_search = searches[-1]
        
        context += "RECHERCHE LA PLUS RÉCENTE:\n"
        context += f"- Requête: {latest_search.get('query', 'N/A')}\n"
        if latest_search.get('country'):
            context += f"- Pays: {latest_search.get('country')}\n"
        if latest_search.get('total'):
            context += f"- Total d'emplois trouvés: {latest_search.get('total')}\n"
        
        # Détails complets de tous les emplois de la dernière recherche
        if latest_search.get('jobs'):
            context += "\nEMPLOIS TROUVÉS (vous pouvez référencer ces emplois par leur numéro ou nom):\n"
            for i, job in enumerate(latest_search.get('jobs', []), 1):
                context += f"\n--- Emploi {i} ---\n"
                context += f"Titre: {job.get('job_title', 'N/A')}\n"
                context += f"Entreprise: {job.get('employer_name', 'N/A')}\n"
                
                location_parts = [job.get('job_city'), job.get('job_state'), job.get('job_country')]
                location = ', '.join([p for p in location_parts if p])
                if location:
                    context += f"Localisation: {location}\n"
                
                if job.get('job_is_remote'):
                    context += "Télétravail: Oui\n"
                
                if job.get('job_employment_type'):
                    context += f"Type: {job.get('job_employment_type')}\n"
                
                if job.get('job_description'):
                    desc = job.get('job_description', '')[:200]
                    context += f"Description: {desc}...\n"
                
                if job.get('job_apply_link'):
                    context += f"Lien candidature: {job.get('job_apply_link')}\n"
                
                if job.get('job_id'):
                    context += f"ID: {job.get('job_id')}\n"
        
        # Si plusieurs recherches, mentionner les précédentes
        if len(searches) > 1:
            other_count = len(searches) - 1
            context += f"\n\nAUTRES RECHERCHES PRÉCÉDENTES ({other_count} autres):\n"
            for i, search in enumerate(searches[-3:-1], 1):  # Avant-dernières recherches
                context += f"- Recherche {i}: {search.get('query', 'N/A')}"
                if search.get('country'):
                    context += f" ({search.get('country')})"
                context += f" - {search.get('total', 0)} emplois trouvés\n"
        
        context += "\n=== FIN DU CONTEXTE DES RECHERCHES ===\n"
        
        return context
    
    def get_latest_job_search(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère la dernière recherche d'emploi d'une session
        
        Args:
            session_id: Identifiant de la session
            
        Returns:
            Dictionnaire de la dernière recherche ou None
        """
        searches = self.get_job_search_history(session_id)
        if not searches:
            return None
        return searches[-1]
    
    def get_job_by_index(self, session_id: str, index: int) -> Optional[Dict[str, Any]]:
        """
        Récupère un emploi spécifique par son index dans la dernière recherche
        
        Args:
            session_id: Identifiant de la session
            index: Index de l'emploi (commence à 1)
            
        Returns:
            Dictionnaire de l'emploi ou None
        """
        latest_search = self.get_latest_job_search(session_id)
        if not latest_search or not latest_search.get('jobs'):
            return None
        
        jobs = latest_search.get('jobs', [])
        if 1 <= index <= len(jobs):
            return jobs[index - 1]
        return None
    
    def find_job_by_name(self, session_id: str, job_title: str = None, employer_name: str = None) -> List[Dict[str, Any]]:
        """
        Trouve des emplois par titre ou nom d'employeur dans la dernière recherche
        
        Args:
            session_id: Identifiant de la session
            job_title: Titre de l'emploi à rechercher (optionnel)
            employer_name: Nom de l'employeur à rechercher (optionnel)
            
        Returns:
            Liste des emplois correspondants
        """
        latest_search = self.get_latest_job_search(session_id)
        if not latest_search or not latest_search.get('jobs'):
            return []
        
        jobs = latest_search.get('jobs', [])
        results = []
        
        for job in jobs:
            match = True
            if job_title:
                job_title_lower = job.get('job_title', '').lower()
                if job_title.lower() not in job_title_lower:
                    match = False
            if employer_name:
                employer_lower = job.get('employer_name', '').lower()
                if employer_name.lower() not in employer_lower:
                    match = False
            
            if match:
                results.append(job)
        
        return results
    
    def update_session_context(self, session_id: str, key: str, value: Any):
        """
        Met à jour une information de contexte pour une session
        
        Args:
            session_id: Identifiant de la session
            key: Clé de l'information
            value: Valeur à stocker
        """
        if session_id not in self.session_context:
            self.session_context[session_id] = {}
        self.session_context[session_id][key] = value
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """
        Récupère toutes les informations de contexte d'une session
        
        Args:
            session_id: Identifiant de la session
            
        Returns:
            Dictionnaire des informations de contexte
        """
        return self.session_context.get(session_id, {})
    


# Instance globale du service
memory_service = MemoryService()

