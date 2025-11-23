"""
Service de gestion de la mémoire conversationnelle
Gère la mémoire par session utilisateur pour maintenir le contexte
"""
from langchain_classic.memory import ConversationBufferMemory
from typing import Dict


class MemoryService:
    """Service pour gérer la mémoire conversationnelle par session"""
    
    def __init__(self):
        """Initialise le service de mémoire"""
        # Dictionnaire pour stocker les mémoires par session
        self.memories: Dict[str, ConversationBufferMemory] = {}
    
    def get_memory(self, session_id: str) -> ConversationBufferMemory:
        """
        Récupère ou crée la mémoire pour une session
        
        Args:
            session_id: Identifiant unique de la session
            
        Returns:
            Instance de ConversationBufferMemory pour la session
        """
        if session_id not in self.memories:
            self.memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                input_key="question"
            )
        
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
        memory.chat_memory.add_user_message(human_message)
        memory.chat_memory.add_ai_message(ai_message)
    
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
        return memory.chat_memory.messages
    
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


# Instance globale du service
memory_service = MemoryService()

