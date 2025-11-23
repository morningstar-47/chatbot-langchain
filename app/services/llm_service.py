"""
Service LLM avec LangChain
Gère l'intégration avec OpenAI et la chaîne conversationnelle avec RAG
"""
from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from typing import Optional, Dict, Any
from app.config import settings
from app.services.vector_store import vector_store_service
from app.services.memory_service import memory_service


class LLMService:
    """Service pour gérer les interactions avec le LLM"""
    
    def __init__(self):
        """Initialise le service LLM avec OpenAI"""
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            api_key=settings.openai_api_key
        )
        self.qa_chain: Optional[ConversationalRetrievalChain] = None
        self._initialize_chain()
    
    def _initialize_chain(self):
        """Initialise la chaîne conversationnelle avec récupération de contexte"""
        # Template de prompt pour la question-réponse amélioré
        template = """Vous êtes un assistant virtuel intelligent qui utilise un système RAG (Retrieval-Augmented Generation) pour fournir des réponses précises et contextuelles.

INSTRUCTIONS IMPORTANTES:
- Utilisez TOUJOURS les informations du contexte fourni ci-dessous pour répondre à la question
- Si le contexte contient des informations pertinentes, basez votre réponse sur ces informations
- Répondez de manière claire, détaillée et utile en français
- Si le contexte ne contient pas d'informations pertinentes, dites-le poliment mais essayez quand même de répondre avec vos connaissances générales si approprié

CONTEXTE RÉCUPÉRÉ DEPUIS LA BASE DE CONNAISSANCES:
{context}

QUESTION DE L'UTILISATEUR: {question}

RÉPONSE (en français, détaillée et basée sur le contexte fourni):"""
        
        QA_PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Créer le retriever depuis le vector store
        try:
            retriever = vector_store_service.get_retriever()
            
            # Créer la chaîne conversationnelle avec RAG
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=None,  # La mémoire sera gérée manuellement par session
                combine_docs_chain_kwargs={"prompt": QA_PROMPT},
                return_source_documents=True,
                verbose=False
            )
        except Exception as e:
            print(f"Attention: Impossible d'initialiser le RAG: {e}")
            print("Le système fonctionnera sans récupération de contexte.")
            self.qa_chain = None
    
    def chat(self, question: str, session_id: str) -> Dict[str, Any]:
        """
        Envoie une question au LLM et retourne la réponse avec le contexte
        
        Args:
            question: Question de l'utilisateur
            session_id: Identifiant de la session pour la mémoire
            
        Returns:
            Dictionnaire contenant la réponse, les sources et l'historique
        """
        if self.qa_chain is None:
            self._initialize_chain()
        
        # Si le RAG n'est pas disponible, utiliser le fallback
        if self.qa_chain is None:
            return self.chat_without_rag(question, session_id)
        
        # Récupérer la mémoire de la session
        memory = memory_service.get_memory(session_id)
        
        # Préparer les inputs avec l'historique
        inputs = {
            "question": question,
            "chat_history": memory.chat_memory.messages
        }
        
        try:
            # Exécuter la chaîne
            result = self.qa_chain.invoke(inputs)
            
            answer = result.get("answer", "Désolé, je n'ai pas pu générer de réponse.")
            source_documents = result.get("source_documents", [])
            
            # Extraire les sources utilisées (dédupliquer par contenu)
            sources = []
            seen_contents = set()
            for doc in source_documents:
                # Normaliser le contenu pour la déduplication
                content_preview = doc.page_content[:200] if len(doc.page_content) > 200 else doc.page_content
                content_hash = hash(doc.page_content.strip())
                
                # Éviter les doublons
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    source_info = {
                        "content": content_preview + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    sources.append(source_info)
            
            # Ajouter à la mémoire
            memory_service.add_message(session_id, question, answer)
            
            return {
                "answer": answer,
                "sources": sources,
                "session_id": session_id
            }
        
        except Exception as e:
            error_message = f"Erreur lors du traitement de la question: {str(e)}"
            print(error_message)
            
            # En cas d'erreur, essayer une réponse simple sans RAG
            try:
                simple_response = self.llm.invoke(question)
                answer = simple_response.content if hasattr(simple_response, 'content') else str(simple_response)
                
                memory_service.add_message(session_id, question, answer)
                
                return {
                    "answer": answer,
                    "sources": [],
                    "session_id": session_id,
                    "error": "RAG non disponible, réponse directe du LLM"
                }
            except Exception as e2:
                return {
                    "answer": "Désolé, une erreur s'est produite. Veuillez réessayer.",
                    "sources": [],
                    "session_id": session_id,
                    "error": str(e2)
                }
    
    def chat_without_rag(self, question: str, session_id: str) -> Dict[str, Any]:
        """
        Chat simple sans récupération de contexte (fallback)
        
        Args:
            question: Question de l'utilisateur
            session_id: Identifiant de la session
            
        Returns:
            Dictionnaire contenant la réponse
        """
        # Récupérer l'historique
        history = memory_service.get_history(session_id)
        
        # Construire le contexte depuis l'historique
        context = ""
        if history:
            for msg in history[-6:]:  # Derniers 6 messages
                if hasattr(msg, 'content'):
                    role = "Utilisateur" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
                    context += f"{role}: {msg.content}\n"
        
        # Construire le prompt
        prompt = f"{context}\nUtilisateur: {question}\nAssistant:"
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            memory_service.add_message(session_id, question, answer)
            
            return {
                "answer": answer,
                "sources": [],
                "session_id": session_id
            }
        except Exception as e:
            return {
                "answer": "Désolé, une erreur s'est produite.",
                "sources": [],
                "session_id": session_id,
                "error": str(e)
            }


# Instance globale du service
llm_service = LLMService()

