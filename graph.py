from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
import json
import re


# Graph State
class GraphState(TypedDict):
    query: str
    documents: List[Document]
    context: str
    answer: str
    is_relevant: bool
    vector_store: object
    retries: int

class LangGraph:

    llm = OllamaLLM(model="llama3:8b")
        
    # Node 1: Retrieve
    def retrieve_node(state: GraphState):
        query = state["query"]
        vector_store = state["vector_store"]

        retriever = vector_store.as_retriever(search_kwargs={"k":3})
        results = retriever.invoke(query)

        context = "\n".join([doc.page_content for doc in results])


        return {
            "documents": results,
            "context" : context
        }
    

    # Node 2: Relevance Check
    def relevance_check(state: GraphState):
        query = state["query"]
        documents = state["documents"]

        context = "\n".join([doc.page_content for doc in documents])

        prompt = f"""
        You are a medical document assistant.

        STRICT RULES:
        - Use ONLY the context
        - If not found, say: "Not available in the document"
        - Answer only as YES or NO

        Context:
        {context}

        User Query:
        {query}

        Answer clearly and concisely.
        """

        result = LangGraph.llm.invoke(prompt).strip().upper()

        return {
            "context": context,
            "is_relevant": "YES" in result
        }


    # Node 3: Generate Answer
    def generate_node(state: GraphState):
        query = state["query"]
        context = state["context"]

        if not context.strip():
            return {
                "answer": "I couldn't find relevant information in the document."
            }

        prompt = prompt = prompt = f"""
        You are a medical document information extractor.

        Your task is to extract structured patient information from the given context.

        STRICT RULES:
        - Use ONLY the provided context
        - Do NOT infer or assume anything
        - If a field is missing, return: "Not available"
        - Be precise and do not add explanations
        - If multiple values exist, return them as a comma-separated list
        - Return ONLY valid JSON
        - Do NOT add any text before or after JSON
        - Output must start with '{' and end with '}'

        Extract the following fields:

        1. Patient Name
        2. Age
        3. Gender
        4. Medical Condition / Disease
        5. Symptoms
        6. Diagnosis
        7. Treatment / Medication
        8. Doctor Name (if available)
        9. Hospital / Clinic (if available)
        10. Date (if available)

        OUTPUT FORMAT (STRICT JSON):
        {{
        "patient_name": "...",
        "age": "...",
        "gender": "...",
        "medical_condition": "...",
        "symptoms": "...",
        "diagnosis": "...",
        "treatment": "...",
        "doctor": "...",
        "hospital": "...",
        "date": "..."
        }}

        Context:
        {context}
        """

        raw_output = LangGraph.llm.invoke(prompt)

        try : 
            json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            clean_json = json.loads(json_match.group())
            
        except Exception as e:
            clean_json = {
                "error" : "Invalid Json outptu",
                "raw_output" : raw_output
            } 

        return {
            "answer": clean_json
        }

    def retry_node(state:GraphState):
        return {
            "retries" : state["retries"]+1
        } 

    # Build Graph
    def build_graph(use_check= False) :
        builder = StateGraph(GraphState)
        builder.add_node("retrieve", LangGraph.retrieve_node)
        builder.add_node("check", LangGraph.relevance_check)
        builder.add_node("generate", LangGraph.generate_node)
        builder.add_node("retry", LangGraph.retry_node)
        builder.set_entry_point("retrieve")
        if use_check :
            builder.add_edge("retrieve", "check")

            # Conditional Logic
            def decide_next(state: GraphState):
                if state["is_relevant"]:
                    return "generate"
                if state["retries"] >=2:
                    return "generate"
                
                return "retry"  # retry


            builder.add_conditional_edges(
                "check",
                decide_next,
                {
                    "generate": "generate",
                    "retry": "retry"
                }
            )
            builder.add_edge("retry", "retrieve")
        
        else : 
            builder.add_edge("retrieve", "generate")

        builder.add_edge("generate", END)

        return builder.compile()
    

    @staticmethod
    def run(query:str, vector_store, use_check= False) :
        graph = LangGraph.build_graph(use_check=use_check)
        result = graph.invoke({
            "query" : query,
            "vector_store" : vector_store,
            "retries": 0
        })
        return result["answer"]