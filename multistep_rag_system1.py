from dotenv import load_dotenv
from typing import TypedDict,List,Literal
import os
import sqlite3

from langchain_core.messages import SystemMessage,AIMessage,HumanMessage,BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.document import Document

from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.sqlite import SqliteSaver

from schema.validator import ValidateQuestion,GradeDocument
from models.llms import llm
from tools import retriever_tool

load_dotenv()

#make sqlite connection
sqlite_conn = sqlite3.connect("1_multistep_rag.sqlite",check_same_thread=False)
checkpointer = SqliteSaver(sqlite_conn)

class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents:List[Document]
    is_topic : str
    rephrased_question :str
    proceed_to_generate: bool
    rephrase_count : int
    question: HumanMessage


def question_rewriter(state:AgentState) -> AgentState:
    """Rewrites the input question into a single standalone question."""

    state["documents"] = []
    state["is_topic"] = ""
    state["proceed_to_generate"] = False
    state["rephrase_count"] = 0
    state["rephrased_question"] = ""

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    if state["question"] not in state["messages"]:
        state["messages"].append(state["question"])
    
    if len(state["messages"]) > 1:
        conversation = state["messages"][:-1]
        question = state["question"].content
        messages = [
            SystemMessage(content="You are a helpful assisatant that rephrases the user's question into a standalone question optimized for retrieval.")

        ]
        messages.extend(conversation)
        messages.append(HumanMessage(content=question))
        
        rephrase_prompt = ChatPromptTemplate.from_messages(messages)
        prompt = rephrase_prompt.format()

      
        response = llm.invoke(prompt)
       
        rephrased_question = response.content.strip()
        state["rephrased_question"] = rephrased_question
    
    else:
        state["rephrased_question"] = state["question"].content

    return state
    
def question_classifier(state:AgentState):
    """Classifies whether the user's question is related to topic."""
    classifier_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a classifier that determines whether the user's question is related to specific topic.
                      Topics:
                            1. Attention is all you need research paper
                      If the question is related to the topic respond with 'Yes',
                      otherwise respond with 'No'. """),
        HumanMessage(content=f"User question:{state["rephrased_question"]}")
    ]
    )
  
    structured_llm = llm.with_structured_output(ValidateQuestion)
    classifier_chain = classifier_prompt | structured_llm
    response = classifier_chain.invoke({})
    state["is_topic"] = response.score.strip()

    return state

def topic_router(state:AgentState)-> Literal["retrieve","off-topic-response"]:

    is_topic = state.get("is_topic","")
    if is_topic.strip().lower() == "yes":
        return "retrieve"
    else:
        return "off-topic-response"
    

   


def retrieve(state:AgentState):

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Get all PDF files in the current directory
    pdf_names = [filename for filename in os.listdir(current_dir) if filename.endswith(".pdf")]

    # Generate full paths for the PDF files
    pdf_dir = [os.path.join(current_dir, pdfname) for pdfname in pdf_names]

 
    retriever = retriever_tool(pdf_dir=pdf_dir)

    documents = retriever.invoke(state["rephrased_question"])
    state["documents"] = documents

    return state



def retrival_grader(state:AgentState):
    system_message = SystemMessage(content="""
                    You are a helpful grader accessing the relevance of a retrived document to user's question.
                    only answer with 'Yes' or 'No'.
                    If the document contains the information relevant to the user's question,respond with 'Yes',
                    otherwise respond with 'No'.""")
    
   
    
    structured_llm = llm.with_structured_output(GradeDocument)

    relevant_docs = []
    for doc in state["documents"]:
        human_message = HumanMessage(content=
                                     f"User question:{state['rephrased_question']}\n\n Retrieved document:{doc.page_content}")
        
        grade_prompt = ChatPromptTemplate.from_messages([system_message,human_message])
        grader_chain = grade_prompt | structured_llm
        response = grader_chain.invoke({})

        if response.score.strip().lower()=="yes":
            relevant_docs.append(doc)
    state["documents"] = relevant_docs
    state["proceed_to_generate"] = len(relevant_docs) >0

    return state

def proceed_router(state:AgentState):
    rephrase_count = state.get("rephrase_count",0)

    if state.get("proceed_to_generate",False):
        return "generate_answer"
    elif rephrase_count >=2:
        return "cannot_answer"
    else:
        return "refine_question"


def refine_question(state:AgentState):

    rephrase_count = state.get("rephrase_count",0)
    if rephrase_count >=2:
        
        return state
    refine_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a helpful assistant that slightly refines user's question for improving retrieval results.
                      provide a slightly adjusted version of the question."""),
        HumanMessage(content=f"Original question: {state["rephrased_question"]}\n provide slightly refine question.")
    ])
    prompt = refine_prompt.format()
  
    response = llm.invoke(prompt)

    state["rephrased_question"] = response.content.strip()
    state["rephrase_count"] = rephrase_count + 1

    return state

def generate_answer(state:AgentState):
    if "messages" not in state or state["messages"] is None:
        raise ValueError("state must include 'messages'")
    
    

   
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based on the chat history and context. Take the latest question into consideration.

        Chat history: {chat_history}
        Context: {context}
        Question: {question}

        Answer wisely and accurately.
        """
    )

    formatted_prompt = prompt.format_messages(
        chat_history=str(state["messages"]),
        context="\n\n".join([doc.page_content for doc in state["documents"]]),
        question=state["rephrased_question"]
    )

    response = llm.invoke(formatted_prompt)

    state["messages"].append(AIMessage(content=response.content.strip()))

    return state

def cannot_answer(state:AgentState):
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(AIMessage(content="I'm sorry.I can't find the information you are seeking for."))
    return state

def off_topic_response(state:AgentState):
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(AIMessage(content="I'm sorry.I can't answer the question."))
    return state


workflow = StateGraph(AgentState)

# creating nodes

workflow.add_node("question_rewritter",question_rewriter)
workflow.add_node("question_classifier", question_classifier)
workflow.add_node("off-topic-response", off_topic_response)
workflow.add_node("retrieve", retrieve)
workflow.add_node("retrieval_grader", retrival_grader)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("refine_question", refine_question)
workflow.add_node("cannot_answer", cannot_answer)

# connecting nodes through edges
workflow.add_edge("question_rewritter","question_classifier")
workflow.add_conditional_edges("question_classifier",topic_router)
workflow.add_edge("retrieve","retrieval_grader")
workflow.add_conditional_edges("retrieval_grader",proceed_router)


workflow.add_edge("refine_question","retrieve")
workflow.add_edge("generate_answer",END)
workflow.add_edge("cannot_answer",END)
workflow.add_edge("off-topic-response",END)
workflow.set_entry_point("question_rewritter")
graph = workflow.compile(checkpointer=checkpointer)





