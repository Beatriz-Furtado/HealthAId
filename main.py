from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from typing_extensions import Dict, Annotated
from langgraph.graph import StateGraph, END, START, MessagesState
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from flask import Flask, render_template, request, jsonify

from pydantic import BaseModel, Field

import os
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key="")

csv_path = "doencas.csv"
loader = CSVLoader(file_path=csv_path)
docs = loader.load()

embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()

def get_documents_for_specialist(age_range, specialty):
    query = f"Filtre doenças para a faixa etária {age_range} e que sejam relevantes para o especialista {specialty}."
    documents = retriever.get_relevant_documents(query)
    return documents
    
class AgentsState(MessagesState):
    
    pacient_symptoms: Annotated[list[str], "Pacient's symptoms"]
    
    age_classification: Annotated[str, "Pacient's age classification"]
        
    selected_specialty: Annotated[list[str], "Selected medical specialties"]

    specialty: Annotated[str, "Name of the medical specialty"]

    
def supervisor(state: AgentsState) -> AgentsState:
    system_prompt = """
    Extract only the age, symptoms and name from the pacient's text.
    According to age, classify between: recem-nascido (0 to 28 days), lactente (29 days to 1 year), pre-escolar (2 years to 4 years) and escolar (5 years to 11 years).
    Return the classification of the pacient's age.
    If the age is not in the classification (over 11 years old) or the request is unclear, respond necessarily with a empty list.
    EXAMPLES FOR THE ANSWER:
    pre-escolar
    recem-nascido
    []
    """
    user_message = state["messages"][-1].content
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    response = llm.invoke(messages)

    messages.append(response)
    state["age_classification"] = response.content

    return {"messages": messages}

class ActivateSpecialty(BaseModel):
    """
    This field should be used to store the specialties related to the patient's symptoms
    """
    activate_specialty: list[str] = Field(description="This field should be used to store the specialties related to the patient's symptoms")

llm_structured = llm.with_structured_output(ActivateSpecialty)

def specialty_identifier(state: AgentsState) -> AgentsState:
    system_prompt = """
    You are a doctor expert in identifying which medical specialties are related to the patient's symptoms.
    Identify which medical specialties below are related to the patient's symptoms saved in the variable pacient_symptoms.
    
    respiratorio: febre, tosse, dificuldade para respirar, chiado, febre, dor de garganta, dificuldade de engolir, rouquidão, produção de muco, congestão nasal, dor facial, falta de ar.
    cardiovascular: dificuldade para respirar, cianose, cansaço, dor no torax, arritmia.
    urinario: febre, dor ao urinar, urina com odor forte, cansaço, inchaço, urina espumosa, sangue na urina, hipertensão, urgencia urinaria, dificuldade para urinar, dor lombar, náuseas.
    dermatologico: pele seca, coceira, lesões avermelhadas, manchas vermelhas na pele, inchaço, bolhas na pele, crostas amareladas, vermilhidão.
    hematologico: manchas roxas na pele, sangramento gengival, fadiga, icterícia, aumento do baço, cansaço, palidez, fraqueza, falta de apetite, olhos amarelados.
    digestivo: vomito, diarreia, febre, regurgitação frequente, irritabilidade, dificuldade para ganhar peso, desidratação, fezes ressecadas, dor abdominal, dificuldade para evacuar, distensão abdominal, perda de peso, perda de apetite, queimação ou dor no estômago.
    neurologico: febre alta, rigidez na nuca, irritabilidade, dor de cabeça intensa, sensibilidade à luz, náusea.
    infeccioso: febre, fraqueza, dificuldade para respirar, alterações no desenvolvimento, baixo peso ao nascer.
    reumatologico: dor nas pernas, febre, dores articulares, manchas na pele, inchaço, rigidez matinal.
    clinico_geral: febre, manchas vermelhas no corpo, coceira, mal-estar, febre, tosse, coriza, abaulamento na região inguinal, dor local, choro, abaulamento no escroto, dor, irritação, inchaço no escroto, líquido ao redor do testículo, vomito, diarreia, irritação na pele, dor no corpo, gases, dor abdominal.
    
    Respond only with a list with the names of the selected medical especialties.
    If the request is unclear or the age is over 11 years old, respond necessarily with a empty list.
    EXAMPLES FOR THE ANSWER:
    [respiratorio, cardiovascular, infeccioso]
    [dermatologico, clinico_geral]
    [digestivo]
    []
    """
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm_structured.invoke(messages)
    
    selected_specialty = response.activate_specialty
    messages.append(AIMessage(content=", ".join(selected_specialty)))
    
    if not selected_specialty:
        message = "Desculpe-me. Por favor, verifique se os dados de entrada estão de acordo com a jurisdição pediátrica.\nInfelizmente, só atendemos crianças de 0 a 11 anos."
        messages.append(SystemMessage(content=message))

    state["selected_specialty"] += selected_specialty
    return {"messages": messages}

def respiratorio(state: AgentsState) -> AgentsState:
    state["specialty"] = "respiratorio"
    age_range = state.get("age_classification")
    filtered_docs = get_documents_for_specialist(age_range, "respiratorio")
    
    system_prompt = f"""
    You are a doctor expert in analyzing symptoms and hypothesizing respiratory system diseases that match the patient's symptoms.
    Here are the documents filtered for diseases related to age {age_range} and specialty respiratoria:
    {filtered_docs}

    Based on these data, provide hypotheses of respiratory diseases that are related to the patient's symptoms and age.   
    """
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages)

    messages.append(response)
    return {"messages": messages}


def cardiovascular(state: AgentsState) -> AgentsState:
    state["specialty"] = "cardiovascular"
    age_range = state.get("age_classification")
    filtered_docs = get_documents_for_specialist(age_range, "cardiovascular")
    
    system_prompt = f"""
    You are a doctor expert in analyzing symptoms and hypothesizing cardiovascular diseases that match the patient's symptoms.
    Here are the documents filtered for diseases related to age {age_range} and specialty cardiovascular:
    {filtered_docs}

    Based on these data, provide hypotheses of cardiovascular diseases that are related to the patient's symptoms and age.   
    """
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages)

    messages.append(response)
    return {"messages": messages}

def urinario(state: AgentsState) -> AgentsState:
    state["specialty"] = "urinario"
    age_range = state.get("age_classification")
    filtered_docs = get_documents_for_specialist(age_range, "urinario")
    
    system_prompt = f"""
    You are a doctor expert in analyzing symptoms and hypothesizing urinary system diseases that match the patient's symptoms.
    Here are the documents filtered for diseases related to age {age_range} and specialty urinario:
    {filtered_docs}

    Based on these data, provide hypotheses of urinary system diseases that are related to the patient's symptoms and age.   
    """
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages)

    messages.append(response)
    return {"messages": messages}

def dermatologico(state: AgentsState) -> AgentsState:
    state["specialty"] = "dermatologico"
    age_range = state.get("age_classification")
    filtered_docs = get_documents_for_specialist(age_range, "dermatologico")
    
    system_prompt = f"""
    You are a doctor expert in analyzing symptoms and hypothesizing dermatological diseases that match the patient's symptoms.
    Here are the documents filtered for diseases related to age {age_range} and specialty dermatologico:
    {filtered_docs}

    Based on these data, provide hypotheses of dermatological diseases that are related to the patient's symptoms and age.   
    """
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages)

    messages.append(response)
    return {"messages": messages}

def hematologico(state: AgentsState) -> AgentsState:
    state["specialty"] = "hematologico"
    age_range = state.get("age_classification")
    filtered_docs = get_documents_for_specialist(age_range, "hematologico")
    
    system_prompt = f"""
    You are a doctor expert in analyzing symptoms and hypothesizing hematological diseases that match the patient's symptoms.
    Here are the documents filtered for diseases related to age {age_range} and specialty hematologico:
    {filtered_docs}

    Based on these data, provide hypotheses of hematological diseases that are related to the patient's symptoms and age.
    """
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages)

    messages.append(response)
    return {"messages": messages}

def digestivo(state: AgentsState) -> AgentsState:
    state["specialty"] = "digestivo"
    age_range = state.get("age_classification")
    filtered_docs = get_documents_for_specialist(age_range, "digestivo")
    
    system_prompt = f"""
    You are a doctor expert in analyzing symptoms and hypothesizing digestive system diseases that match the patient's symptoms.
    Here are the documents filtered for diseases related to age {age_range} and specialty digestivo:
    {filtered_docs}

    Based on these data, provide hypotheses of digestive system diseases that are related to the patient's symptoms and age.
    """
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages)

    messages.append(response)
    return {"messages": messages}

def neurologico(state: AgentsState) -> AgentsState:
    state["specialty"] = "neurologico"
    age_range = state.get("age_classification")
    filtered_docs = get_documents_for_specialist(age_range, "neurologico")
    
    system_prompt = f"""
    You are a doctor expert in analyzing symptoms and hypothesizing neurological diseases that match the patient's symptoms.
    Here are the documents filtered for diseases related to age {age_range} and specialty neurologico:
    {filtered_docs}

    Based on these data, provide hypotheses of neurological diseases that are related to the patient's symptoms and age.
    """
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages)

    messages.append(response)
    return {"messages": messages}

def infeccioso(state: AgentsState) -> AgentsState:
    state["specialty"] = "infeccioso"
    age_range = state.get("age_classification")
    filtered_docs = get_documents_for_specialist(age_range, "infeccioso")
    
    system_prompt = f"""
    You are a doctor expert in analyzing symptoms and hypothesizing infectious system diseases that match the patient's symptoms.
    Here are the documents filtered for diseases related to age {age_range} and specialty infeccioso:
    {filtered_docs}

    Based on these data, provide hypotheses of infectious diseases that are only related to the patient's symptoms and age.
    """
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages)

    messages.append(response)
    return {"messages": messages}

def reumatologico(state: AgentsState) -> AgentsState:
    state["specialty"] = "reumatologico"
    age_range = state.get("age_classification")
    filtered_docs = get_documents_for_specialist(age_range, "reumatologico")
    
    system_prompt = f"""
    You are a doctor expert in analyzing symptoms and hypothesizing rheumatological system diseases that match the patient's symptoms.
    Here are the documents filtered for diseases related to age {age_range} and specialty reumatologico:
    {filtered_docs}

    Based on these data, provide hypotheses of rheumatological diseases that are related to the patient's symptoms and age.
    """
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages)

    messages.append(response)
    return {"messages": messages}

def clinico_geral(state: AgentsState) -> AgentsState:
    state["specialty"] = "clinico_geral"
    age_range = state.get("age_classification")
    filtered_docs = get_documents_for_specialist(age_range, "clinico_geral")
    
    system_prompt = f"""
    You are a doctor expert in analyzing symptoms and hypothesizing non-specific system diseases that match the patient's symptoms.
    Here are the documents filtered for diseases related to age {age_range} and specialty clinico_geral:
    {filtered_docs}

    Based on these data, provide hypotheses of non-specific diseases that are related to the patient's symptoms and age.
    """
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages)

    messages.append(response)
    return {"messages": messages}

def recommended_medical_examination(state: AgentsState) -> AgentsState:
    system_prompt = """
    You are a doctor expert in analyzing symptoms, diseases and recommending complementary medical exams.
       
    """
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages)

    messages.append(response)
    return {"messages": messages}

def manager(state: AgentsState) -> AgentsState:
    system_prompt = """
    You are a medical writing specialist responsible for connecting all data received, without skipping or overwriting any data.
    Structure textually for each disease hypothesis. At the end, add the recommended medical examaminations.

    Structure the response in a medical document format. In the header indicate the pacient's name, age, age category identified and symptoms. 
    Then state the hypotheses collected with each disease symptoms and a brief description of the correlation of the symptoms with the patient's symptoms.
    Add the recommended medical examinations for the pacient. 
    At the end, sign with the phrase: 'Cuide-se! Com carinho, HealthAId.'
    Do it like a medical record.
    Translate all the awnser to portuguese.
    """
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages)

    messages.append(response)
    return {"messages": messages}

def get_name(state: AgentsState) -> list[str]:
    retorno = state["selected_specialty"]
    if retorno == '[]':
        return [END]
    else:
        return retorno

builder = StateGraph(AgentsState)

builder.add_node("supervisor", supervisor)
builder.add_node("specialty_identifier", specialty_identifier)
builder.add_node("respiratorio", respiratorio)
builder.add_node("cardiovascular", cardiovascular)
builder.add_node("urinario", urinario)
builder.add_node("dermatologico", dermatologico)
builder.add_node("hematologico", hematologico)
builder.add_node("digestivo", digestivo)
builder.add_node("neurologico", neurologico)
builder.add_node("infeccioso", infeccioso)
builder.add_node("reumatologico", reumatologico)
builder.add_node("clinico_geral", clinico_geral)
builder.add_node("recommended_medical_examination", recommended_medical_examination)
builder.add_node("manager", manager)

builder.add_edge(START, "supervisor")
builder.add_edge("supervisor", "specialty_identifier")

builder.add_conditional_edges("specialty_identifier", get_name, then="recommended_medical_examination")

builder.add_edge("recommended_medical_examination", "manager")
builder.add_edge("manager", END)

graph = builder.compile()

output_file = "HealthAId.png"
graph_image = graph.get_graph().draw_mermaid_png()

with open(output_file, "wb") as f:
    f.write(graph_image)

print(f"Grafo salvo como {output_file}" + "\n")

def process_input(user_input):
    print(user_input)
    initial_state = AgentsState(
        messages=[HumanMessage(content=user_input)],
        pacient_symptoms=[],
        age_classification="",
        selected_specialty=[],
        specialty=""
    )

    final_message = None
    for event in graph.stream(initial_state):
        for value in event.values():
            if isinstance(value, dict) and 'messages' in value:
                for message in value['messages']:
                    if hasattr(message, 'content'):
                        final_message = message.content
    
    print(final_message)

    if final_message:
        return final_message
    else:
        return "Resposta não encontrada."
