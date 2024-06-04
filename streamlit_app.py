import streamlit as st
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core import PromptTemplate
from llama_index.core import get_response_synthesizer
from llama_index.llms.anthropic import Anthropic
from qdrant_client import QdrantClient

# LOADING SECRETS

COHERE_API_KEY = st.secrets.COHERE_API_KEY
QDRANT_API_KEY = st.secrets.QDRANT_API_KEY
QDRANT_CLUSTER = st.secrets.QDRANT_CLUSTER
QDRANT_COLLECTION = st.secrets.QDRANT_COLLECTION

# CLIENTS

llm = Anthropic(model="claude-3-haiku-20240307")
cohere_rerank = CohereRerank(api_key=COHERE_API_KEY, top_n=2)
cohere_embed_model = CohereEmbedding(
    cohere_api_key=COHERE_API_KEY,
    model_name="embed-multilingual-v3.0",
    input_type="search_document",
)

Settings.llm = llm
Settings.embed_model = cohere_embed_model

# STREAMLIT APP

st.set_page_config(page_title="Ejemplo de asistente RAG", page_icon="💬", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Ejemplo de asistente RAG 💬")
st.info("¡Hola! Soy un asistente virtual que te ayudará a encontrar información dentro de los documentos PDF. ¡Hazme una pregunta!")
         
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        { "role": "assistant", "content": "Hazme una pregunta!" }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Obteniendo índices de los documentos, esta operación puede tardar unos minutos..."):
        qdrant = QdrantClient(
            url=QDRANT_CLUSTER, 
            api_key=QDRANT_API_KEY,
        )
        vector_store = QdrantVectorStore(
            collection_name=QDRANT_COLLECTION,
            client=qdrant,
            enable_hybrid=True,
            batch_size=20,
        )
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        return index

index = load_data()

if "query_engine" not in st.session_state.keys():
    qa_prompt_tmpl = (
        "A continuación se encuentra la información de contexto.\n"
        "-------------------------------"
        "{context_str}\n"
        "-------------------------------"
        "Dada la información de contexto y sin conocimiento previo,"
        "responde a la consulta. Por favor, sé conciso y completo.\n"
        "Si el contexto no contiene una respuesta a la consulta,"
        "responde con \"¡No lo sé!\"."
        "Consulta: {query_str}\n"
        "Respuesta: "
    )
    qa_prompt = PromptTemplate(qa_prompt_tmpl)
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=3,
        sparse_top_k=10,
        vector_store_query_mode="hybrid"
    )

    response_synthesizer = get_response_synthesizer(
        text_qa_template=qa_prompt,
        response_mode="compact",
        streaming=True
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[cohere_rerank],
    )
    st.session_state.query_engine = query_engine

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ingresa tu consulta"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        streaming_response = st.session_state.query_engine.query(prompt)
        response = st.write_stream(streaming_response.response_gen)
    message = { "role": "assistant", "content": response }
    st.session_state.messages.append(message) # Add response to message history