import streamlit as st
import config
import os
from rag_engine import RAGPipeline
from guardrails import HealthGuardrail
from evaluator import RagasEvaluator
from langchain_core.messages import HumanMessage, AIMessage

# --- UI Setup ---
st.set_page_config(page_title="MediGuard RAG Bot", layout="wide")
st.title("🏥 MediGuard: Conversational RAG Bot")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Google API Key", type="password")
    guardrail_enabled = st.toggle("Enable Guardrails", value=True)
    
    st.markdown("---")
    st.header("Knowledge Base")
    uploaded_file = st.file_uploader("Upload Medical PDF", type="pdf")
    
    st.markdown("---")
    st.header("Evaluation")
    if st.button("📊 Evaluate Last Response"):
        run_evaluation = True
    else:
        run_evaluation = False

    if not api_key:
        st.warning("Please enter API Key")
        st.stop()

# --- Initialization ---
config.setup_env(api_key)

@st.cache_resource(show_spinner=False)
def load_system(key):
    rag = RAGPipeline(key)
    guard = HealthGuardrail(rag.llm)
    evaluator = RagasEvaluator(rag)
    return rag, guard, evaluator

rag_system, guard_system, eval_system = load_system(api_key)

# --- File Upload Logic ---
if uploaded_file:
    temp_path = "temp_source.pdf"
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            rag_system.rebuild_index(temp_path)
            st.session_state.last_uploaded = uploaded_file.name
            st.success("✅ Knowledge base updated!")
            st.session_state.messages = [{"role": "assistant", "content": f"I have read {uploaded_file.name}. Ask me questions!"}]

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me about medical topics."}]

# --- Helper: Get Chat History for LangChain ---
def get_chat_history():
    history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history

# --- Chat Loop ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Display metrics if they were saved with this specific message
        if "metrics" in msg:
            with st.expander("📊 Evaluation Score"):
                m = msg["metrics"]
                c1, c2 = st.columns(2)
                c1.metric("Faithfulness", f"{m['faithfulness']:.2f}")
                c2.metric("Relevance", f"{m['answer_relevancy']:.2f}")

# --- Input Logic ---
if prompt := st.chat_input("Ask a follow-up question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 1. Input Guardrail
    is_safe = True
    if guardrail_enabled:
        with st.spinner("Checking safety..."):
            is_safe = guard_system.check_input_safety(prompt)

    if not is_safe:
        error_msg = "🚫 Guardrail Block: Please ask health-related questions only."
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        with st.chat_message("assistant"):
            st.error(error_msg)
    else:
        # 2. Conversational RAG
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = rag_system.get_conversational_chain()
                
                # Convert history for the model
                chat_history = get_chat_history()
                
                # Run the chain (input + history)
                result = chain.invoke({
                    "input": prompt,
                    "chat_history": chat_history
                })
                
                response_text = result["answer"]
                
                # 3. Output Guardrail
                if guardrail_enabled:
                    if not guard_system.check_output_safety(response_text):
                        response_text = "⚠️ Safety Alert: Response flagged as potentially unsafe."

                st.markdown(response_text)
                
                # Save context for potential evaluation later
                # We extract the retrieved docs from the result
                retrieved_docs = [doc.page_content for doc in result.get("context", [])]
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "context_snapshot": retrieved_docs # Save this invisibly for evaluation
                })

# --- Evaluation Logic (Triggered by Button) ---
if run_evaluation:
    # Check if we have enough history to evaluate
    if len(st.session_state.messages) >= 2:
        last_ai_msg = st.session_state.messages[-1]
        last_user_msg = st.session_state.messages[-2]
        
        if last_ai_msg["role"] == "assistant" and "context_snapshot" in last_ai_msg:
            with st.spinner("📊 Running Ragas Evaluation on last response..."):
                metrics = eval_system.evaluate_response(
                    query=last_user_msg["content"],
                    response=last_ai_msg["content"],
                    context_list=last_ai_msg["context_snapshot"]
                )
                
                # Save metrics to the message so they persist
                st.session_state.messages[-1]["metrics"] = metrics
                st.rerun() # Rerun to show the metrics in the chat loop
        else:
            st.warning("No recent assistant response to evaluate.")
    else:
        st.warning("Start a conversation first!")