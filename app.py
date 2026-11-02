# app.py

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import inspect, text

from agent_setup import setup_agent_executor
from db_connection import get_db_connection
from sql_generator import generate_sql
from sql_safety import validate_sql

load_dotenv()

st.set_page_config(page_title="NL2SQL Chatbot", page_icon="🤖", layout="wide")

MODE_SQL_ONLY = "Generate SQL only"
MODE_FULL = "Generate SQL and run on database"


# ---------- Shared resources ----------

@st.cache_resource
def get_db():
    """One database connection object shared by the whole app."""
    return get_db_connection()


@st.cache_resource
def get_answer_cache():
    """Remembers answers so a repeated question does not use a Gemini request."""
    return {}


@st.cache_resource
def get_llm():
    """Gemini model used by the SQL-only mode (one request per question)."""
    return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"), temperature=0)


def get_engine():
    """Returns the SQLAlchemy engine from LangChain's SQLDatabase object."""
    db = get_db()
    return getattr(db, "_engine", None) or db.engine


# ---------- Helper functions ----------

def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "429" in msg or "resource_exhausted" in low or "quota" in low:
        return ("The Gemini daily free-tier limit has been reached. "
                "Please try again later, when the quota resets.")
    if "1054" in msg or "unknown column" in low:
        return ("That question refers to a column that does not exist. "
                "Check the column names in the sidebar and rephrase your question.")
    if "1146" in msg or "doesn't exist" in low:
        return "That question refers to a table that does not exist in this database."
    if "access denied" in low or "1045" in msg:
        return "The database login failed. Check DB_USER and DB_PASSWORD in your .env file."
    return f"Something went wrong: {msg}"


def to_text(output) -> str:
    """Gemini can return the answer as a list of content blocks; convert to plain text."""
    if isinstance(output, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in output)
    return str(output)


def extract_sql(intermediate_steps):
    """Returns the last SQL query the agent sent to the database, or None."""
    sql = None
    for action, _observation in intermediate_steps:
        if getattr(action, "tool", "") == "sql_db_query":
            tool_input = action.tool_input
            sql = tool_input.get("query") if isinstance(tool_input, dict) else tool_input
    return sql


def fetch_table(sql: str):
    """Validates and runs the SQL to build a result table. Returns (sql, dataframe, warning)."""
    ok, message, safe_sql = validate_sql(sql)
    if not ok:
        return sql, None, f"Query blocked: {message}"
    try:
        with get_engine().connect() as conn:
            df = pd.read_sql(text(safe_sql), conn)
        return safe_sql, df, None
    except Exception as e:
        return safe_sql, None, friendly_error(e)


def ask(question: str, mode: str):
    """Returns (result_dict, answered_from_cache)."""
    cache = get_answer_cache()
    key = f"{mode}|{question.strip().lower()}"
    if key in cache:
        return cache[key], True

    # Mode 1: only write the SQL query. Nothing is run on the database.
    if mode == MODE_SQL_ONLY:
        try:
            sql, warning = generate_sql(question, get_engine(), get_llm())
        except Exception as e:
            return {"content": friendly_error(e), "sql": None, "df": None,
                    "warning": None, "sql_only": True}, False
        result = {"content": "", "sql": sql, "df": None, "warning": warning, "sql_only": True}
        cache[key] = result
        return result, False

    # Mode 2: full agent. Writes the SQL, runs it, and answers in plain English.
    try:
        response = st.session_state.agent_executor.invoke({"input": question})
    except Exception as e:
        return {"content": friendly_error(e), "sql": None, "df": None, "warning": None}, False

    sql = extract_sql(response.get("intermediate_steps", []))
    shown_sql, df, warning = (None, None, None)
    if sql:
        shown_sql, df, warning = fetch_table(sql)

    result = {
        "content": to_text(response["output"]),
        "sql": shown_sql,
        "df": df,
        "warning": warning,
    }
    cache[key] = result
    return result, False


def render_assistant(msg: dict):
    if msg.get("sql"):
        if msg.get("sql_only"):
            st.markdown("**Generated SQL query:**")
            st.code(msg["sql"], language="sql")
        else:
            with st.expander("🧾 Generated SQL"):
                st.code(msg["sql"], language="sql")
    if msg.get("warning"):
        st.warning(msg["warning"])
    df = msg.get("df")
    if df is not None:
        if df.empty:
            st.info("The query ran successfully but returned no rows.")
        else:
            st.dataframe(df, hide_index=True)
    if msg.get("content"):
        st.markdown(msg["content"])


# ---------- Page ----------

st.title("🤖 NL2SQL Chatbot for MySQL")
st.write("Type your question in plain English and get the SQL query for it.")

# Agent is created once per session (used only in the full mode)
if "agent_executor" not in st.session_state:
    with st.spinner("Initializing the AI agent... Please wait."):
        st.session_state.agent_executor = setup_agent_executor()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Mode")
    mode = st.radio(
        "What should the chatbot do?",
        [MODE_SQL_ONLY, MODE_FULL],
        help="'Generate SQL only' writes the query and does NOT run it on the database.",
    )
    if mode == MODE_SQL_ONLY:
        st.caption("The query is only written, never executed. 1 Gemini request per question.")
    else:
        st.caption("The query is written, executed on MySQL (read-only), and explained.")

    st.divider()
    st.header("🗄️ Database")
    st.write(f"**Name:** {os.getenv('DB_NAME', 'ecom')}")
    try:
        inspector = inspect(get_engine())
        for table in inspector.get_table_names():
            with st.expander(f"Table: {table}", expanded=True):
                for col in inspector.get_columns(table):
                    st.write(f"- {col['name']} ({col['type']})")
    except Exception as e:
        st.error(friendly_error(e))
    st.caption("🔒 Read-only: only SELECT queries are allowed.")

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.agent_executor = setup_agent_executor()  # also resets agent memory
        st.rerun()
    if st.button("♻️ Clear answer cache"):
        get_answer_cache().clear()
        st.success("Cache cleared.")

# Chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            render_assistant(m)
        else:
            st.markdown(m["content"])

# New question
if prompt := st.chat_input("Type your question in English..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result, from_cache = ask(prompt, mode)
        render_assistant(result)
        if from_cache:
            st.caption("⚡ Answered from cache (no Gemini request used)")

    st.session_state.messages.append({"role": "assistant", **result})