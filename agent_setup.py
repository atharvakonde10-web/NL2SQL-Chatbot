# agent_setup.py

import os  # NEW

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent

from langchain.memory import ConversationBufferMemory

from db_connection import get_db_connection
from sql_safety import validate_sql


def _guard_database(db):
    """
    Every query the agent sends to MySQL goes through validate_sql first.
    Blocked queries are never executed; the agent receives an error message instead.
    """
    original_run = db.run

    def guarded_run(command, *args, **kwargs):
        ok, message, safe_sql = validate_sql(command)
        if not ok:
            return f"Error: query blocked. {message}"
        return original_run(safe_sql, *args, **kwargs)

    db.run = guarded_run


def setup_agent_executor():
    """
    Sets up the SQL agent with Gemini, MySQL database,
    conversation memory, and error handling.
    Returns the agent executor.
    """

    # 1. Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),  # CHANGED: model name comes from .env
        temperature=0
    )

    # 2. Connect to MySQL database
    db = get_db_connection()
    _guard_database(db)

    # 3. Conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output"  # needed because the agent returns 2 outputs
    )

    # 4. SQL toolkit
    toolkit = SQLDatabaseToolkit(
        db=db,
        llm=llm
    )

    # 5. Create SQL agent
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        agent_executor_kwargs={"return_intermediate_steps": True},  # lets the UI read the SQL
    )

    return agent_executor


# Test the agent when this file is run directly
if __name__ == "__main__":

    agent_executor = setup_agent_executor()

    questions = [
        "Which city has the most orders?",
    ]

    for q in questions:
        print(f"\nQ: {q}")

        try:
            response = agent_executor.invoke({"input": q})
            print("A:", response["output"])

        except Exception as e:
            print("Error:", e)