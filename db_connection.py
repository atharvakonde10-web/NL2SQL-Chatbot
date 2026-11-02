# db_connection.py

import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, text

# Load environment variables from .env file
load_dotenv()

def get_db_connection():
    """
    Creates and returns a LangChain SQLDatabase connection object for your MySQL database.
    Reads the connection details securely from your .env file.
    """
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    # ✅ Safety check for missing values
    if not all([db_user, db_password, db_host, db_port, db_name]):
        raise ValueError("❌ Missing one or more required environment variables in .env")

    # MySQL connection URI
    db_uri = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    # LangChain SQLDatabase object
    db = SQLDatabase.from_uri(db_uri)
    return db


if __name__ == "__main__":
    """
    Run this file directly to test the database connection:
    $ python db_connection.py
    """
    print("🔍 Attempting to connect to the database...")

    try:
        # 1. Test SQLAlchemy connection
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")
        db_name = os.getenv("DB_NAME")
        db_uri = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

        engine = create_engine(db_uri)
        with engine.connect() as connection:
            result = connection.execute(text("SELECT NOW();"))
            for row in result:
                print(f"✅ SQLAlchemy connection successful! MySQL Server Time: {row[0]}")

        # 2. Test LangChain SQLDatabase wrapper
        db = get_db_connection()
        print("✅ LangChain SQLDatabase object created successfully!")
        print("📊 Tables found by LangChain:", db.get_usable_table_names())

    except Exception as e:
        print("❌ Database connection failed. Please check your .env file and MySQL server.")
        print(f"Error: {e}")
