# sql_generator.py
# English question -> SQL query text only. The query is NOT run on the database.
# It reads only the table structure (table and column names), never the data rows.

import re

from sqlalchemy import inspect

from sql_safety import validate_sql

PROMPT = """You are a MySQL expert. Convert the user's question into ONE MySQL SELECT query.

Database schema:
{schema}

Rules:
- Return ONLY the SQL query. No explanation, no markdown, no comments.
- Use only the tables and columns listed in the schema above.
- Write a read-only SELECT query. Never use DROP, DELETE, UPDATE, INSERT or ALTER.
- If the question cannot be answered with this schema, return exactly: CANNOT_ANSWER

Question: {question}
SQL:"""


def get_schema_text(engine) -> str:
    """Table and column names with types (structure only, no data rows)."""
    inspector = inspect(engine)
    lines = []
    for table in inspector.get_table_names():
        cols = ", ".join(f"{c['name']} {c['type']}" for c in inspector.get_columns(table))
        lines.append(f"Table {table}: {cols}")
    return "\n".join(lines)


def _to_text(content) -> str:
    """Gemini may return a list of content blocks; convert to plain text."""
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def _clean(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text, flags=re.I).strip()
    return text


def generate_sql(question: str, engine, llm):
    """
    Returns (sql, warning). Uses exactly one LLM request.
    sql is None when nothing safe could be produced; warning explains why.
    """
    prompt = PROMPT.format(schema=get_schema_text(engine), question=question.strip())
    response = llm.invoke(prompt)
    sql = _clean(_to_text(response.content))

    if not sql or "CANNOT_ANSWER" in sql.upper():
        return None, "This question cannot be answered with the tables and columns in this database."

    ok, message, _ = validate_sql(sql)
    if not ok:
        return None, f"Query blocked: {message}"

    return sql.rstrip(";").strip() + ";", None


if __name__ == "__main__":
    # Offline test with a fake LLM and an in-memory SQLite table (no Gemini, no MySQL).
    from sqlalchemy import create_engine, text

    class FakeResponse:
        def __init__(self, content):
            self.content = content

    class FakeLLM:
        def __init__(self, answer):
            self.answer = answer

        def invoke(self, prompt):
            assert "Table orders:" in prompt and "order_id" in prompt
            return FakeResponse(self.answer)

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE orders (order_id INTEGER, city VARCHAR(50))"))

    cases = [
        ("How many orders?", "```sql\nSELECT COUNT(*) FROM orders\n```", True),
        ("Delete everything", "DELETE FROM orders", False),
        ("Show salary", "CANNOT_ANSWER", False),
        ("List blocks", [{"type": "text", "text": "SELECT city FROM orders;"}], True),
    ]
    passed = 0
    for question, answer, expect_sql in cases:
        sql, warning = generate_sql(question, engine, FakeLLM(answer))
        good = (sql is not None) == expect_sql
        passed += good
        print(f"{'PASS' if good else 'FAIL'} | {question!r} -> {sql or warning}")
    print(f"\n{passed}/{len(cases)} tests passed")