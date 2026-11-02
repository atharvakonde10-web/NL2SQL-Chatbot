# sql_safety.py
# Validates SQL before it is executed. Only single, read-only SELECT queries pass.

import re

MAX_ROWS = 100

# Whole-word matches only, so a column like "created_at" is not blocked.
BLOCKED_WORDS = [
    "drop", "delete", "update", "insert", "alter", "truncate", "create",
    "grant", "revoke", "rename", "call", "load", "outfile", "dumpfile",
]


def _strip_string_literals(sql: str) -> str:
    """Replace quoted text with '' so words inside strings are not checked."""
    return re.sub(r"'(?:[^'\\]|\\.|'')*'|\"(?:[^\"\\]|\\.)*\"", "''", sql)


def validate_sql(sql: str):
    """
    Returns (is_ok, message, safe_sql).
    safe_sql is the cleaned query with a LIMIT added when none was present.
    """
    if not sql or not str(sql).strip():
        return False, "Empty query.", ""

    cleaned = str(sql).strip()
    cleaned = re.sub(r"^```(?:sql)?\s*|\s*```$", "", cleaned, flags=re.I).strip()
    cleaned = cleaned.rstrip(";").strip()

    probe = _strip_string_literals(cleaned).lower()

    if ";" in probe:
        return False, "Multiple SQL statements are not allowed.", ""
    if "--" in probe or "/*" in probe or "#" in probe:
        return False, "SQL comments are not allowed.", ""
    if not re.match(r"\s*select\b", probe):
        return False, "Only SELECT queries are allowed. This chatbot is read-only.", ""
    for word in BLOCKED_WORDS:
        if re.search(rf"\b{word}\b", probe):
            return False, f"'{word.upper()}' is not allowed. This chatbot is read-only.", ""

    if not re.search(r"\blimit\b", probe):
        cleaned = f"{cleaned} LIMIT {MAX_ROWS}"

    return True, "OK", cleaned


if __name__ == "__main__":
    # Offline self-test: no database and no Gemini request needed.
    tests = [
        ("SELECT * FROM orders", True),
        ("select city, count(*) from orders group by city;", True),
        ("SELECT * FROM orders LIMIT 5", True),
        ("SELECT * FROM orders WHERE customer_name = 'drop table'", True),
        ("SELECT order_date FROM orders", True),
        ("DROP TABLE orders", False),
        ("DELETE FROM orders WHERE order_id = 1", False),
        ("UPDATE orders SET price_per_unit = 0", False),
        ("INSERT INTO orders (city) VALUES ('X')", False),
        ("SELECT * FROM orders; DROP TABLE orders", False),
        ("SELECT * FROM orders -- comment", False),
        ("SELECT * FROM orders INTO OUTFILE '/tmp/x.txt'", False),
        ("", False),
    ]
    failed = 0
    for query, expected in tests:
        ok, message, safe = validate_sql(query)
        status = "PASS" if ok == expected else "FAIL"
        failed += status == "FAIL"
        print(f"{status} | allowed={ok!s:5} | {query!r} -> {safe or message}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")