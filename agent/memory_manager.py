# agent/memory_manager.py

import psycopg2
from langchain.memory import ConversationBufferMemory
from typing import Optional

class MemoryManager:
    def __init__(
        self,
        pg_host: str = "localhost",
        pg_port: int = 5432,
        pg_database: str = "mydb",
        pg_user: str = "myuser",
        pg_password: str = "mypassword"
    ):
        """
        Initializes short-term memory and a connection to Postgres for long-term memory & feedback.
        """
        self.short_term_memory = ConversationBufferMemory(memory_key="chat_history")
        self.conn = psycopg2.connect(
            host=pg_host,
            port=pg_port,
            database=pg_database,
            user=pg_user,
            password=pg_password
        )
        self.conn.autocommit = True

    def get_short_term_memory(self) -> ConversationBufferMemory:
        return self.short_term_memory

    def add_long_term_memory(self, key: str, value: str):
        """
        Store or update a memory entry in Postgres.
        """
        with self.conn.cursor() as cur:
            sql = """
                INSERT INTO long_term_memory (memory_key, memory_value)
                VALUES (%s, %s)
                ON CONFLICT (memory_key)
                DO UPDATE SET memory_value = EXCLUDED.memory_value;
            """
            cur.execute(sql, (key, value))

    def retrieve_long_term_memory(self, key: str) -> Optional[str]:
        """
        Retrieve a memory entry by key.
        """
        with self.conn.cursor() as cur:
            sql = "SELECT memory_value FROM long_term_memory WHERE memory_key = %s;"
            cur.execute(sql, (key,))
            row = cur.fetchone()
            return row[0] if row else None

    def store_feedback(self, user_query: str, feedback_text: str):
        """
        Store user feedback in Postgres.
        """
        with self.conn.cursor() as cur:
            sql = "INSERT INTO feedback (user_query, feedback_text) VALUES (%s, %s);"
            cur.execute(sql, (user_query, feedback_text))
