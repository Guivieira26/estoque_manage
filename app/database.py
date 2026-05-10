
import sqlite3
import os

DB_NAME = 'estoque.db'
# Use o caminho absoluto para evitar criar/abrir bancos em diretórios diferentes
DB_PATH = os.path.abspath(DB_NAME)

def get_db_path():
    return DB_PATH

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
                CREATE TABLE IF NOT EXISTS products (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   name TEXT NOT NULL UNIQUE,
                   quantity INTEGER NOT NULL DEFAULT 0,
                   price REAL NOT NULL DEFAULT 0.0,
                   date_added TEXT NOT NULL
                   )                   
    ''')
    conn.commit()
    conn.close()
