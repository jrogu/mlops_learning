import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def create_database():
    conn = psycopg2.connect(
        dbname='postgres',
        user='postgres',
        password='passwrd',
        host='db',  
        port='5432'
    )

    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname='postgres'")
    exists = cur.fetchone()
    if not exists:
        cur.execute('CREATE DATABASE postgres')
    else:
        print("Database 'postgres' already exists")

    cur.close()
    conn.close()
    print("Database created")

def create_table_and_insert_rows():
    conn = psycopg2.connect(
        dbname='postgres',
        user='postgres',
        password='passwrd',
        host='db',  
        port='5432'
    )

    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS test (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            age INT
        )
    """)

    cur.execute("INSERT INTO test (name, age) VALUES (%s, %s)", ("Alice", 30))
    cur.execute("INSERT INTO test (name, age) VALUES (%s, %s)", ("Bob", 25))

    conn.commit()

    cur.close()
    conn.close()
    print("Table created and rows inserted")

if __name__ == '__main__':
    create_database()
    create_table_and_insert_rows()
