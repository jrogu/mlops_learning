import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
from dotenv import load_dotenv

load_dotenv()


def connect():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
    )


def add_row(prediction, probability):
    conn = connect()

    cur = conn.cursor()

    cur.execute(
        """ INSERT INTO results 
                (prediction, probability) 
                VALUES (%s, %s)""",
        (prediction, probability),
    )
    conn.commit()
    cur.close()
    conn.close()


def create_database():
    conn = connect()

    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname='postgres'")
    exists = cur.fetchone()
    if not exists:
        cur.execute("CREATE DATABASE postgres")
    else:
        print("Database 'postgres' already exists")

    cur.close()
    conn.close()
    print("Database created")


def create_table_and_insert_rows():
    conn = connect()

    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            id SERIAL PRIMARY KEY,
            prediction VARCHAR(100),
            probability FLOAT
        )
    """
    )

    conn.commit()

    cur.close()
    conn.close()
    print("Table created and rows inserted")


if __name__ == "__main__":

    create_database()
    create_table_and_insert_rows()
