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

    cur.execute("SELECT 1 FROM pg_database WHERE datname=postgres")
    exists = cur.fetchone()
    if not exists:
        cur.execute('CREATE DATABASE postgres')
    else:
        print("Database 'yourdbname' already exists")

    cur.close()
    conn.close()
    print("Database created")
if __name__ == '__main__':
    create_database()