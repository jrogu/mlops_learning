import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def create_database():
    conn = psycopg2.connect(
        dbname='postgres',
        user='postgres',
        password='passwrd',
        host='localhost',  
        port='5432'
    )

    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    cur = conn.cursor()

    cur.execute('CREATE DATABASE db')

    # Close the connection
    cur.close()
    conn.close()

if __name__ == '__main__':
    create_database()