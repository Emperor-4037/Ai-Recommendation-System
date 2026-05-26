import sqlite3

db_path = "d:/Ai Recommendation System/hyperopt_study.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute("SELECT number, state FROM trials")
rows = c.fetchall()
print(f"Current states: {rows}")

# Optuna sometimes uses 'RUNNING' as a string in the DB rather than 0
c.execute("UPDATE trials SET state = 'FAIL' WHERE state = 'RUNNING'")
rowcount = c.rowcount
print(f"Updated {rowcount} zombie trials (string format).")

c.execute("SELECT number, state FROM trials")
print(f"States after update: {c.fetchall()}")

conn.commit()
conn.close()
