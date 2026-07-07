import sqlite3
import pandas
import numpy as np

conn = sqlite3.connect("dataset/reddit-comments-may-2015/database.sqlite")
cursor = conn.cursor()

df = pandas.read_sql_query("SELECT * FROM May2015 LIMIT 20000", conn)
print(df.head())

cursor.close()
conn.close()
