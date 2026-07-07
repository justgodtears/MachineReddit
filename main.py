import sqlite3
import pandas
import numpy as np

conn = sqlite3.connect("dataset/reddit-comments-may-2015/database.sqlite")
cursor = conn.cursor()

random_array = np.random.choice(range(54504410), 20000, replace=False)

array_to_list = ",".join(map(str, random_array))

df = pandas.read_sql_query(f"""SELECT * 
                              FROM May2015 
                              WHERE rowid IN ({array_to_list});
                           """, conn)


print(df.head())

cursor.close()
conn.close()
