import sqlite3
import pandas as pd
import numpy as np

# DB config
conn = sqlite3.connect("dataset/reddit-comments-may-2015/database.sqlite")
cursor = conn.cursor()

# Array with random table rows index to have randomness in data
random_array = np.random.choice(range(54504410), 20000, replace=False)
array_to_list = ",".join(map(str, random_array))

# Dataset with random indexes
df = pd.read_sql_query(f"""SELECT * 
                              FROM May2015 
                              WHERE rowid IN ({array_to_list});
                           """, conn)

# Filtering removed and deleted comments
filter_deleted = df[(df['body'] != "[deleted]") & (df['body'] != '[removed]')]

# Helper column with words count in 'body'
filter_deleted["words_count"] = filter_deleted['body'].apply(lambda row: len(row.split()))

# Filtering comments that are less than 10 percentile at words count
filter_short_comms = filter_deleted[filter_deleted['words_count'] >= filter_deleted['words_count'].quantile(0.10)]

# Removing bot comments
## Saving repeat count of 'body' value in new columns
filter_short_comms['repeat_count'] = filter_short_comms['body'].map(filter_short_comms['body'].value_counts())
## Filtering all comments that have more than 15 words and repeats more than once
filter_bots = filter_short_comms[~((filter_short_comms['repeat_count'] > 1) & (filter_short_comms['words_count'] > 15))]

print("Number of rows before removing bots:", len(filter_short_comms))
print("Number of rows after removing bots:", len(filter_bots))

cursor.close()
conn.close()
