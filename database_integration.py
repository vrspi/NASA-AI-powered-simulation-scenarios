import sqlite3

def create_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS fits_data
                      (filename TEXT, date_obs TEXT, exptime REAL, brightness REAL)''')
    conn.commit()
    conn.close()

def insert_data(db_path, data):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO fits_data VALUES (?, ?, ?, ?)", data)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    db_path = "neossat_data.db"
    create_database(db_path)
    # Insert data as needed