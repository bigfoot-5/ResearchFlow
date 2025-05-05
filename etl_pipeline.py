import sqlite3
from datetime import datetime

# Your extracted HubSpot data
contacts = [{'id': '271585531090', 'properties': {'createdate': '2025-04-26T10:11:40.204Z', 'email': 'bh@hubspot.com', 'firstname': 'Brian', 'hs_object_id': '271585531090', 'lastmodifieddate': '2025-04-26T10:13:06.914Z', 'lastname': 'Halligan (Sample Contact)'}, 'createdAt': '2025-04-26T10:11:40.204Z', 'updatedAt': '2025-04-26T10:13:06.914Z', 'archived': False}, {'id': '271637752006', 'properties': {'createdate': '2025-04-26T10:11:39.841Z', 'email': 'emailmaria@hubspot.com', 'firstname': 'Maria', 'hs_object_id': '271637752006', 'lastmodifieddate': '2025-04-26T10:11:52.436Z', 'lastname': 'Johnson (Sample Contact)'}, 'createdAt': '2025-04-26T10:11:39.841Z', 'updatedAt': '2025-04-26T10:11:52.436Z', 'archived': False}, {'id': '272110637250', 'properties': {'createdate': '2025-04-28T09:10:53.547Z', 'email': 'vincenzomigliore@hotmail.com', 'firstname': 'vincenzo', 'hs_object_id': '272110637250', 'lastmodifieddate': '2025-04-28T13:00:24.766Z', 'lastname': 'migliore'}, 'createdAt': '2025-04-28T09:10:53.547Z', 'updatedAt': '2025-04-28T13:00:24.766Z', 'archived': False}]  # paste your contacts dict here
companies = [{'id': '138616529118', 'properties': {'createdate': '2025-04-26T10:11:40.378Z', 'domain': 'hubspot.com', 'hs_lastmodifieddate': '2025-04-26T10:11:56.309Z', 'hs_object_id': '138616529118', 'name': 'HubSpot'}, 'createdAt': '2025-04-26T10:11:40.378Z', 'updatedAt': '2025-04-26T10:11:56.309Z', 'archived': False}]  # paste your companies dict here
deals =  [{'id': '189331863792', 'properties': {'amount': '1000.0', 'closedate': '2024-01-03T09:37:11.118Z', 'createdate': '2025-04-26T10:13:04.516Z', 'dealname': 'HubSpot - New Deal (Sample Deal)', 'dealstage': 'appointmentscheduled', 'hs_lastmodifieddate': '2025-04-26T10:13:06.236Z', 'hs_object_id': '189331863792', 'pipeline': 'default'}, 'createdAt': '2025-04-26T10:13:04.516Z', 'updatedAt': '2025-04-26T10:13:06.236Z', 'archived': False}] # paste your deals dict here

def flatten_contact(contact):
    props = contact["properties"]
    return (
        contact["id"],
        props.get("firstname", ""),
        props.get("lastname", ""),
        props.get("email", ""),
        props.get("createdate", ""),
        props.get("lastmodifieddate", ""),
    )

def flatten_company(company):
    props = company["properties"]
    return (
        company["id"],
        props.get("name", ""),
        props.get("domain", ""),
        props.get("createdate", ""),
    )

def flatten_deal(deal):
    props = deal["properties"]
    return (
        deal["id"],
        props.get("dealname", ""),
        float(props.get("amount", 0.0)),
        props.get("dealstage", ""),
        props.get("pipeline", ""),
        props.get("closedate", ""),
        props.get("createdate", "")
    )

def create_tables(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id TEXT PRIMARY KEY,
            firstname TEXT,
            lastname TEXT,
            email TEXT,
            createdate TEXT,
            lastmodifieddate TEXT
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id TEXT PRIMARY KEY,
            name TEXT,
            domain TEXT,
            createdate TEXT
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS deals (
            id TEXT PRIMARY KEY,
            dealname TEXT,
            amount REAL,
            dealstage TEXT,
            pipeline TEXT,
            closedate TEXT,
            createdate TEXT
        );
    """)
    conn.commit()

def load_data():
    conn = sqlite3.connect("hubspot_etl.db")
    create_tables(conn)

    cur = conn.cursor()
    for c in contacts:
        cur.execute("INSERT OR REPLACE INTO contacts VALUES (?, ?, ?, ?, ?, ?)", flatten_contact(c))
    for comp in companies:
        cur.execute("INSERT OR REPLACE INTO companies VALUES (?, ?, ?, ?)", flatten_company(comp))
    for d in deals:
        cur.execute("INSERT OR REPLACE INTO deals VALUES (?, ?, ?, ?, ?, ?, ?)", flatten_deal(d))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    load_data()
    print("✅ ETL complete. Data loaded into SQLite.")