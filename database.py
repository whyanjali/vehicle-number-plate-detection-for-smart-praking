import sqlite3
import os
import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parking_system.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL, -- 'assistant' or 'driver'
        full_name TEXT NOT NULL,
        phone TEXT,
        vehicle_plate TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 2. Parking Slots Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS parking_slots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        slot_number TEXT UNIQUE NOT NULL,
        section TEXT NOT NULL, -- 'Section A (Ground)', 'Section B (Level 1)', 'Section C (EV/Priority)'
        slot_type TEXT NOT NULL DEFAULT 'Standard', -- 'Standard', 'EV', 'Compact'
        is_occupied INTEGER NOT NULL DEFAULT 0, -- 0 for Vacant, 1 for Occupied
        current_plate TEXT,
        vehicle_type TEXT,
        entry_time TIMESTAMP,
        ticket_id TEXT
    )
    """)

    # 3. Parking Records / Tickets Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS parking_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket_id TEXT UNIQUE NOT NULL,
        plate_number TEXT NOT NULL,
        vehicle_type TEXT DEFAULT 'Car',
        slot_number TEXT NOT NULL,
        entry_time TIMESTAMP NOT NULL,
        exit_time TIMESTAMP,
        duration_minutes INTEGER DEFAULT 0,
        hourly_rate REAL DEFAULT 30.0,
        total_fee REAL DEFAULT 0.0,
        status TEXT NOT NULL DEFAULT 'ACTIVE', -- 'ACTIVE', 'COMPLETED'
        created_by TEXT DEFAULT 'assistant'
    )
    """)

    # Seed Default Users if empty
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO users (username, password_hash, role, full_name, phone, vehicle_plate)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            "assistant",
            generate_password_hash("admin123"),
            "assistant",
            "Rajesh Kumar (Parking Attendant)",
            "+91 98765 43210",
            None
        ))

        cursor.execute("""
            INSERT INTO users (username, password_hash, role, full_name, phone, vehicle_plate)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            "driver",
            generate_password_hash("driver123"),
            "driver",
            "Anjali Sharma (Driver)",
            "+91 98123 45678",
            "DL3CAB1234"
        ))

    # Seed Parking Slots if empty
    cursor.execute("SELECT COUNT(*) FROM parking_slots")
    if cursor.fetchone()[0] == 0:
        initial_slots = [
            # Section A (Ground Floor)
            ("A-01", "Section A (Ground)", "Standard"),
            ("A-02", "Section A (Ground)", "Standard"),
            ("A-03", "Section A (Ground)", "Standard"),
            ("A-04", "Section A (Ground)", "Standard"),
            ("A-05", "Section A (Ground)", "Standard"),
            ("A-06", "Section A (Ground)", "Standard"),
            # Section B (Level 1)
            ("B-01", "Section B (Level 1)", "Standard"),
            ("B-02", "Section B (Level 1)", "Standard"),
            ("B-03", "Section B (Level 1)", "Compact"),
            ("B-04", "Section B (Level 1)", "Compact"),
            ("B-05", "Section B (Level 1)", "Standard"),
            ("B-06", "Section B (Level 1)", "Standard"),
            # Section C (EV & Priority)
            ("C-01", "Section C (EV Priority)", "EV"),
            ("C-02", "Section C (EV Priority)", "EV"),
            ("C-03", "Section C (EV Priority)", "Standard"),
            ("C-04", "Section C (EV Priority)", "Standard"),
        ]

        cursor.executemany("""
            INSERT INTO parking_slots (slot_number, section, slot_type, is_occupied)
            VALUES (?, ?, ?, 0)
        """, initial_slots)

        # Pre-seed 2 sample occupied slots for realistic dashboard view
        now = datetime.datetime.now()
        t1 = (now - datetime.timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M:%S")
        t2 = (now - datetime.timedelta(minutes=110)).strftime("%Y-%m-%d %H:%M:%S")

        ticket1 = f"TKT-{int(now.timestamp()) - 2700}"
        ticket2 = f"TKT-{int(now.timestamp()) - 6600}"

        cursor.execute("""
            UPDATE parking_slots
            SET is_occupied = 1, current_plate = 'MH12AB1234', vehicle_type = 'Car', entry_time = ?, ticket_id = ?
            WHERE slot_number = 'A-01'
        """, (t1, ticket1))

        cursor.execute("""
            INSERT INTO parking_records (ticket_id, plate_number, vehicle_type, slot_number, entry_time, status, created_by)
            VALUES (?, 'MH12AB1234', 'Car', 'A-01', ?, 'ACTIVE', 'assistant')
        """, (ticket1, t1))

        cursor.execute("""
            UPDATE parking_slots
            SET is_occupied = 1, current_plate = 'DL8CAF5021', vehicle_type = 'SUV', entry_time = ?, ticket_id = ?
            WHERE slot_number = 'B-02'
        """, (t2, ticket2))

        cursor.execute("""
            INSERT INTO parking_records (ticket_id, plate_number, vehicle_type, slot_number, entry_time, status, created_by)
            VALUES (?, 'DL8CAF5021', 'SUV', 'B-02', ?, 'ACTIVE', 'assistant')
        """, (ticket2, t2))

    conn.commit()
    conn.close()

def authenticate_user(username, password):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if user and check_password_hash(user["password_hash"], password):
        return dict(user)
    return None

def get_user_by_username(username):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return dict(user) if user else None

def get_all_slots():
    conn = get_db_connection()
    slots = conn.execute("SELECT * FROM parking_slots ORDER BY slot_number ASC").fetchall()
    conn.close()
    return [dict(s) for s in slots]

def get_vacant_slots(slot_type=None):
    conn = get_db_connection()
    if slot_type:
        slots = conn.execute(
            "SELECT * FROM parking_slots WHERE is_occupied = 0 AND slot_type = ? ORDER BY slot_number ASC",
            (slot_type,)
        ).fetchall()
    else:
        slots = conn.execute(
            "SELECT * FROM parking_slots WHERE is_occupied = 0 ORDER BY slot_number ASC"
        ).fetchall()
    conn.close()
    return [dict(s) for s in slots]

def find_optimal_vacant_slot(preferred_type='Standard'):
    vacant = get_vacant_slots(preferred_type)
    if vacant:
        return vacant[0]
    all_vacant = get_vacant_slots()
    if all_vacant:
        return all_vacant[0]
    return None

def allot_slot(plate_number, slot_number=None, vehicle_type="Car", created_by="assistant"):
    plate_number = plate_number.strip().upper()
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if vehicle is already parked
    existing = cursor.execute(
        "SELECT * FROM parking_slots WHERE current_plate = ? AND is_occupied = 1",
        (plate_number,)
    ).fetchone()
    if existing:
        conn.close()
        return {
            "success": False,
            "message": f"Vehicle {plate_number} is already parked in Slot {existing['slot_number']}!"
        }

    # If slot_number not provided, find optimal vacant slot
    if not slot_number:
        opt = cursor.execute(
            "SELECT * FROM parking_slots WHERE is_occupied = 0 ORDER BY slot_number ASC LIMIT 1"
        ).fetchone()
        if not opt:
            conn.close()
            return {"success": False, "message": "Sorry, parking is currently full!"}
        target_slot = opt["slot_number"]
    else:
        target_slot = slot_number.strip().upper()
        slot = cursor.execute(
            "SELECT * FROM parking_slots WHERE slot_number = ?", (target_slot,)
        ).fetchone()
        if not slot:
            conn.close()
            return {"success": False, "message": f"Slot {target_slot} does not exist."}
        if slot["is_occupied"] == 1:
            conn.close()
            return {"success": False, "message": f"Slot {target_slot} is already occupied by {slot['current_plate']}."}

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ticket_id = f"TKT-{int(datetime.datetime.now().timestamp())}"

    # Update slot
    cursor.execute("""
        UPDATE parking_slots
        SET is_occupied = 1, current_plate = ?, vehicle_type = ?, entry_time = ?, ticket_id = ?
        WHERE slot_number = ?
    """, (plate_number, vehicle_type, now_str, ticket_id, target_slot))

    # Insert record
    cursor.execute("""
        INSERT INTO parking_records (ticket_id, plate_number, vehicle_type, slot_number, entry_time, status, created_by)
        VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?)
    """, (ticket_id, plate_number, vehicle_type, target_slot, now_str, created_by))

    conn.commit()
    conn.close()

    return {
        "success": True,
        "message": f"Slot {target_slot} successfully allotted to {plate_number}!",
        "ticket": {
            "ticket_id": ticket_id,
            "plate_number": plate_number,
            "slot_number": target_slot,
            "vehicle_type": vehicle_type,
            "entry_time": now_str,
            "hourly_rate": 30.0
        }
    }

def release_slot(slot_number_or_plate):
    identifier = slot_number_or_plate.strip().upper()
    conn = get_db_connection()
    cursor = conn.cursor()

    slot = cursor.execute("""
        SELECT * FROM parking_slots 
        WHERE (slot_number = ? OR current_plate = ?) AND is_occupied = 1
    """, (identifier, identifier)).fetchone()

    if not slot:
        conn.close()
        return {"success": False, "message": f"No active parked vehicle found for '{identifier}'."}

    slot_dict = dict(slot)
    slot_num = slot_dict["slot_number"]
    plate_num = slot_dict["current_plate"]
    entry_time_str = slot_dict["entry_time"]
    ticket_id = slot_dict["ticket_id"]

    now = datetime.datetime.now()
    exit_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Calculate duration and fee
    try:
        entry_dt = datetime.datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
        duration_minutes = max(1, int((now - entry_dt).total_seconds() / 60))
    except Exception:
        duration_minutes = 15

    # Fee: ₹30 per hour (minimum 1 hour charge)
    hourly_rate = 30.0
    hours = max(1, (duration_minutes + 59) // 60)
    total_fee = float(hours * hourly_rate)

    # Vacate the slot
    cursor.execute("""
        UPDATE parking_slots
        SET is_occupied = 0, current_plate = NULL, vehicle_type = NULL, entry_time = NULL, ticket_id = NULL
        WHERE slot_number = ?
    """, (slot_num,))

    # Update record
    if ticket_id:
        cursor.execute("""
            UPDATE parking_records
            SET exit_time = ?, duration_minutes = ?, total_fee = ?, status = 'COMPLETED'
            WHERE ticket_id = ?
        """, (exit_time_str, duration_minutes, total_fee, ticket_id))
    else:
        cursor.execute("""
            UPDATE parking_records
            SET exit_time = ?, duration_minutes = ?, total_fee = ?, status = 'COMPLETED'
            WHERE slot_number = ? AND plate_number = ? AND status = 'ACTIVE'
        """, (exit_time_str, duration_minutes, total_fee, slot_num, plate_num))

    conn.commit()
    conn.close()

    return {
        "success": True,
        "message": f"Vehicle {plate_num} checked out from Slot {slot_num}!",
        "receipt": {
            "ticket_id": ticket_id or "N/A",
            "plate_number": plate_num,
            "slot_number": slot_num,
            "vehicle_type": slot_dict.get("vehicle_type", "Car"),
            "entry_time": entry_time_str,
            "exit_time": exit_time_str,
            "duration_minutes": duration_minutes,
            "hours_billed": hours,
            "hourly_rate": hourly_rate,
            "total_fee": total_fee
        }
    }

def get_active_ticket_for_plate(plate_number):
    clean_plate = plate_number.strip().upper()
    conn = get_db_connection()
    slot = conn.execute("""
        SELECT * FROM parking_slots 
        WHERE current_plate = ? AND is_occupied = 1
    """, (clean_plate,)).fetchone()
    conn.close()
    if not slot:
        return None

    slot_dict = dict(slot)
    now = datetime.datetime.now()
    try:
        entry_dt = datetime.datetime.strptime(slot_dict["entry_time"], "%Y-%m-%d %H:%M:%S")
        duration_minutes = max(1, int((now - entry_dt).total_seconds() / 60))
    except Exception:
        duration_minutes = 0

    hours = max(1, (duration_minutes + 59) // 60)
    estimated_fee = hours * 30.0

    return {
        "ticket_id": slot_dict.get("ticket_id") or "ACTIVE",
        "plate_number": slot_dict["current_plate"],
        "slot_number": slot_dict["slot_number"],
        "section": slot_dict["section"],
        "slot_type": slot_dict["slot_type"],
        "vehicle_type": slot_dict.get("vehicle_type", "Car"),
        "entry_time": slot_dict["entry_time"],
        "duration_minutes": duration_minutes,
        "estimated_fee": estimated_fee
    }

def get_system_stats():
    conn = get_db_connection()
    total_slots = conn.execute("SELECT COUNT(*) FROM parking_slots").fetchone()[0]
    occupied_slots = conn.execute("SELECT COUNT(*) FROM parking_slots WHERE is_occupied = 1").fetchone()[0]
    vacant_slots = total_slots - occupied_slots

    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    today_records = conn.execute("""
        SELECT COUNT(*), COALESCE(SUM(total_fee), 0)
        FROM parking_records
        WHERE entry_time LIKE ? AND status = 'COMPLETED'
    """, (f"{today_str}%",)).fetchone()

    today_checkouts = today_records[0]
    today_revenue = today_records[1]

    active_vehicles = conn.execute("""
        SELECT slot_number, current_plate, vehicle_type, entry_time, ticket_id
        FROM parking_slots
        WHERE is_occupied = 1
        ORDER BY entry_time DESC
    """).fetchall()

    conn.close()

    return {
        "total_slots": total_slots,
        "occupied_slots": occupied_slots,
        "vacant_slots": vacant_slots,
        "occupancy_rate": round((occupied_slots / total_slots * 100) if total_slots else 0, 1),
        "today_checkouts": today_checkouts,
        "today_revenue": round(today_revenue, 2),
        "active_vehicles": [dict(v) for v in active_vehicles]
    }
