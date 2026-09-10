import os
import base64
from functools import wraps
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import database
import alpr_service

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

# Ensure database is initialized on startup
database.init_db()

# Decorators for route protection
def login_required(role=None):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'username' not in session:
                return redirect(url_for('login', next=request.url))
            if role and session.get('role') != role:
                if session.get('role') == 'assistant':
                    return redirect(url_for('assistant_dashboard'))
                else:
                    return redirect(url_for('driver_dashboard'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.context_processor
def inject_user():
    user = None
    if 'username' in session:
        user = {
            'username': session.get('username'),
            'role': session.get('role'),
            'full_name': session.get('full_name'),
            'vehicle_plate': session.get('vehicle_plate')
        }
    return dict(user=user)

# --- WEB PAGE ROUTES ---

@app.route('/')
def index():
    if 'username' in session:
        if session.get('role') == 'assistant':
            return redirect(url_for('assistant_dashboard'))
        return redirect(url_for('driver_dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        if 'username' in session:
            return redirect(url_for('index'))
        return render_template('login.html', error=None, role='assistant')

    # POST handling
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()
    selected_role = request.form.get('role', 'assistant')

    # Check database
    user = database.authenticate_user(username, password)
    
    # If user entered plate number directly for driver portal
    if not user and selected_role == 'driver':
        # Auto-create or login guest driver with plate
        clean_plate = username.upper()
        if len(clean_plate) >= 4:
            user = {
                'username': clean_plate,
                'role': 'driver',
                'full_name': f"Driver ({clean_plate})",
                'vehicle_plate': clean_plate
            }

    if user:
        session['username'] = user['username']
        session['role'] = user['role']
        session['full_name'] = user['full_name']
        session['vehicle_plate'] = user.get('vehicle_plate')

        if user['role'] == 'assistant':
            return redirect(url_for('assistant_dashboard'))
        return redirect(url_for('driver_dashboard'))

    return render_template('login.html', error="Invalid credentials. Please try again.", role=selected_role)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/assistant')
@login_required(role='assistant')
def assistant_dashboard():
    return render_template('assistant_dashboard.html')

@app.route('/driver')
@login_required(role='driver')
def driver_dashboard():
    return render_template('driver_dashboard.html')

# --- JSON REST APIs ---

@app.route('/api/current_user')
def api_current_user():
    if 'username' in session:
        return jsonify({
            'authenticated': True,
            'user': {
                'username': session.get('username'),
                'role': session.get('role'),
                'full_name': session.get('full_name'),
                'vehicle_plate': session.get('vehicle_plate')
            }
        })
    return jsonify({'authenticated': False})

@app.route('/api/slots', methods=['GET'])
def api_slots():
    try:
        slots = database.get_all_slots()
        return jsonify({'success': True, 'slots': slots})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def api_stats():
    try:
        stats = database.get_system_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/sample_images', methods=['GET'])
def api_sample_images():
    try:
        samples = alpr_service.get_sample_images()
        return jsonify({'success': True, 'samples': samples})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/capture_plate', methods=['POST'])
def api_capture_plate():
    try:
        data = request.get_json() or {}
        image_input = None

        if 'image_b64' in data and data['image_b64']:
            raw_b64 = data['image_b64']
            if ',' in raw_b64:
                raw_b64 = raw_b64.split(',', 1)[1]
            image_input = base64.b64decode(raw_b64)
        elif 'sample_name' in data and data['sample_name']:
            sample_path = alpr_service.get_sample_image_path(data['sample_name'])
            if not sample_path:
                return jsonify({'success': False, 'message': 'Sample image not found.'}), 404
            image_input = sample_path
        else:
            return jsonify({'success': False, 'message': 'No image data provided.'}), 400

        # Run ALPR Recognition
        result = alpr_service.recognize_plate(image_input)
        return jsonify({'success': True, 'result': result})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/allot_slot', methods=['POST'])
def api_allot_slot():
    try:
        data = request.get_json() or {}
        plate = data.get('plate_number', '').strip()
        slot = data.get('slot_number', None)
        v_type = data.get('vehicle_type', 'Car')
        created_by = session.get('role', 'assistant')

        if not plate:
            return jsonify({'success': False, 'message': 'Plate number is required.'}), 400

        res = database.allot_slot(
            plate_number=plate,
            slot_number=slot,
            vehicle_type=v_type,
            created_by=created_by
        )
        return jsonify(res)

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/release_slot', methods=['POST'])
def api_release_slot():
    try:
        data = request.get_json() or {}
        identifier = data.get('identifier', '').strip()

        if not identifier:
            return jsonify({'success': False, 'message': 'Slot or plate identifier required.'}), 400

        res = database.release_slot(identifier)
        return jsonify(res)

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/driver_ticket/<path:plate>')
def api_driver_ticket(plate):
    try:
        ticket = database.get_active_ticket_for_plate(plate)
        if ticket:
            return jsonify({'found': True, 'ticket': ticket})
        return jsonify({'found': False, 'message': 'No active parked vehicle found for this plate.'})
    except Exception as e:
        return jsonify({'found': False, 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"===============================================================")
    print(f"🚀 Smart Parking Assistant Web Service is starting...")
    print(f"👉 Local Access URL: http://127.0.0.1:{port}")
    print(f"👤 Assistant Demo Login: username='assistant', password='admin123'")
    print(f"🚗 Driver Demo Login:    username='driver',    password='driver123'")
    print(f"===============================================================")
    app.run(host='0.0.0.0', port=port, debug=False)
