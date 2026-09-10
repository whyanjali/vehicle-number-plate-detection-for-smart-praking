import os
import sys
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

PDF_FILENAME = "Smart_Parking_Assistant_Synopsis.pdf"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        # Don't draw header/footer on cover page (page 1)
        if self._pageNumber == 1:
            return

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Header
        self.drawString(54, 11 * inch - 36, "Smart Parking Assistant with Real-Time ALPR — Project Synopsis")
        self.setStrokeColor(colors.HexColor("#D0D0D0"))
        self.setLineWidth(0.5)
        self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)

        # Footer
        self.line(54, 48, 8.5 * inch - 54, 48)
        self.drawString(54, 34, "Greater Noida Institute of Technology (GNIOT) | AKTU, Lucknow")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 34, page_text)

        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        PDF_FILENAME,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_cover = ParagraphStyle(
        'CoverTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=26,
        alignment=1, # Center
        textColor=colors.HexColor("#0B1B3D")
    )

    sub_cover = ParagraphStyle(
        'CoverSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        alignment=1,
        textColor=colors.HexColor("#2C3E50")
    )

    h1_style = ParagraphStyle(
        'MainH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        spaceAfter=12,
        textColor=colors.HexColor("#0B1B3D"),
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'MainH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        spaceBefore=10,
        spaceAfter=8,
        textColor=colors.HexColor("#1A365D"),
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'MainBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=15,
        alignment=4, # Justified
        spaceAfter=8,
        textColor=colors.HexColor("#1F2937")
    )

    bullet_style = ParagraphStyle(
        'MainBullet',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=5,
        textColor=colors.HexColor("#1F2937")
    )

    code_block_style = ParagraphStyle(
        'CodeBlock',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#111827")
    )

    story = []

    # =========================================================================
    # PAGE 1: COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("Synopsis<br/>On", sub_cover))
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>Smart Parking Assistant with Real-Time Vehicle Number Plate Recognition (ALPR)</b>", title_cover))
    story.append(Spacer(1, 30))

    # GNIOT Box Representation
    gniot_table = Table(
        [[Paragraph("<b>GNIOT<br/><font size=9>GROUP OF INSTITUTIONS</font></b>", ParagraphStyle('GniotL', fontName='Helvetica-Bold', fontSize=16, leading=18, alignment=1, textColor=colors.HexColor("#0B1B3D"))),
          Paragraph("<b>GNIOT</b><br/><font size=8 color='#1E40AF'>ENGG. INSTITUTE</font>", ParagraphStyle('GniotR', fontName='Helvetica-Bold', fontSize=22, leading=22, alignment=1, textColor=colors.HexColor("#F59E0B")))]],
        colWidths=[150, 200]
    )
    gniot_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor("#0B1B3D")),
        ('INNERGRID', (0,0), (-1,-1), 1, colors.HexColor("#E5E7EB")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(gniot_table)
    story.append(Spacer(1, 35))

    story.append(Paragraph("<b>BACHELOR OF TECHNOLOGY</b>", sub_cover))
    story.append(Spacer(1, 4))
    story.append(Paragraph("DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING (AI)", sub_cover))
    story.append(Spacer(1, 4))
    story.append(Paragraph("3<sup>rd</sup> Semester (2024–25)", sub_cover))
    story.append(Spacer(1, 50))

    # Submitted By and Supervisor Table
    sub_table = Table(
        [
            [Paragraph("<b>Submitted By:</b>", ParagraphStyle('SubH', fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor("#0B1B3D"))),
             Paragraph("<b>Project Supervisor:</b>", ParagraphStyle('SupH', fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor("#0B1B3D")))],
            [Paragraph("Anjali Tiwari (2301321520024)<br/>Ankit Raj (2301321520025)<br/>Ankita (2301321520026)", ParagraphStyle('SubD', fontName='Helvetica', fontSize=9.5, leading=14)),
             Paragraph("Mr. Pancham<br/><i>Assistant Professor</i><br/>Department of CSE-AI", ParagraphStyle('SupD', fontName='Helvetica', fontSize=9.5, leading=14))]
        ],
        colWidths=[250, 250]
    )
    sub_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(sub_table)

    story.append(Spacer(1, 50))
    story.append(Paragraph("<b>Greater Noida Institute of Technology (Engg. Institute), Greater Noida</b>", sub_cover))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Dr. A.P.J. Abdul Kalam Technical University, Lucknow</b>", sub_cover))
    story.append(Spacer(1, 15))
    story.append(Paragraph("October, 2024", ParagraphStyle('DateSub', fontName='Helvetica', fontSize=10, alignment=1, textColor=colors.HexColor("#4B5563"))))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: INDEX
    # =========================================================================
    story.append(Paragraph("<b>INDEX</b>", ParagraphStyle('IndexH', fontName='Helvetica-Bold', fontSize=16, leading=20, alignment=1, spaceAfter=20)))

    index_data = [
        [Paragraph("<b>S.No.</b>", ParagraphStyle('IdxH1', fontName='Helvetica-Bold', fontSize=10)),
         Paragraph("<b>Topic</b>", ParagraphStyle('IdxH2', fontName='Helvetica-Bold', fontSize=10)),
         Paragraph("<b>Page No.</b>", ParagraphStyle('IdxH3', fontName='Helvetica-Bold', fontSize=10, alignment=2))]
    ]

    toc_rows = [
        ("1.", "Introduction", "3"),
        ("", "1.1. Problem Statement", "4"),
        ("", "1.2. Scope of Project", "5"),
        ("2.", "Tools / Environment Used", "6"),
        ("3.", "Analysis Document", "7"),
        ("", "3.1. E-R Diagram & Database Schema", "7"),
        ("", "3.2. System Architecture & Data Flow Diagram", "8"),
        ("4.", "Limitations of the Project", "9"),
        ("5.", "Result and Future Scope of the Project", "10"),
        ("", "5.1. Result and Key Findings", "10"),
        ("", "5.2. Future Scope and Enhancements", "10")
    ]

    for s_no, topic, pg in toc_rows:
        index_data.append([
            Paragraph(f"<b>{s_no}</b>" if s_no else "", ParagraphStyle('T1', fontName='Helvetica-Bold' if s_no else 'Helvetica', fontSize=9.5)),
            Paragraph(f"<b>{topic}</b>" if s_no else f"&nbsp;&nbsp;&nbsp;&nbsp;{topic}", ParagraphStyle('T2', fontName='Helvetica-Bold' if s_no else 'Helvetica', fontSize=9.5)),
            Paragraph(pg, ParagraphStyle('T3', fontName='Helvetica', fontSize=9.5, alignment=2))
        ])

    index_table = Table(index_data, colWidths=[40, 400, 60])
    index_table.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,0), 1.5, colors.HexColor("#0B1B3D")),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('LINEBELOW', (0,1), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
    ]))
    story.append(index_table)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: 1. INTRODUCTION
    # =========================================================================
    story.append(Paragraph("1. INTRODUCTION", h1_style))
    story.append(Paragraph("<b>INTRODUCTION TO SMART PARKING & ALPR SYSTEM</b>", h2_style))

    story.append(Paragraph(
        "With the rapid pace of urbanization and the continuous increase in automobile density across metropolitan centers, "
        "traditional parking management infrastructures have become severe operational bottlenecks. Manual parking systems—which "
        "rely heavily on human security personnel issuing hand-written paper slips, plastic barcode tokens, or physical cash collection—suffer "
        "from notable systemic delays, long vehicular queues at entry boom barriers, high labor overheads, and frequent revenue leakage.",
        body_style
    ))

    story.append(Paragraph(
        "The <b>Smart Parking Assistant with Real-Time Automatic License Plate Recognition (ALPR)</b> is an intelligent, automated "
        "computer vision and web-based solution designed to transform physical parking operations into a seamless, digitized process. "
        "The system integrates modern deep learning object detection algorithms (YOLOv8) with advanced optical character recognition (EasyOCR) "
        "to automatically detect arriving vehicles, isolate license plate bounding boxes, extract registration alphanumeric strings, and "
        "assign optimal vacant parking bays without requiring physical intervention.",
        body_style
    ))

    story.append(Paragraph(
        "Unlike conventional standalone recognition scripts, this project bridges the physical computer vision pipeline with a full-stack, "
        "interactive web application. It features a <b>Dual-Role Architecture</b> tailored to two principal stakeholders: "
        "<br/>• <b>Parking Assistant (Gate Attendant):</b> Equipped with a live operations console featuring real-time camera/image ALPR scanning, "
        "an interactive 2D parking lot occupancy map, 1-click slot allotment, and an automated checkout billing desk. "
        "<br/>• <b>Vehicle Driver:</b> Equipped with a self-service portal displaying live vacant bay counts, self-check-in allotment, "
        "a high-contrast Digital Parking Pass with QR/barcode reference, and turn-by-turn wayfinding directions from Entry Gate 1 to their assigned spot.",
        body_style
    ))

    story.append(Paragraph(
        "By automating vehicle tracking from entry to checkout, the system accelerates ingress and egress flow, minimizes fuel wastage caused by "
        "drivers aimlessly circling parking garages, guarantees transparent revenue logging, and establishes an intelligent smart-city foundation.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: 1.1 PROBLEM STATEMENT
    # =========================================================================
    story.append(Paragraph("1.1 PROBLEM STATEMENT", h1_style))

    story.append(Paragraph(
        "Contemporary urban parking facilities in commercial malls, university campuses, IT parks, and transit hubs face critical operational "
        "inefficiencies that compromise safety, throughput, and driver satisfaction:",
        body_style
    ))

    story.append(Paragraph(
        "<b>1. Manual Ingress Bottlenecks and Gate Queuing:</b> Gate attendants must manually inspect each vehicle, enter plate details into "
        "handheld machines, or issue physical paper tokens. During morning rush hours and peak commercial times, this manual process causes "
        "severe queues spilling over onto public roadways, creating congestion and elevated carbon emissions.",
        bullet_style
    ))

    story.append(Paragraph(
        "<b>2. Lack of Real-Time Bay Occupancy Visibility:</b> Upon clearing the security gate, drivers possess zero visibility regarding which aisles, "
        "floors, or sections contain vacant spaces. Research shows that drivers spend an average of 7 to 14 minutes merely circling parking floors "
        "to locate a vacant space, resulting in elevated stress, wasted fuel, and internal garage gridlocks.",
        bullet_style
    ))

    story.append(Paragraph(
        "<b>3. Financial Disputes and Revenue Leakage:</b> Paper slips and magnetic tokens are easily misplaced, stolen, or damaged. Calculating "
        "parking duration based on hand-written receipts frequently creates disputes between motorists and attendants, while cash transactions without "
        "immutable audit logs lead to substantial revenue leakages for facility operators.",
        bullet_style
    ))

    story.append(Paragraph(
        "<b>4. High Complexity of Indian Vehicle License Plates:</b> Implementing automated ALPR in Indian environments presents unique challenges. "
        "Indian plates feature diverse regional state codes (DL, HR, MH, UP, KA, etc.), variable non-standard font variations, two-line stacked layouts "
        "(common on two-wheelers and commercial vehicles), and severe environmental degradation such as dust, scratches, low ambient illumination, "
        "and camera glare.",
        bullet_style
    ))

    story.append(Paragraph(
        "<b>The Challenge:</b> To develop a robust, end-to-end intelligent Smart Parking Assistant capable of delivering high-accuracy license "
        "plate recognition across challenging real-world Indian road scenarios, maintaining an atomic real-time map of vacant vs. occupied spaces, "
        "and automating bay allotment and tariff billing through an accessible dual-role interface.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: 1.2 SCOPE OF PROJECT
    # =========================================================================
    story.append(Paragraph("1.2 SCOPE OF PROJECT", h1_style))

    story.append(Paragraph(
        "The scope of this project encompasses the end-to-end design, implementation, evaluation, and deployment of a full-stack Smart Parking "
        "Assistant. It spans the following core functional modules:",
        body_style
    ))

    scope_items = [
        ("Vehicle & License Plate Detection Module", 
         "Dual deep learning pipeline utilizing YOLOv8n for vehicle identification (cars, SUVs, motorcycles) and a fine-tuned custom YOLOv8 model "
         "for locating license plate bounding boxes with context padding to prevent character clipping."),
        ("Multi-Pass Image Enhancement & Preprocessing", 
         "Super-resolution cubic upscaling (minimum 120px height), Bilateral noise filtering to maintain crisp character boundaries, Contrast "
         "Limited Adaptive Histogram Equalization (CLAHE), and Otsu adaptive binarization."),
        ("Optical Character Recognition (OCR) Engine", 
         "EasyOCR character extraction operating across multiple preprocessed image representations, equipped with spatial token sorting that "
         "assembles fragmented bounding boxes horizontally or vertically based on plate aspect ratio."),
        ("Indian Plate Format Normalizer & Validator", 
         "Rule-based character disambiguation correcting common optical confusions (O/0, I/1, Z/2, S/5, B/8) and matching candidate strings "
         "against standard RTO state codes and the national Indian plate regex: ^[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{4}$."),
        ("Interactive 2D Parking Bay Map & Allotment Engine", 
         "Dynamic visual bay grid representing Ground Floor (Section A), Level 1 (Section B), and EV Priority (Section C). Implements automated "
         "optimal vacant slot assignment and manual operator overrides."),
        ("Parking Assistant Operations Console", 
         "Web-based attendant control center supporting live webcam streaming, photo uploads, test dataset presets, 2D occupancy visualization, "
         "duration counters, and 1-click checkout with printable receipts."),
        ("Driver Self-Service Portal", 
         "Driver portal showing real-time vacancy counts, self-check-in, digital parking passes with barcodes, and turn-by-turn navigation "
         "directions from Entry Gate 1 to the assigned parking bay."),
        ("Transaction Logging & Tariff Calculation", 
         "Atomic SQLite database maintaining audit logs of entry/exit timestamps, elapsed durations, hourly tariff calculation (₹30/hr), and revenue metrics.")
    ]

    for title, desc in scope_items:
        story.append(Paragraph(f"• <b>{title}:</b> {desc}", bullet_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 6: 2. TOOLS / ENVIRONMENT USED
    # =========================================================================
    story.append(Paragraph("2. TOOLS / ENVIRONMENT USED", h1_style))

    story.append(Paragraph(
        "Below are the specialized tools, programming languages, libraries, and environments utilized in developing the Smart Parking Assistant:",
        body_style
    ))

    tools_data = [
        [Paragraph("<b>Category</b>", ParagraphStyle('TC1', fontName='Helvetica-Bold', fontSize=9.5)),
         Paragraph("<b>Technology / Tool</b>", ParagraphStyle('TC2', fontName='Helvetica-Bold', fontSize=9.5)),
         Paragraph("<b>Functional Purpose</b>", ParagraphStyle('TC3', fontName='Helvetica-Bold', fontSize=9.5))]
    ]

    tools_rows = [
        ("Programming Language", "Python 3.10 / 3.11", "Core language for computer vision, AI inference, and backend web logic."),
        ("Web Framework", "Flask 3.1", "Micro-web framework providing session management, template rendering, and RESTful JSON APIs."),
        ("Deep Learning Framework", "Ultralytics YOLOv8 & PyTorch", "Dual-stage YOLOv8 models for vehicle classification and license plate bounding box localization."),
        ("Computer Vision", "OpenCV (cv2) 4.13", "Image preprocessing: cubic upscaling, bilateral filtering, CLAHE contrast enhancement, and annotation rendering."),
        ("OCR Engine", "EasyOCR 1.7", "Deep learning optical character reader with PyTorch-backed alphanumeric extraction."),
        ("Database", "SQLite 3", "Relational database engine for atomic storage of users, parking slots, and historical parking tickets."),
        ("Frontend UI", "HTML5, CSS3, JavaScript", "Modern responsive interface with glassmorphism dark-mode aesthetics."),
        ("UI Components", "Bootstrap 5.3 & FontAwesome 6", "Responsive grid system, interactive modals, navigation pills, and vector iconography."),
        ("Development IDE", "Visual Studio Code (VS Code)", "Source code editing, virtual environment debugging (yolov8_env), and terminal management."),
        ("Version Control", "Git & GitHub", "Repository hosting, change tracking, and continuous code synchronization.")
    ]

    for cat, tech, purp in tools_rows:
        tools_data.append([
            Paragraph(f"<b>{cat}</b>", ParagraphStyle('R1', fontName='Helvetica-Bold', fontSize=8.5)),
            Paragraph(tech, ParagraphStyle('R2', fontName='Helvetica', fontSize=8.5)),
            Paragraph(purp, ParagraphStyle('R3', fontName='Helvetica', fontSize=8.5))
        ])

    tools_table = Table(tools_data, colWidths=[120, 150, 230])
    tools_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F3F4F6")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(tools_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>Hardware Requirements:</b>", h2_style))
    story.append(Paragraph("• <b>Camera Sensor:</b> Standard HD USB webcam, integrated laptop camera, or RTSP IP surveillance camera (minimum 720p).", bullet_style))
    story.append(Paragraph("• <b>Processor (CPU):</b> Intel Core i5 / AMD Ryzen 5 (or higher) with multi-threading capabilities.", bullet_style))
    story.append(Paragraph("• <b>System Memory (RAM):</b> 8 GB minimum (16 GB recommended for concurrent YOLO + EasyOCR execution).", bullet_style))
    story.append(Paragraph("• <b>Storage:</b> Minimum 5 GB free disk space for PyTorch models, weight files (`best.pt`, `yolov8n.pt`), and database logs.", bullet_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 7: 3. ANALYSIS DOCUMENT (E-R & Schema)
    # =========================================================================
    story.append(Paragraph("3. ANALYSIS DOCUMENT", h1_style))
    story.append(Paragraph("3.1. ENTITY-RELATIONSHIP (E-R) & DATABASE SCHEMA", h2_style))

    story.append(Paragraph(
        "The relational database schema is modeled in SQLite (`parking_system.db`) and consists of three principal entities: "
        "<b>USERS</b>, <b>PARKING_SLOTS</b>, and <b>PARKING_RECORDS</b>. Below is the structural data dictionary and relational mapping:",
        body_style
    ))

    schema_data = [
        [Paragraph("<b>Table</b>", ParagraphStyle('SH1', fontName='Helvetica-Bold', fontSize=8.5)),
         Paragraph("<b>Field Name</b>", ParagraphStyle('SH2', fontName='Helvetica-Bold', fontSize=8.5)),
         Paragraph("<b>Type & Constraint</b>", ParagraphStyle('SH3', fontName='Helvetica-Bold', fontSize=8.5)),
         Paragraph("<b>Description</b>", ParagraphStyle('SH4', fontName='Helvetica-Bold', fontSize=8.5))]
    ]

    schema_rows = [
        ("users", "id", "INTEGER (PK, AI)", "Unique user identifier."),
        ("users", "username", "TEXT (Unique)", "Login username (e.g. 'assistant', 'driver')."),
        ("users", "password_hash", "TEXT", "Securely hashed password (PBKDF2-SHA256)."),
        ("users", "role", "TEXT", "'assistant' for staff, 'driver' for vehicle owners."),
        ("users", "vehicle_plate", "TEXT", "Registered vehicle license plate number."),
        ("parking_slots", "slot_number", "TEXT (Unique, PK)", "Physical bay identifier (e.g. 'A-01', 'B-02', 'C-01')."),
        ("parking_slots", "section", "TEXT", "Facility zone ('Section A (Ground)', 'Section B', 'Section C (EV)')."),
        ("parking_slots", "slot_type", "TEXT", "Category ('Standard', 'Compact', 'EV')."),
        ("parking_slots", "is_occupied", "INTEGER", "Binary status: 0 for Vacant (Green), 1 for Occupied (Red)."),
        ("parking_slots", "current_plate", "TEXT (FK)", "Plate number of currently parked vehicle."),
        ("parking_slots", "entry_time", "TIMESTAMP", "Time vehicle was allotted to this bay."),
        ("parking_records", "ticket_id", "TEXT (Unique, PK)", "Unique ticket reference (e.g. 'TKT-1788989032')."),
        ("parking_records", "plate_number", "TEXT", "License plate number of vehicle."),
        ("parking_records", "duration_minutes", "INTEGER", "Total elapsed minutes between entry and exit."),
        ("parking_records", "total_fee", "REAL", "Calculated parking charge (₹30 per billed hour)."),
        ("parking_records", "status", "TEXT", "Ticket status ('ACTIVE' or 'COMPLETED').")
    ]

    for tbl, fld, typ, desc in schema_rows:
        schema_data.append([
            Paragraph(f"<b>{tbl}</b>", ParagraphStyle('SR1', fontName='Helvetica-Bold', fontSize=8)),
            Paragraph(fld, ParagraphStyle('SR2', fontName='Helvetica', fontSize=8)),
            Paragraph(typ, ParagraphStyle('SR3', fontName='Helvetica', fontSize=8)),
            Paragraph(desc, ParagraphStyle('SR4', fontName='Helvetica', fontSize=8))
        ])

    schema_table = Table(schema_data, colWidths=[90, 100, 110, 200])
    schema_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F3F4F6")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(schema_table)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 8: 3.2 DATA FLOW & SYSTEM BLOCK DIAGRAM
    # =========================================================================
    story.append(Paragraph("3.2. SYSTEM ARCHITECTURE & DATA FLOW DIAGRAM", h1_style))

    story.append(Paragraph(
        "The system pipeline processes camera inputs through a sequential five-stage computer vision and web architecture:",
        body_style
    ))

    dfd_ascii = (
        "+-------------------------------------------------------------------------+\n"
        "|                 STAGE 1: IMAGE ACQUISITION & DETECTION                  |\n"
        "|  Camera / Upload / Preset  --> YOLOv8n (Vehicle) + Custom YOLOv8 (Plate)|\n"
        "+-------------------------------------------------------------------------+\n"
        "                                     |\n"
        "                                     v\n"
        "+-------------------------------------------------------------------------+\n"
        "|                 STAGE 2: PREPROCESSING & ENHANCEMENT                    |\n"
        "|  +8% Horiz / +12% Vert Padding --> Cubic Upscaling --> CLAHE + Bilateral|\n"
        "+-------------------------------------------------------------------------+\n"
        "                                     |\n"
        "                                     v\n"
        "+-------------------------------------------------------------------------+\n"
        "|                 STAGE 3: MULTI-PASS OPTICAL OCR                         |\n"
        "|  EasyOCR Engine --> Spatial Token Sorting (Aspect-Ratio Left-to-Right)  |\n"
        "+-------------------------------------------------------------------------+\n"
        "                                     |\n"
        "                                     v\n"
        "+-------------------------------------------------------------------------+\n"
        "|                 STAGE 4: INDIAN PLATE NORMALIZATION                     |\n"
        "|  State Code Validation (DL/MH/HR/KA) --> O/0, I/1, Z/2, S/5 Correction  |\n"
        "+-------------------------------------------------------------------------+\n"
        "                                     |\n"
        "                                     v\n"
        "+-------------------------------------------------------------------------+\n"
        "|                 STAGE 5: ALLOTMENT ENGINE & WEB DASHBOARDS              |\n"
        "|  Allot Nearest Vacant Bay --> SQLite DB Update --> Active Pass / Ticket |\n"
        "|  Assistant Console: 2D Grid & Billing  | Driver Portal: Pass & Wayfinding|\n"
        "+-------------------------------------------------------------------------+"
    )
    story.append(Paragraph(f"<pre>{dfd_ascii}</pre>", code_block_style))
    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>Detailed Data Flow Explanation:</b>", h2_style))
    story.append(Paragraph(
        "<b>Level 0 (Context):</b> The external vehicle interacts with the ALPR Scanner. Extracted registration numbers are routed to the "
        "Central Parking Controller, which consults the Bay Allocation Table, allots an available slot, and renders operational views.",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>Level 1 (Decomposition):</b> Frame capture $\\rightarrow$ Plate localization $\\rightarrow$ CLAHE enhancement $\\rightarrow$ OCR token extraction "
        "$\\rightarrow$ Indian regex validation $\\rightarrow$ Database status toggle (0 to 1) $\\rightarrow$ Real-time WebSocket/polling refresh on 2D map.",
        bullet_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 9: 4. LIMITATIONS OF THE PROJECT
    # =========================================================================
    story.append(Paragraph("4. LIMITATIONS OF THE PROJECT", h1_style))

    story.append(Paragraph(
        "While the Smart Parking Assistant delivers robust real-time accuracy and automation, several operational and environmental "
        "limitations must be noted:",
        body_style
    ))

    limitations = [
        ("Extreme Environmental & Weather Interference",
         "Severe weather conditions such as dense monsoon rains, heavy fog, snow, or mud splatters on the license plate surface can obscure "
         "character contours, leading to reduced OCR confidence scores."),
        ("Non-Standard & Customized License Plates",
         "Although Indian law mandates High Security Registration Plates (HSRP), many older vehicles continue to use customized, artistic, "
         "or localized regional language fonts (e.g. Devanagari numerals), which deviate from the standard alphanumeric OCR vocabulary."),
        ("Steep Perspective & Angular Distortion",
         "When vehicles approach gate cameras at extreme horizontal or vertical angles exceeding 45 degrees, the resulting perspective skew "
         "compresses letters, necessitating planar homography rectification for optimal character segmentation."),
        ("Lighting Extremes & Headlight Glare",
         "Direct high-beam headlight glare at night can temporarily saturate optical camera sensors (over-exposure), obscuring the plate area "
         "unless equipped with polarized optical filters or infrared (IR) night illuminators."),
        ("Absence of Physical Enforcement Barriers",
         "As a software-driven intelligent assistant, the system assigns bays digitally. Without physical robotic bollards or gate boom barrier "
         "relays, it cannot physically prevent an unauthorized car from occupying an assigned bay before the valid motorist reaches it.")
    ]

    for title, desc in limitations:
        story.append(Paragraph(f"• <b>{title}:</b> {desc}", bullet_style))
        story.append(Spacer(1, 4))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 10: 5. RESULT AND FUTURE SCOPE OF THE PROJECT
    # =========================================================================
    story.append(Paragraph("5. RESULT AND FUTURE SCOPE OF THE PROJECT", h1_style))

    story.append(Paragraph("5.1. RESULT AND DEMONSTRATED OUTCOMES", h2_style))
    story.append(Paragraph(
        "The development, integration, and benchmarking of the Smart Parking Assistant demonstrated the following concrete outcomes:",
        body_style
    ))

    results = [
        ("High-Accuracy License Plate Detection",
         "Through the integration of context padding, multi-pass enhancement, and spatial token sorting, the ALPR pipeline achieved up to "
         "<b>100% confidence</b> on standard test dataset plates (e.g., extracting complete 10-character plates such as 'HR26BR9044' and 'HR26CH3604')."),
        ("Automated Instant Bay Allotment",
         "The smart allotment engine successfully allocates the optimal vacant space in under <b>0.5 seconds</b> upon plate verification, updating "
         "the visual 2D bay map from Green (Vacant) to Red (Occupied) in real time."),
        ("Dual-Role Dashboard Operational Efficiency",
         "The web interface cleanly separates operational controls: attendants can scan plates and issue checkout receipts with 1-click billing, "
         "while drivers receive a high-contrast Digital Parking Pass with turn-by-turn guidance from Entry Gate 1 to their specific bay."),
        ("Full Audit Trail & Accounting Integrity",
         "The relational SQLite database maintains permanent transaction logs including ticket IDs, entry/exit timestamps, billed hours, and "
         "total revenue, eliminating manual paperwork and revenue leakage.")
    ]

    for title, desc in results:
        story.append(Paragraph(f"• <b>{title}:</b> {desc}", bullet_style))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 10))
    story.append(Paragraph("5.2. FUTURE SCOPE & ENHANCEMENTS", h2_style))
    story.append(Paragraph(
        "The capabilities of this Smart Parking platform can be extended in future iterations through the following enhancements:",
        body_style
    ))

    future_items = [
        ("Embedded Edge Deployment (IoT)",
         "Optimizing the YOLOv8 and EasyOCR models using TensorRT / ONNX runtime for deployment on low-power edge computing devices "
         "such as the <b>NVIDIA Jetson Nano</b> or <b>Raspberry Pi 5</b> installed directly inside the gate boom barrier housing."),
        ("Automated Boom Barrier Interfacing",
         "Integrating GPIO microcontroller relays (ESP32 / Arduino) to automatically lift physical boom barriers upon successful plate validation."),
        ("FASTag & UPI Payment Gateway Integration",
         "Connecting RFID FASTag readers and generating dynamic Bharat-QR / UPI codes (PhonePe, Google Pay, Paytm) on checkout for contactless "
         "cashless settlements."),
        ("CCTV Overhead Slot Occupancy Detection",
         "Deploying wide-angle overhead cameras with polygon zone detection to continuously monitor whether physical bays are vacant, "
         "misaligned, or occupied by unauthorized vehicles."),
        ("Mobile App with Pre-Booking Navigation",
         "Developing a native Android/iOS mobile application allowing motorists to pre-book parking bays and receive GPS navigation directly "
         "to their assigned bay.")
    ]

    for title, desc in future_items:
        story.append(Paragraph(f"• <b>{title}:</b> {desc}", bullet_style))
        story.append(Spacer(1, 3))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {PDF_FILENAME}")

if __name__ == '__main__':
    build_pdf()
