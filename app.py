
import os
import random
import re
import uuid
import secrets
from datetime import datetime, timezone
from io import BytesIO
from functools import wraps

from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_file, flash, session, g, abort
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.engine import make_url
from PIL import Image, ImageStat, ImageDraw, ImageFilter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage,
    Table, TableStyle
)
from markupsafe import escape
from language_support import init_language
from translations import translate_dr_level, translate_stage

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = bool(os.environ.get("RENDER"))

init_language(app)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

database_url = os.environ.get("DATABASE_URL")
if not database_url:
    raise RuntimeError("DATABASE_URL is not configured.")

db_url = make_url(database_url)
if db_url.drivername in ("postgres", "postgresql"):
    db_url = db_url.set(drivername="postgresql+psycopg")

query_parameters = dict(db_url.query)
query_parameters["sslmode"] = "require"
db_url = db_url.set(query=query_parameters)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url.render_as_string(hide_password=False)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,
    "pool_recycle": 300
}

db = SQLAlchemy(app)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


# -------------------- DATABASE MODELS --------------------

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), unique=True, nullable=True, index=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    phone = db.Column(db.String(20))
    registration_number = db.Column(db.String(100))
    medical_registration_number = db.Column(db.String(100))
    role = db.Column(db.String(20), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    verification_status = db.Column(
        db.String(20), nullable=False, default="not_applicable"
    )
    created_at = db.Column(
        db.DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Screening(db.Model):
    __tablename__ = "screenings"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(
        db.Integer, db.ForeignKey("users.id"), nullable=False, index=True
    )
    image_filename = db.Column(db.String(255), nullable=False)
    grade = db.Column(db.Integer, nullable=False)
    analysis_mode = db.Column(db.String(50), nullable=False, default="simulated")
    created_at = db.Column(
        db.DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )


class DoctorReview(db.Model):
    __tablename__ = "doctor_reviews"

    id = db.Column(db.Integer, primary_key=True)
    screening_id = db.Column(
        db.Integer, db.ForeignKey("screenings.id"), nullable=False, index=True
    )
    doctor_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    classification_comment = db.Column(db.Text)
    heatmap_comment = db.Column(db.Text)
    progress_comment = db.Column(db.Text)
    overall_advice = db.Column(db.Text)
    created_at = db.Column(
        db.DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )


with app.app_context():
    db.create_all()


# -------------------- CLASSIFICATION LABELS --------------------

DR_LEVELS = {
    0: {
        "name": "No DR",
        "description": "No diabetic retinopathy detected",
        "severity": "Normal"
    },
    1: {
        "name": "Mild Non-Proliferative DR",
        "description": "Early signs of diabetic retinopathy",
        "severity": "Mild"
    },
    2: {
        "name": "Moderate Non-Proliferative DR",
        "description": "More noticeable retinal changes",
        "severity": "Moderate"
    },
    3: {
        "name": "Severe Non-Proliferative DR",
        "description": "Advanced retinal changes",
        "severity": "Severe"
    },
    4: {
        "name": "Proliferative DR",
        "description": "Advanced stage of diabetic retinopathy",
        "severity": "Proliferative"
    }
}


# -------------------- AUTHENTICATION --------------------

def get_csrf_token():
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_hex(32)
    return session["_csrf_token"]


@app.context_processor
def inject_template_values():
    return {
        "csrf_token": get_csrf_token,
        "current_user": getattr(g, "current_user", None)
    }


@app.before_request
def load_logged_in_user():
    user_id = session.get("user_id")
    g.current_user = db.session.get(User, user_id) if user_id else None


def validate_csrf():
    submitted = request.form.get("csrf_token", "")
    stored = session.get("_csrf_token", "")
    return bool(stored and secrets.compare_digest(submitted, stored))


def login_required(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        if not g.current_user:
            flash("Please log in to continue.", "error")
            return redirect(url_for("login", role="patient"))
        return function(*args, **kwargs)
    return wrapper


def role_required(role):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            if not g.current_user:
                flash("Please log in to continue.", "error")
                return redirect(url_for("login", role=role))
            if g.current_user.role != role:
                abort(403)
            return function(*args, **kwargs)
        return wrapper
    return decorator


def generate_patient_id():
    year = datetime.now().year
    for _ in range(30):
        value = f"RET-{year}-{secrets.randbelow(100000):05d}"
        if not User.query.filter_by(patient_id=value).first():
            return value
    raise RuntimeError("Could not generate a unique Patient ID.")


# -------------------- IMAGE CHECKS --------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_file(file):
    if not file or not file.filename:
        return False, "Please select an image."
    if not allowed_file(file.filename):
        return False, "Only JPG, JPEG and PNG images are allowed."
    return True, "File format verified successfully."


def verify_image_quality(filepath):
    try:
        with Image.open(filepath) as image:
            image.verify()

        with Image.open(filepath) as image:
            width, height = image.size

            if width < 100 or height < 100:
                return False, "Image resolution is too low."

            if width > 10000 or height > 10000:
                return False, "Image dimensions are too large."

        return True, "Image is readable and resolution is acceptable."

    except Exception:
        return False, "The uploaded file is not a valid image."


def is_retinal_image_demo(filepath):
    try:
        with Image.open(filepath) as image:
            image = image.convert("RGB")
            width, height = image.size

            if not width or not height or not 0.5 <= width / height <= 2.0:
                return False, "The image shape is not suitable for this demo."

            gray = image.convert("L")
            centre = gray.crop((
                int(width * .25), int(height * .25),
                int(width * .75), int(height * .75)
            ))

            size = max(1, min(width, height) // 10)

            corners = [
                gray.crop((0, 0, size, size)),
                gray.crop((width-size, 0, width, size)),
                gray.crop((0, height-size, size, height)),
                gray.crop((width-size, height-size, width, height))
            ]

            centre_brightness = ImageStat.Stat(centre).mean[0]
            corner_brightness = sum(
                ImageStat.Stat(c).mean[0] for c in corners
            ) / 4

            if centre_brightness < 10:
                return False, "The image appears too dark."

            if centre_brightness < corner_brightness * .65:
                return False, "The image did not pass the basic visual check."

            return True, "The image passed the basic demonstration check."

    except Exception:
        return False, "Unable to analyse the uploaded image."


def verify_all_stages(filepath):
    quality = verify_image_quality(filepath)

    retinal = (
        is_retinal_image_demo(filepath) if quality[0]
        else (False, "Skipped because the image quality check failed.")
    )

    stages = [
        {
            "name": "File Validation",
            "status": "passed",
            "message": "File format verified successfully."
        },
        {
            "name": "Image Quality Check",
            "status": "passed" if quality[0] else "failed",
            "message": quality[1]
        },
        {
            "name": "Retinal Image Demonstration Check",
            "status": "passed" if retinal[0] else "failed",
            "message": retinal[1]
        }
    ]

    return quality[0] and retinal[0], stages


# -------------------- SIMULATED HEATMAP --------------------

def generate_simulated_heatmap(filepath, seed="retinaai"):
    """
    Generates a repeatable illustrative heatmap overlay.

    This is not generated by a trained AI model and does not
    identify diabetic-retinopathy regions.
    """

    with Image.open(filepath) as source:
        original = source.convert("RGB")
        original.thumbnail((1200, 1200))

    width, height = original.size

    if width < 1 or height < 1:
        raise ValueError("Invalid image dimensions.")

    # Create an intensity mask.
    mask = Image.new("L", (width, height), 12)
    draw = ImageDraw.Draw(mask)

    # Use the same seed for the same screening.
    rng = random.Random(str(seed))

    # Add illustrative activation regions.
    for _ in range(9):
        x = rng.randint(
            max(0, int(width * 0.12)),
            max(0, int(width * 0.88))
        )
        y = rng.randint(
            max(0, int(height * 0.12)),
            max(0, int(height * 0.88))
        )

        radius = rng.randint(
            max(8, int(min(width, height) * 0.05)),
            max(12, int(min(width, height) * 0.17))
        )

        intensity = rng.randint(120, 255)

        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=intensity
        )

    # Smooth the regions into soft heatmap-like areas.
    blur_radius = max(8, int(min(width, height) * 0.07))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Build a blue -> green -> yellow -> red colour palette.
    palette = []

    colour_stops = [
        (0,   (0, 20, 255)),     # Blue
        (85,  (0, 230, 100)),    # Green
        (170, (255, 235, 0)),    # Yellow
        (255, (255, 20, 0))      # Red
    ]

    for value in range(256):
        for index in range(len(colour_stops) - 1):
            start_value, start_colour = colour_stops[index]
            end_value, end_colour = colour_stops[index + 1]

            if start_value <= value <= end_value:
                fraction = (
                    (value - start_value) /
                    (end_value - start_value)
                )

                colour = tuple(
                    int(
                        start_colour[channel] +
                        (end_colour[channel] - start_colour[channel]) *
                        fraction
                    )
                    for channel in range(3)
                )

                palette.extend(colour)
                break

    # Convert the mask using the custom colour palette.
    palette_image = mask.convert("P")
    palette_image.putpalette(palette)

    coloured_heatmap = palette_image.convert("RGB")

    # Blend the illustrative colours with the original image.
    result = Image.blend(original, coloured_heatmap, 0.48)

    output = BytesIO()
    result.save(output, format="PNG")
    output.seek(0)

    return output


# -------------------- ACCOUNT ROUTES --------------------

@app.route("/account")
def account():
    return render_template("account.html")


@app.route("/register/<role>", methods=["GET", "POST"])
def register(role):
    if role not in ("patient", "doctor"):
        return redirect(url_for("account"))

    if request.method == "POST":
        if not validate_csrf():
            flash("Your form session expired. Please try again.", "error")
            return redirect(url_for("register", role=role))

        name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("Please complete all required fields.", "error")
            return redirect(url_for("register", role=role))

        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            flash("Please enter a valid email address.", "error")
            return redirect(url_for("register", role=role))

        if len(password) < 8:
            flash("Password must contain at least 8 characters.", "error")
            return redirect(url_for("register", role=role))

        if User.query.filter_by(email=email).first():
            flash("An account with this email already exists.", "error")
            return redirect(url_for("register", role=role))

        user = User(full_name=name, email=email, role=role)
        user.set_password(password)

        if role == "patient":
            phone = request.form.get("phone", "").strip()
            reg = request.form.get("registration_number", "").strip()

            if not phone or not reg:
                flash("Please enter your phone and registration number.", "error")
                return redirect(url_for("register", role=role))

            if User.query.filter_by(
                role="patient", registration_number=reg
            ).first():
                flash("That registration number is already registered.", "error")
                return redirect(url_for("register", role=role))

            user.phone = phone
            user.registration_number = reg
            user.patient_id = generate_patient_id()
            user.verification_status = "not_applicable"

        else:
            med = request.form.get(
                "medical_registration_number", ""
            ).strip()

            if not med:
                flash("Please enter your medical registration number.", "error")
                return redirect(url_for("register", role=role))

            if User.query.filter_by(
                role="doctor", medical_registration_number=med
            ).first():
                flash("That medical registration number is already registered.", "error")
                return redirect(url_for("register", role=role))

            user.medical_registration_number = med
            user.verification_status = "unverified"

        try:
            db.session.add(user)
            db.session.commit()

            session.clear()
            session["user_id"] = user.id
            get_csrf_token()

            flash("Your account has been created successfully.", "success")
            return redirect(url_for("dashboard"))

        except Exception:
            db.session.rollback()
            app.logger.exception("Account registration failed")
            flash("Unable to create your account. Please try again.", "error")

    return render_template("auth.html", mode="register", role=role)


@app.route("/login/<role>", methods=["GET", "POST"])
def login(role):
    if role not in ("patient", "doctor"):
        return redirect(url_for("account"))

    if request.method == "POST":
        if not validate_csrf():
            flash("Your form session expired. Please try again.", "error")
            return redirect(url_for("login", role=role))

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email, role=role).first()

        if not user or not user.check_password(password):
            flash("Invalid email or password.", "error")
            return redirect(url_for("login", role=role))

        session.clear()
        session["user_id"] = user.id
        get_csrf_token()

        return redirect(url_for("dashboard"))

    return render_template("auth.html", mode="login", role=role)


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    if not validate_csrf():
        abort(400)

    session.clear()
    flash("You have been logged out.", "success")

    return redirect(url_for("account"))


@app.route("/dashboard")
@login_required
def dashboard():
    if g.current_user.role == "doctor":
        return render_template(
            "dashboard.html",
            user=g.current_user,
            screenings=[]
        )

    screenings = Screening.query.filter_by(
        patient_id=g.current_user.id
    ).order_by(Screening.created_at.desc()).all()

    reviews = (
        db.session.query(DoctorReview, Screening, User)
        .join(Screening, DoctorReview.screening_id == Screening.id)
        .join(User, Screening.patient_id == User.id)
        .filter(User.id == g.current_user.id)
        .order_by(DoctorReview.created_at.desc())
        .all()
    )

    return render_template(
        "dashboard.html",
        user=g.current_user,
        screenings=screenings,
        reviews=reviews
    )


# -------------------- SCREENING ROUTES --------------------

@app.route("/")
def home():
    if not g.current_user:
        return redirect(url_for("account"))

    if g.current_user.role != "patient":
        return redirect(url_for("dashboard"))

    return render_template("index.html")


def get_owned_screening(screening_id):
    screening = db.session.get(Screening, screening_id)

    if not screening:
        abort(404)

    if (
        g.current_user.role != "patient" or
        screening.patient_id != g.current_user.id
    ):
        abort(403)

    return screening


def show_classification(screening, stages):
    language = session.get("language", "en")

    translated_levels = {
        grade: translate_dr_level(language, grade)
        for grade in DR_LEVELS
    }

    translated_stages = [
        {
            **stage,
            "name": translate_stage(language, stage["name"]),
            "message": translate_stage(language, stage["message"])
        }
        for stage in stages
    ]

    return render_template(
        "classification.html",
        filename=screening.image_filename,
        image_url=url_for(
            "static",
            filename=f"uploads/{screening.image_filename}"
        ),
        heatmap_url=url_for(
            "screening_heatmap",
            screening_id=screening.id
        ),
        grade=screening.grade,
        levels=translated_levels,
        stages=translated_stages,
        screening=screening,
        report_id=f"RET-{screening.id:06d}",
        report_url=url_for(
            "download_report",
            screening_id=screening.id
        ),
        demo_mode=True
    )


@app.route("/analyze", methods=["POST"])
@role_required("patient")
def analyze():
    file = request.files.get("retinal_image")
    valid, message = validate_file(file)

    if not valid:
        flash(message, "error")
        return redirect(url_for("home"))

    extension = secure_filename(file.filename).rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{extension}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(filepath)

        success, stages = verify_all_stages(filepath)

        if not success:
            os.remove(filepath)

            return render_template(
                "result.html",
                success=False,
                message="The uploaded image did not pass all verification stages.",
                stages=stages
            )

        screening = Screening(
            patient_id=g.current_user.id,
            image_filename=filename,
            grade=random.randint(0, 4),
            analysis_mode="simulated"
        )

        db.session.add(screening)
        db.session.commit()

        return show_classification(screening, stages)

    except Exception:
        db.session.rollback()

        if os.path.exists(filepath):
            os.remove(filepath)

        app.logger.exception("Screening processing failed")
        flash("An error occurred while processing your image.", "error")

        return redirect(url_for("home"))


@app.route("/screening/<int:screening_id>")
@role_required("patient")
def view_screening(screening_id):
    screening = get_owned_screening(screening_id)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        screening.image_filename
    )

    if not os.path.isfile(filepath):
        flash("The image file is no longer available on this server.", "error")
        return redirect(url_for("dashboard"))

    _, stages = verify_all_stages(filepath)

    return show_classification(screening, stages)


@app.route("/heatmap/<int:screening_id>")
@role_required("patient")
def screening_heatmap(screening_id):
    screening = get_owned_screening(screening_id)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        screening.image_filename
    )

    if not os.path.isfile(filepath):
        abort(404)

    try:
        heatmap = generate_simulated_heatmap(
            filepath,
            seed=screening.image_filename
        )

        return send_file(
            heatmap,
            mimetype="image/png",
            as_attachment=False,
            download_name=f"RetinaAI_Heatmap_{screening.id}.png"
        )

    except Exception:
        app.logger.exception("Heatmap generation failed")
        abort(500)


@app.route("/demo-result", methods=["POST"])
@role_required("patient")
def demo_result():
    if not validate_csrf():
        abort(400)

    try:
        screening_id = int(request.form.get("screening_id", ""))
        grade = int(request.form.get("grade", ""))
    except (ValueError, TypeError):
        abort(400)

    if grade not in DR_LEVELS:
        abort(400)

    screening = get_owned_screening(screening_id)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        screening.image_filename
    )

    if not os.path.isfile(filepath):
        flash("The image file is no longer available.", "error")
        return redirect(url_for("dashboard"))

    success, stages = verify_all_stages(filepath)

    if not success:
        flash("The image no longer passes the demonstration checks.", "error")
        return redirect(url_for("dashboard"))

    screening.grade = grade
    db.session.commit()

    return show_classification(screening, stages)


# -------------------- PDF REPORT GENERATION --------------------

def build_screening_pdf(screening, patient):
    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        screening.image_filename
    )

    if not os.path.isfile(filepath):
        return None

    level = DR_LEVELS[screening.grade]
    report_ref = f"RET-{screening.id:06d}"
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#1769AA"),
        fontSize=19,
        spaceAfter=12
    )

    note_style = ParagraphStyle(
        "ReportNote",
        parent=styles["Normal"],
        textColor=colors.HexColor("#555555"),
        fontSize=9,
        leading=14
    )

    story = [
        Paragraph("RETINAAI - RETINAL IMAGE ANALYSIS", title_style),
        Paragraph(
            "Simulated Educational Demonstration Report",
            styles["Heading2"]
        ),
        Spacer(1, 10)
    ]

    details = [
        ["Patient Name", str(patient.full_name)],
        ["Patient ID", str(patient.patient_id or "N/A")],
        ["Report Reference", report_ref],
        [
            "Screening Date",
            screening.created_at.strftime("%d-%m-%Y %H:%M UTC")
        ],
        ["Analysis Mode", "SIMULATED DEMONSTRATION"],
        ["Selected Grade", str(screening.grade)],
        ["Classification", level["name"]],
        ["Severity Label", level["severity"]]
    ]

    table = Table(details, colWidths=[45 * mm, 105 * mm])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2FA")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#174A75")),
        ("GRID", (0, 0), (-1, -1), .5, colors.HexColor("#D5DEE8")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    story.extend([
        table,
        Spacer(1, 15),
        Paragraph("Uploaded Image", styles["Heading2"])
    ])

    try:
        story.append(
            PDFImage(
                filepath,
                width=100 * mm,
                height=75 * mm,
                kind="proportional"
            )
        )
        story.append(Spacer(1, 15))

    except Exception:
        story.append(
            Paragraph("Image preview unavailable.", styles["Normal"])
        )

    # Add the same simulated heatmap to the PDF.
    try:
        heatmap_buffer = generate_simulated_heatmap(
            filepath,
            seed=screening.image_filename
        )

        story.extend([
            Paragraph(
                "Simulated Heatmap",
                styles["Heading2"]
            ),
            Paragraph(
                "Blue, green, yellow and red colours form an illustrative "
                "overlay only. This image is not generated by a trained AI "
                "model or Grad-CAM and does not identify disease regions.",
                note_style
            ),
            Spacer(1, 8),
            PDFImage(
                heatmap_buffer,
                width=100 * mm,
                height=75 * mm,
                kind="proportional"
            ),
            Spacer(1, 15)
        ])

    except Exception:
        app.logger.exception("PDF heatmap generation failed")
        story.append(
            Paragraph(
                "Heatmap preview unavailable.",
                styles["Normal"]
            )
        )

    story.extend([
        Paragraph("Important Disclaimer", styles["Heading2"]),
        Paragraph(
            "This is a simulated academic demonstration. The displayed "
            "grade is randomly generated or manually selected; no trained "
            "AI model is used. The heatmap is an illustrative colour overlay "
            "and is not a genuine AI explanation or disease-region map. "
            "Image checks cannot reliably confirm that an image is a retinal "
            "photograph. This report is not a medical diagnosis and must not "
            "be used for clinical decisions or treatment.",
            note_style
        )
    ])

    doc.build(story)
    buffer.seek(0)

    return buffer, report_ref


@app.route("/download-report/<int:screening_id>")
@login_required
def download_report(screening_id):
    screening = db.session.get(Screening, screening_id)

    if not screening:
        abort(404)

    if g.current_user.role == "patient":
        if screening.patient_id != g.current_user.id:
            abort(403)

        patient = g.current_user

    elif g.current_user.role == "doctor":
        patient = db.session.get(User, screening.patient_id)

        if not patient or patient.role != "patient":
            abort(404)

    else:
        abort(403)

    result = build_screening_pdf(screening, patient)

    if result is None:
        flash("The image file is no longer available on this server.", "error")
        return redirect(url_for("dashboard"))

    buffer, report_ref = result

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"RetinaAI_{report_ref}.pdf",
        mimetype="application/pdf"
    )


# -------------------- DOCTOR PATIENT SEARCH --------------------

@app.route("/doctor/patients", methods=["POST"])
@role_required("doctor")
def doctor_search():
    if not validate_csrf():
        abort(400)

    patient_id = request.form.get("patient_id", "").strip().upper()

    if not patient_id:
        flash("Enter a Patient ID to search.", "error")
        return redirect(url_for("dashboard"))

    patient = User.query.filter_by(
        patient_id=patient_id,
        role="patient"
    ).first()

    if not patient:
        flash("No patient was found with that Patient ID.", "error")
        return redirect(url_for("dashboard"))

    return redirect(
        url_for("doctor_patient", patient_id=patient.patient_id)
    )


@app.route("/doctor/patient/<patient_id>")
@role_required("doctor")
def doctor_patient(patient_id):
    patient = User.query.filter_by(
        patient_id=patient_id.upper(),
        role="patient"
    ).first_or_404()

    screenings = Screening.query.filter_by(
        patient_id=patient.id
    ).order_by(Screening.created_at.desc()).all()

    review_rows = (
        db.session.query(DoctorReview, Screening)
        .join(Screening, DoctorReview.screening_id == Screening.id)
        .filter(Screening.patient_id == patient.id)
        .order_by(DoctorReview.created_at.desc())
        .all()
    )

    reviews_by_screening = {}

    for review, screening in review_rows:
        reviews_by_screening.setdefault(
            screening.id, []
        ).append(review)

    return render_template(
        "doctor_patient.html",
        patient=patient,
        screenings=screenings,
        reviews_by_screening=reviews_by_screening,
        levels={
            grade: translate_dr_level(
                session.get("language", "en"),
                grade
            )
            for grade in DR_LEVELS
        }
    )


@app.route("/doctor/screening/<int:screening_id>")
@role_required("doctor")
def doctor_screening(screening_id):
    screening = db.session.get(Screening, screening_id)

    if not screening:
        abort(404)

    patient = db.session.get(User, screening.patient_id)

    if not patient or patient.role != "patient":
        abort(404)

    reviews = DoctorReview.query.filter_by(
        screening_id=screening.id
    ).order_by(DoctorReview.created_at.desc()).all()

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        screening.image_filename
    )

    image_available = os.path.isfile(filepath)

    return render_template(
        "doctor_screening.html",
        screening=screening,
        patient=patient,
        reviews=reviews,
        level=translate_dr_level(
            session.get("language", "en"),
            screening.grade
        ),
        image_available=image_available
    )


@app.route("/doctor/image/<int:screening_id>")
@role_required("doctor")
def doctor_screening_image(screening_id):
    screening = db.session.get(Screening, screening_id)

    if not screening:
        abort(404)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        screening.image_filename
    )

    if not os.path.isfile(filepath):
        abort(404)

    try:
        with Image.open(filepath) as image:
            image.verify()

        with Image.open(filepath) as image:
            mime = Image.MIME.get(image.format)

    except Exception:
        abort(404)

    if mime not in ("image/jpeg", "image/png"):
        abort(404)

    return send_file(
        filepath,
        mimetype=mime,
        as_attachment=False
    )


@app.route(
    "/doctor/screening/<int:screening_id>/review",
    methods=["POST"]
)
@role_required("doctor")
def doctor_save_review(screening_id):
    if not validate_csrf():
        abort(400)

    screening = db.session.get(Screening, screening_id)

    if not screening:
        abort(404)

    patient = db.session.get(User, screening.patient_id)

    if not patient or patient.role != "patient":
        abort(404)

    classification_comment = request.form.get(
        "classification_comment", ""
    ).strip()

    heatmap_comment = request.form.get(
        "heatmap_comment", ""
    ).strip()

    progress_comment = request.form.get(
        "progress_comment", ""
    ).strip()

    overall_advice = request.form.get(
        "overall_advice", ""
    ).strip()

    fields = [
        classification_comment,
        heatmap_comment,
        progress_comment,
        overall_advice
    ]

    if not any(fields):
        flash("Enter at least one comment before saving.", "error")
        return redirect(
            url_for("doctor_screening", screening_id=screening.id)
        )

    if any(len(value) > 5000 for value in fields):
        flash("Each review field must be 5,000 characters or fewer.", "error")
        return redirect(
            url_for("doctor_screening", screening_id=screening.id)
        )

    review = DoctorReview(
        screening_id=screening.id,
        doctor_id=g.current_user.id,
        classification_comment=classification_comment,
        heatmap_comment=heatmap_comment,
        progress_comment=progress_comment,
        overall_advice=overall_advice
    )

    try:
        db.session.add(review)
        db.session.commit()
        flash("Review saved successfully.", "success")

    except Exception:
        db.session.rollback()
        app.logger.exception("Unable to save doctor review")
        flash("Unable to save the review. Please try again.", "error")

    return redirect(
        url_for("doctor_screening", screening_id=screening.id)
    )


# -------------------- ERROR HANDLERS --------------------

@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "result.html",
        success=False,
        message="The uploaded file is too large. Maximum allowed size is 10 MB.",
        stages=[]
    ), 413


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
