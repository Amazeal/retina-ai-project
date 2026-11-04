
import os
import random
import re
import uuid
from datetime import datetime
from io import BytesIO

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file
)
from werkzeug.utils import secure_filename

from PIL import Image, ImageStat

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle
)


# --------------------------------------------------
# FLASK CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = os.path.join(
    "static",
    "uploads"
)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


# --------------------------------------------------
# FIVE DR SEVERITY LEVELS
# --------------------------------------------------

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


# --------------------------------------------------
# FILE VALIDATION
# --------------------------------------------------

def allowed_file(filename):
    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return extension in ALLOWED_EXTENSIONS


def validate_file(file):
    if file is None:
        return False, "No image was uploaded."

    if file.filename == "":
        return False, "Please select an image."

    if not allowed_file(file.filename):
        return False, "Only JPG, JPEG and PNG images are allowed."

    return True, "File format verified successfully."


# --------------------------------------------------
# IMAGE QUALITY CHECK
# --------------------------------------------------

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


# --------------------------------------------------
# BASIC RETINAL IMAGE DEMONSTRATION CHECK
# --------------------------------------------------

def is_retinal_image_demo(filepath):
    """
    This is only a simple image-based demonstration check.

    It is NOT a trained retinal image classifier.
    It cannot reliably determine whether an image is
    actually a retinal fundus photograph.
    """

    try:
        with Image.open(filepath) as image:
            image = image.convert("RGB")

            width, height = image.size

            if width == 0 or height == 0:
                return False, "Unable to analyse the image dimensions."

            aspect_ratio = width / height

            # Broad shape check
            if not 0.5 <= aspect_ratio <= 2.0:
                return False, "The image shape is not suitable for this demo."

            # Compare the centre and corner brightness.
            # This is a weak visual heuristic, not AI.
            grayscale = image.convert("L")

            centre_box = (
                int(width * 0.25),
                int(height * 0.25),
                int(width * 0.75),
                int(height * 0.75)
            )

            centre = grayscale.crop(centre_box)

            corner_size = max(1, min(width, height) // 10)

            corners = [
                grayscale.crop((0, 0, corner_size, corner_size)),
                grayscale.crop((width - corner_size, 0, width, corner_size)),
                grayscale.crop((0, height - corner_size, corner_size, height)),
                grayscale.crop((
                    width - corner_size,
                    height - corner_size,
                    width,
                    height
                ))
            ]

            centre_brightness = ImageStat.Stat(centre).mean[0]

            corner_brightness = sum(
                ImageStat.Stat(corner).mean[0]
                for corner in corners
            ) / len(corners)

            if centre_brightness < 10:
                return False, "The image appears too dark."

            if centre_brightness < corner_brightness * 0.65:
                return False, "The image did not pass the basic visual check."

            return True, "The image passed the basic demonstration check."

    except Exception:
        return False, "Unable to analyse the uploaded image."


# --------------------------------------------------
# VERIFICATION STAGES
# --------------------------------------------------

def create_stages(file_status, quality_status, retinal_status):
    return [
        {
            "name": "File Validation",
            "status": "passed" if file_status[0] else "failed",
            "message": file_status[1]
        },
        {
            "name": "Image Quality Check",
            "status": "passed" if quality_status[0] else "failed",
            "message": quality_status[1]
        },
        {
            "name": "Retinal Image Demonstration Check",
            "status": "passed" if retinal_status[0] else "failed",
            "message": retinal_status[1]
        }
    ]


def verify_all_stages(filepath):
    quality_status = verify_image_quality(filepath)

    if not quality_status[0]:
        retinal_status = (
            False,
            "Skipped because the image quality check failed."
        )

        file_status = (True, "File format verified successfully.")

        stages = create_stages(
            file_status,
            quality_status,
            retinal_status
        )

        return False, stages

    retinal_status = is_retinal_image_demo(filepath)

    file_status = (True, "File format verified successfully.")

    stages = create_stages(
        file_status,
        quality_status,
        retinal_status
    )

    success = quality_status[0] and retinal_status[0]

    return success, stages


# --------------------------------------------------
# DISPLAY CLASSIFICATION PAGE
# --------------------------------------------------

def show_classification(filename, grade, stages):
    report_id = uuid.uuid4().hex

    image_url = url_for(
        "static",
        filename=f"uploads/{filename}"
    )

    report_url = url_for(
        "download_report",
        filename=filename,
        grade=grade,
        report_id=report_id
    )

    return render_template(
        "classification.html",
        filename=filename,
        image_url=image_url,
        grade=grade,
        levels=DR_LEVELS,
        stages=stages,
        report_id=report_id,
        report_url=report_url,
        demo_mode=True
    )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# UPLOAD AND ANALYSE IMAGE
# --------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():

    if "retinal_image" not in request.files:
        return render_template(
            "result.html",
            success=False,
            message="No image was uploaded.",
            stages=[]
        )

    file = request.files["retinal_image"]

    valid, message = validate_file(file)

    if not valid:
        return render_template(
            "result.html",
            success=False,
            message=message,
            stages=[]
        )

    original_name = secure_filename(file.filename)

    extension = original_name.rsplit(".", 1)[1].lower()

    unique_filename = f"{uuid.uuid4().hex}.{extension}"

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_filename
    )

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

        # Random result for demonstration only.
        # This is NOT an AI prediction.
        grade = random.randint(0, 4)

        return show_classification(
            unique_filename,
            grade,
            stages
        )

    except Exception as error:
        if os.path.exists(filepath):
            os.remove(filepath)

        app.logger.exception("Image processing error")

        return render_template(
            "result.html",
            success=False,
            message="An error occurred while processing the image.",
            stages=[]
        )


# --------------------------------------------------
# DEMO MODE - MANUAL CLASSIFICATION SELECTION
# --------------------------------------------------

@app.route("/demo-result", methods=["POST"])
def demo_result():

    filename = request.form.get("filename", "")
    grade_value = request.form.get("grade", "")

    # Prevent directory traversal
    filename = os.path.basename(filename)

    if not filename or not allowed_file(filename):
        return redirect(url_for("home"))

    try:
        grade = int(grade_value)

        if grade not in DR_LEVELS:
            return redirect(url_for("home"))

    except (ValueError, TypeError):
        return redirect(url_for("home"))

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.isfile(filepath):
        return redirect(url_for("home"))

    success, stages = verify_all_stages(filepath)

    if not success:
        return render_template(
            "result.html",
            success=False,
            message="The image did not pass verification.",
            stages=stages
        )

    # The selected grade is manually chosen for the demo.
    # It is not generated by a trained AI model.
    return show_classification(
        filename,
        grade,
        stages
    )


# --------------------------------------------------
# DOWNLOAD PDF REPORT
# --------------------------------------------------

@app.route(
    "/download-report/<filename>/<int:grade>/<report_id>"
)
def download_report(filename, grade, report_id):

    filename = os.path.basename(filename)

    if not allowed_file(filename):
        return redirect(url_for("home"))

    if grade not in DR_LEVELS:
        return redirect(url_for("home"))

    if not re.fullmatch(r"[a-f0-9]{32}", report_id):
        return redirect(url_for("home"))

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.isfile(filepath):
        return redirect(url_for("home"))

    level = DR_LEVELS[grade]

    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
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
        fontSize=20,
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        textColor=colors.HexColor("#555555"),
        fontSize=9,
        leading=14
    )

    story = []

    story.append(
        Paragraph(
            "RETINAAI - RETINAL IMAGE ANALYSIS",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Diabetic Retinopathy Demonstration Report",
            styles["Heading2"]
        )
    )

    story.append(Spacer(1, 8))

    report_data = [
        ["Report Reference", report_id[:12].upper()],
        ["Date", datetime.now().strftime("%d-%m-%Y %H:%M")],
        ["Image File", filename],
        ["Analysis Mode", "SIMULATED DEMONSTRATION"],
        ["Selected Grade", str(grade)],
        ["Classification", level["name"]],
        ["Severity", level["severity"]]
    ]

    report_table = Table(
        report_data,
        colWidths=[45 * mm, 105 * mm]
    )

    report_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2FA")),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#174A75")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D5DEE8")),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
        ])
    )

    story.append(report_table)
    story.append(Spacer(1, 18))

    story.append(
        Paragraph(
            "Uploaded Retinal Image",
            styles["Heading2"]
        )
    )

    try:
        retinal_image = PDFImage(
            filepath,
            width=100 * mm,
            height=75 * mm,
            kind="proportional"
        )

        story.append(retinal_image)
        story.append(Spacer(1, 15))

    except Exception:
        story.append(
            Paragraph(
                "Image preview could not be added to this report.",
                styles["Normal"]
            )
        )

    story.append(
        Paragraph(
            "Five-Level Classification Scale",
            styles["Heading2"]
        )
    )

    scale_data = [
        ["Grade", "Classification"]
    ]

    for level_number, level_info in DR_LEVELS.items():
        label = level_info["name"]

        if level_number == grade:
            label = "SELECTED: " + label

        scale_data.append([
            str(level_number),
            label
        ])

    scale_table = Table(
        scale_data,
        colWidths=[25 * mm, 125 * mm]
    )

    scale_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1769AA")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D5DEE8")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 7),
            ("BACKGROUND", (0, grade + 1), (-1, grade + 1),
             colors.HexColor("#DCEFE5")),
            ("TEXTCOLOR", (0, grade + 1), (-1, grade + 1),
             colors.HexColor("#17643B"))
        ])
    )

    story.append(scale_table)
    story.append(Spacer(1, 18))

    story.append(
        Paragraph(
            "Important Disclaimer",
            styles["Heading2"]
        )
    )

    disclaimer = (
        "This project is a simulated educational demonstration. "
        "The classification shown in this report is randomly generated "
        "or manually selected for demonstration purposes. "
        "The application does not use a trained artificial intelligence "
        "model and does not provide a medical diagnosis. "
        "This report must not be used for clinical decisions, "
        "medical treatment, or patient care."
    )

    story.append(
        Paragraph(
            disclaimer,
            subtitle_style
        )
    )

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "Generated by RetinaAI - Academic Mini Project",
            subtitle_style
        )
    )

    document.build(story)

    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name="RetinaAI_Report.pdf",
        mimetype="application/pdf"
    )


# --------------------------------------------------
# FILE TOO LARGE ERROR
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "result.html",
        success=False,
        message="The uploaded file is too large. Maximum allowed size is 10 MB.",
        stages=[]
    ), 413


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
