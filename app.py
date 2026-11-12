"""
AsphaltAi - Step 1 build
Upload a road photo -> YOLO finds potholes -> save a geotagged report -> see it on a Bengaluru map.

Run with:  streamlit run app.py
"""

import html
import json
import random
import sqlite3
import time
import urllib.parse
import urllib.request
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from streamlit_folium import st_folium

# ----------------------------------------------------------------------
# 1. SETTINGS - everything you might want to change lives here
# ----------------------------------------------------------------------
APP_NAME = "AsphaltAi"  # change the project name here; it sets the browser tab and the page heading
st.set_page_config(page_title=APP_NAME, page_icon="🕳️", layout="wide")

DB_PATH = Path("potholeiq.db")              # SQLite database file (created automatically)
UPLOAD_DIR = Path("data/uploads")           # original photos of saved reports
ANNOTATED_DIR = Path("data/annotated")      # photos with boxes drawn on them
LOCAL_WEIGHTS = Path("models/best.pt")      # model file; downloaded here on first run

# Free, public pothole model on Hugging Face (no account or API key needed).
# To try a different model later, change these two lines.
HF_REPO = "Samdutse/pothole-yolov8"
HF_FILE = "best.pt"

# There is no "how sure" cutoff. Every pothole the model finds is shown with its own
# confidence, whether that is 30%, 50% or 100%. Only detections below NOISE_FLOOR are
# ignored, because below about 10% they are almost always random noise. Set it to 0.01
# to see everything the model produces (a clean road may then show several junk boxes).
# Set ONLY_MOST_CONFIDENT = True to keep just the single most confident detection.
NOISE_FLOOR = 0.10
ONLY_MOST_CONFIDENT = False

BENGALURU_CENTER = (12.9716, 77.5946)
BENGALURU_BOUNDS = (12.7, 13.3, 77.3, 77.9)   # south, north, west, east limits for saved reports

# OpenStreetMap's free address lookup (no key) asks every app to identify itself.
# Put your own email or GitHub link in the brackets before you share the project.
NOMINATIM_USER_AGENT = "AsphaltAi-student-project/1.0 (add-your-email-or-github-here)"
SEVERITY_COLORS = {"Low": "#2e9e4f", "Medium": "#f39c12", "High": "#d62728", "Unclear": "#7f7f7f"}

# Severity rules (explained in estimate_severity). Sizes are the share of the PHOTO a pothole covers.
SURE_ENOUGH = 0.50      # detections at or above this count as confirmed potholes
MODERATE_SHARE = 0.03   # a pothole covering 3% or more of the photo counts as moderate in size
LARGE_SHARE = 0.10      # 10% or more counts as large
MANY_POTHOLES = 4       # this many confirmed potholes in one photo counts as High
IMAGE_SIZE = 640        # model input size: smaller is faster but may miss small potholes

for folder in (UPLOAD_DIR, ANNOTATED_DIR, LOCAL_WEIGHTS.parent):
    folder.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 2. DEMO DATA - fake reports placed at real Bengaluru localities so the
#    map and dashboard are usable the moment you open the app.
#    (Three spots have 3 reports each, so "repeated reports" shows up.)
# ----------------------------------------------------------------------
DEMO_SPOTS = [
    ("Silk Board Junction", 12.9177, 77.6238, 3),
    ("Hebbal Flyover", 13.0358, 77.5970, 3),
    ("Marathahalli Bridge", 12.9591, 77.6974, 3),
    ("KR Puram Hanging Bridge", 13.0070, 77.6960, 1),
    ("Electronic City Phase 1", 12.8452, 77.6602, 1),
    ("Majestic (KG Road)", 12.9767, 77.5713, 1),
    ("Indiranagar 100 Feet Road", 12.9784, 77.6408, 1),
    ("Koramangala 80 Feet Road", 12.9352, 77.6245, 1),
    ("Jayanagar 4th Block", 12.9308, 77.5838, 1),
    ("Yeshwanthpur Circle", 13.0285, 77.5409, 1),
    ("Whitefield Main Road", 12.9698, 77.7500, 1),
    ("Bannerghatta Road", 12.8950, 77.5990, 1),
    ("HSR Layout Sector 2", 12.9116, 77.6389, 1),
    ("BTM 2nd Stage", 12.9166, 77.6101, 1),
    ("Banashankari Bus Stand", 12.9255, 77.5468, 1),
    ("Malleshwaram 8th Cross", 13.0035, 77.5647, 1),
    ("Rajajinagar Industrial Town", 12.9910, 77.5520, 1),
    ("Yelahanka New Town", 13.1007, 77.5963, 1),
    ("Sarjapur Road", 12.9100, 77.6870, 1),
    ("Bellandur Outer Ring Road", 12.9304, 77.6784, 1),
    ("Hennur Main Road", 13.0358, 77.6400, 1),
    ("Domlur Flyover", 12.9609, 77.6387, 1),
    ("Basavanagudi Bull Temple Road", 12.9422, 77.5750, 1),
    ("JP Nagar 6th Phase", 12.9063, 77.5857, 1),
    ("Peenya Industrial Area", 13.0285, 77.5197, 1),
]

DEMO_NOTES = [
    "Deep pothole near the signal, two-wheelers swerving around it",
    "Multiple potholes after the recent rain",
    "Road edge broken near the bus stop",
    "Large pothole in the left lane",
    "Patchwork washed away, potholes reappearing",
    "Water-filled pothole, hard to see at night",
    "Cluster of potholes near the flyover ramp",
]


# ----------------------------------------------------------------------
# 3. DATABASE - a tiny SQLite file that stores every report
# ----------------------------------------------------------------------
def db_execute(sql, params=()):
    """Run one write statement (INSERT / DELETE / CREATE)."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(sql, params)
        conn.commit()


def db_read(sql, params=()):
    """Run a SELECT and return the result as a pandas table."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def seed_demo_data():
    rng = random.Random(42)  # fixed seed -> same demo data every time
    now = datetime.now()
    rows = []
    for locality, lat, lon, n_reports in DEMO_SPOTS:
        for _ in range(n_reports):
            severity = rng.choices(["Low", "Medium", "High"], weights=[4, 4, 3])[0]
            count = {"Low": 1, "Medium": rng.randint(2, 3), "High": rng.randint(4, 6)}[severity]
            created = now - timedelta(days=rng.randint(0, 30), hours=rng.randint(0, 23))
            rows.append((
                created.strftime("%Y-%m-%d %H:%M"),
                lat + rng.uniform(-0.0006, 0.0006),
                lon + rng.uniform(-0.0006, 0.0006),
                locality,
                rng.choice(DEMO_NOTES),
                count,
                round(rng.uniform(0.45, 0.92), 2),
                severity,
                None,
                None,
            ))
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.executemany(
            """INSERT INTO reports
               (created_at, latitude, longitude, locality, description,
                pothole_count, max_confidence, severity, image_path, annotated_path)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        conn.commit()


def init_db():
    db_execute(
        """CREATE TABLE IF NOT EXISTS reports (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               created_at TEXT NOT NULL,
               latitude REAL NOT NULL,
               longitude REAL NOT NULL,
               locality TEXT,
               description TEXT,
               pothole_count INTEGER NOT NULL,
               max_confidence REAL,
               severity TEXT NOT NULL,
               image_path TEXT,
               annotated_path TEXT
           )"""
    )
    if db_read("SELECT COUNT(*) AS n FROM reports")["n"].iloc[0] == 0:
        seed_demo_data()


# ----------------------------------------------------------------------
# 4. THE AI MODEL
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading pothole model (first run downloads it)...")
def _load_model():
    """Load YOLO once and keep it in memory. Downloads the weights if missing."""
    from ultralytics import YOLO

    if not LOCAL_WEIGHTS.exists():
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, local_dir=str(LOCAL_WEIGHTS.parent))
    model = YOLO(str(LOCAL_WEIGHTS))
    # One practice run now, so the first real photo does not pay the slow start-up cost.
    model.predict(Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE)), imgsz=IMAGE_SIZE, verbose=False)
    return model


def get_model():
    """Returns (model, None) on success or (None, error_text) so the app never crashes."""
    try:
        return _load_model(), None
    except Exception as exc:  # noqa: BLE001 - we want to show any failure nicely
        return None, str(exc)


def draw_detections(image, boxes, confs):
    """Return a copy of the photo with a blue box and the confidence on every detection."""
    picture = image.copy()
    draw = ImageDraw.Draw(picture)
    line = max(2, picture.width // 250)
    size = max(14, picture.width // 40)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow without a font size option
        font = ImageFont.load_default()
    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        draw.rectangle([x1, y1, x2, y2], outline="#1f3fff", width=line)
        draw.text((x1 + 3, max(0, y1 - size - 2)), f"Pothole {conf:.2f}", fill="white",
                  font=font, stroke_width=2, stroke_fill="#1f3fff")
    return picture


def run_detection(model, image, conf):
    """
    Run YOLO on one image. Returns boxes, confidences, a photo with ALL detections drawn,
    time in ms, and a photo with only the confirmed detections (SURE_ENOUGH or more) drawn.
    """
    start = time.perf_counter()
    result = model.predict(image, conf=conf, imgsz=IMAGE_SIZE, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000
    all_confs = result.boxes.conf.tolist()
    if ONLY_MOST_CONFIDENT and len(all_confs) > 1:
        best = max(range(len(all_confs)), key=lambda i: all_confs[i])
        result = result[[best]]           # keep only the detection the model is most sure about
    boxes = result.boxes.xyxy.tolist()    # [x1, y1, x2, y2] for each pothole
    confs = result.boxes.conf.tolist()    # how sure the model is (0 to 1)
    strong = [i for i, c in enumerate(confs) if c >= SURE_ENOUGH]
    annotated_all = draw_detections(image, boxes, confs)
    annotated_strong = draw_detections(image, [boxes[i] for i in strong], [confs[i] for i in strong])
    return boxes, confs, annotated_all, elapsed_ms, annotated_strong


def estimate_severity(boxes, confs, image_w, image_h):
    """
    Rule-based severity estimate. Returns a dict: label, reasons (plain-English list),
    reliability, and how many potholes were confirmed.

    What it uses
      - Size: the share of the PHOTO each pothole covers (a stand-in for real size).
      - Number: how many potholes are in the photo.
    What it deliberately does NOT use
      - Confidence. A pothole the model is 95% sure about can still be small. Confidence
        only decides which detections count (SURE_ENOUGH) and how far to trust the estimate.
      - Depth. A flat photo cannot show how deep a hole is, so depth is never claimed.
    Known limits
      - The same pothole looks bigger in a close-up photo than in a far-away one.
      - Road damage around the pothole (cracks, patches) is not considered yet.

    Rules, using only confirmed detections (confidence at or above SURE_ENOUGH)
      High    : any pothole covers LARGE_SHARE or more of the photo, OR two or more potholes
                are each at least moderate in size, OR there are MANY_POTHOLES or more.
      Medium  : any pothole is at least moderate in size, OR there are two or more potholes.
      Low     : a single small pothole.
      Unclear : nothing reaches SURE_ENOUGH, so a person should look at the photo.
    """
    photo_area = image_w * image_h
    confirmed = [(conf, (x2 - x1) * (y2 - y1) / photo_area)
                 for conf, (x1, y1, x2, y2) in zip(confs, boxes) if conf >= SURE_ENOUGH]

    if not confirmed:
        if not boxes:
            return {"label": "None", "reasons": ["No pothole was detected."],
                    "reliability": "-", "confirmed": 0}
        return {"label": "Unclear",
                "reasons": [f"The model's best detection is only {max(confs):.0%} sure, below the "
                            f"{SURE_ENOUGH:.0%} needed to judge severity. Check the photo yourself."],
                "reliability": "Low", "confirmed": 0}

    shares = [share for _, share in confirmed]
    largest = max(shares)
    count = len(confirmed)
    moderate_or_bigger = sum(1 for share in shares if share >= MODERATE_SHARE)

    reasons = [f"Confirmed potholes (model at least {SURE_ENOUGH:.0%} sure): {count}",
               f"The largest covers {largest:.1%} of the photo"]
    if largest >= LARGE_SHARE or moderate_or_bigger >= 2 or count >= MANY_POTHOLES:
        label = "High"
        if largest >= LARGE_SHARE:
            reasons.append(f"High: the largest pothole covers {LARGE_SHARE:.0%} or more of the photo")
        if moderate_or_bigger >= 2:
            reasons.append("High: two or more potholes are each at least moderate in size")
        if count >= MANY_POTHOLES:
            reasons.append(f"High: there are {MANY_POTHOLES} or more potholes in the photo")
    elif largest >= MODERATE_SHARE or count >= 2:
        label = "Medium"
        if largest >= MODERATE_SHARE:
            reasons.append(f"Medium: the largest pothole covers {MODERATE_SHARE:.0%} or more of the photo")
        if count >= 2:
            reasons.append("Medium: there are two or more potholes")
    else:
        label = "Low"
        reasons.append("Low: a single small pothole")

    weaker = len(boxes) - count
    if weaker:
        reasons.append(f"{weaker} weaker detection(s) below {SURE_ENOUGH:.0%} were not counted")

    reliability = "Good" if min(conf for conf, _ in confirmed) >= 0.60 else "Fair"
    return {"label": label, "reasons": reasons, "reliability": reliability, "confirmed": count}


def read_gps_from_exif(image):
    """Phone cameras can store GPS inside the photo. Returns (lat, lon) or None."""
    try:
        gps = image.getexif().get_ifd(0x8825)  # 0x8825 = the GPS section of EXIF
        if not gps or 2 not in gps or 4 not in gps:
            return None

        def to_degrees(value):
            d, m, s = (float(x) for x in value)
            return d + m / 60 + s / 3600

        lat, lon = to_degrees(gps[2]), to_degrees(gps[4])
        if gps.get(1) == "S":
            lat = -lat
        if gps.get(3) == "W":
            lon = -lon
        return lat, lon
    except Exception:  # noqa: BLE001
        return None


def short_locality_from_address(data):
    """Turn an OpenStreetMap reply into a short place name such as 'Kaveri Nagar'."""
    address = data.get("address", {})
    parts = [address.get("road"), address.get("neighbourhood") or address.get("suburb")]
    parts = [p for p in parts if p]
    if not parts and data.get("display_name"):
        parts = [p.strip() for p in data["display_name"].split(",")[:2]]
    return ", ".join(dict.fromkeys(parts)) or None   # dict.fromkeys drops repeated words


@st.cache_data(show_spinner=False, ttl=86400)
def lookup_locality(lat, lon):
    """
    Ask OpenStreetMap (free, no key) which place these coordinates are in. Only the
    coordinates are sent, never the photo. Returns a short place name, or None if the
    lookup fails (for example when there is no internet).
    """
    query = urllib.parse.urlencode({"format": "jsonv2", "lat": lat, "lon": lon,
                                    "zoom": 16, "addressdetails": 1})
    request = urllib.request.Request(
        f"https://nominatim.openstreetmap.org/reverse?{query}",
        headers={"User-Agent": NOMINATIM_USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=6) as response:
            return short_locality_from_address(json.load(response))
    except Exception:  # noqa: BLE001 - a failed lookup must never break the app
        return None


def save_report(image, annotated, lat, lon, locality, description, confs, boxes, severity):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = UPLOAD_DIR / f"{stamp}_original.jpg"
    ann_path = ANNOTATED_DIR / f"{stamp}_annotated.jpg"
    image.save(img_path, quality=90)
    annotated.save(ann_path, quality=90)
    db_execute(
        """INSERT INTO reports
           (created_at, latitude, longitude, locality, description,
            pothole_count, max_confidence, severity, image_path, annotated_path)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            lat,
            lon,
            locality.strip() or None,
            description.strip() or None,
            len([c for c in confs if c >= SURE_ENOUGH]) or len(boxes),
            round(max(confs), 2),
            severity,
            str(img_path),
            str(ann_path),
        ),
    )


# ----------------------------------------------------------------------
# 5. PAGE: DETECT & REPORT
# ----------------------------------------------------------------------
def page_detect():
    st.header("Detect potholes and file a report")

    model, load_error = get_model()
    if model is None:
        st.error("The pothole model could not be loaded, so detection is unavailable.")
        st.write(
            "Common fixes: check your internet connection (the model downloads on first run), "
            "or download `best.pt` manually from the Hugging Face page of the model and "
            "place it at `models/best.pt`, then refresh this page."
        )
        with st.expander("Technical details"):
            st.code(load_error)
        return

    st.sidebar.caption("Model classes: " + ", ".join(model.names.values()))
    conf = NOISE_FLOOR
    st.sidebar.caption(f"Every pothole the model finds is shown with its own confidence. "
                       f"Anything below {NOISE_FLOOR:.0%} is ignored as noise.")

    uploaded = st.file_uploader("Upload a road photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload a photo to begin.")
        return

    # Open the image safely - a corrupt file shows a message instead of crashing.
    try:
        original = Image.open(uploaded)
        gps = read_gps_from_exif(original)                 # read GPS before any editing
        image = ImageOps.exif_transpose(original).convert("RGB")  # fix phone rotation
    except Exception:  # noqa: BLE001
        st.error("This file could not be read as an image. Please upload a valid JPG or PNG.")
        return

    # Only re-run the AI when the photo or threshold changes (not when you type in a box).
    key = (uploaded.name, uploaded.size, conf, IMAGE_SIZE, "v2")
    if st.session_state.get("det_key") != key:
        with st.spinner("Detecting potholes..."):
            st.session_state["det"] = run_detection(model, image, conf)
        st.session_state["det_key"] = key
    boxes, confs, annotated_all, elapsed_ms, annotated_strong = st.session_state["det"]

    severity_info = estimate_severity(boxes, confs, *image.size)
    severity = severity_info["label"]

    # Confirmed = the model is at least SURE_ENOUGH sure. Weaker detections stay in the table.
    strong_count = sum(1 for c in confs if c >= SURE_ENOUGH)
    weaker = len(boxes) - strong_count
    show_weak = True   # with nothing confirmed, show whatever the model found
    if weaker and strong_count:
        show_weak = st.checkbox(f"Also draw the {weaker} weaker detection(s) "
                                f"(below {SURE_ENOUGH:.0%}) on the photo", value=False)
    annotated = annotated_all if show_weak else annotated_strong

    left, right = st.columns(2)
    left.image(image, caption="Original photo")
    right.image(annotated, caption="Detections")

    m1, m2, m3, m4 = st.columns(4)
    low_confidence = bool(confs) and max(confs) < SURE_ENOUGH   # the model is not very sure
    if strong_count:
        m1.metric("Confirmed potholes", strong_count)
    else:
        m1.metric("Possible pothole (unsure)" if boxes else "Potholes detected", len(boxes))
    m2.metric("Highest confidence", f"{max(confs):.0%}" if confs else "-")
    m3.metric("Severity (estimate)", severity)
    m4.metric("Inference time", f"{elapsed_ms:.0f} ms")
    if weaker and strong_count:
        st.caption(f"Plus {weaker} weaker detection(s) below {SURE_ENOUGH:.0%}. They are listed "
                   "in the table below but not counted.")

    if not boxes:
        st.info("The model did not find any pothole in this photo, so there is nothing to report.")
        return

    photo_area = image.size[0] * image.size[1]
    detections = pd.DataFrame({
        "Pothole": list(range(1, len(boxes) + 1)),
        "Confidence": [f"{c:.0%}" for c in confs],
        "Share of photo": [f"{(x2 - x1) * (y2 - y1) / photo_area:.1%}" for x1, y1, x2, y2 in boxes],
    })
    st.dataframe(detections, hide_index=True)

    with st.expander("How was this severity decided?"):
        for reason in severity_info["reasons"]:
            st.write("- " + reason)
        st.caption(f"Estimate reliability: {severity_info['reliability']} "
                   "(Good = every counted detection is at least 60% sure).")
        st.caption("Severity is a rough estimate from a flat photo. It cannot measure depth, and a "
                   "close-up makes any pothole look bigger. Confidence says how sure the model is "
                   "that something IS a pothole. It does not say how bad the pothole is.")

    if low_confidence:
        st.warning(f"The model is only {max(confs):.0%} sure about this detection. "
                   "Check the photo yourself before saving a report.")

    st.subheader("Save as a report")
    suggested_locality = ""
    if gps:
        st.success(f"Location read from the photo's GPS data: {gps[0]:.5f}, {gps[1]:.5f}")
        suggested_locality = lookup_locality(round(gps[0], 5), round(gps[1], 5)) or ""
        if suggested_locality:
            st.caption("Locality suggested from those coordinates using OpenStreetMap. "
                       "Check it and edit it if needed.")
        else:
            st.caption("Could not look up the locality (this needs an internet connection). "
                       "Please type it in.")
    else:
        st.caption("This photo has no GPS data stored in the file (WhatsApp and screenshots remove it). "
                   "If coordinates are printed on the photo, type them in below. Otherwise "
                   "right-click the spot in Google Maps and copy the coordinates.")
    # With GPS in the photo the boxes are filled in. Without it they start EMPTY, so a report
    # can never be saved at a made-up default location by accident.
    default_lat, default_lon = gps if gps else (None, None)

    with st.form("report_form"):
        lat = st.number_input("Latitude (required)", value=default_lat, step=0.0001,
                              format="%.6f", placeholder="e.g. 12.9177")
        lon = st.number_input("Longitude (required)", value=default_lon, step=0.0001,
                              format="%.6f", placeholder="e.g. 77.6238")
        locality = st.text_input("Locality / road name (required)", value=suggested_locality,
                                 placeholder="e.g. Silk Board Junction")
        description = st.text_area("Description (optional)")
        submitted = st.form_submit_button("Save report")

    if submitted:
        if not locality.strip():
            st.error("Please enter the locality or road name before saving the report.")
        elif lat is None or lon is None:
            st.error("Please enter both the latitude and the longitude before saving the report.")
        elif not (BENGALURU_BOUNDS[0] <= lat <= BENGALURU_BOUNDS[1]
                  and BENGALURU_BOUNDS[2] <= lon <= BENGALURU_BOUNDS[3]):
            st.error("These coordinates are outside Bengaluru, so the report was NOT saved. "
                     "Latitude is the north-south number (about 12.9 in Bengaluru) and longitude "
                     "is the east-west number (about 77.6). Check both, and that they are not swapped.")
        else:
            save_report(image, annotated, lat, lon, locality, description, confs, boxes, severity)
            st.success("Report saved! Open the Dashboard page to see it on the map.")


# ----------------------------------------------------------------------
# 6. PAGE: DASHBOARD
# ----------------------------------------------------------------------
def clean_text(value):
    """Blank database fields can come back as None or NaN; turn both into an empty string."""
    return "" if pd.isna(value) else str(value)


def page_dashboard():
    st.header("Pothole reports across Bengaluru")

    df = db_read("SELECT * FROM reports ORDER BY created_at DESC")
    if df.empty:
        st.info("No reports yet. Save one from the Detect & report page.")
        return

    levels = ["Low", "Medium", "High", "Unclear"]
    chosen = st.multiselect("Show severity", levels, default=levels)
    view = df[df["severity"].isin(chosen)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Reports", len(view))
    c2.metric("High severity", int((view["severity"] == "High").sum()))
    c3.metric("Potholes detected", int(view["pothole_count"].sum()))

    # Map: one coloured circle per report. Click a circle to see its details.
    bengaluru_map = folium.Map(location=BENGALURU_CENTER, zoom_start=11, tiles="OpenStreetMap")
    for row in view.itertuples():
        name = html.escape(clean_text(row.locality)) or "Unnamed location"
        note = html.escape(clean_text(row.description))
        popup_html = (f"<b>{name}</b><br>Severity: {row.severity}<br>"
                      f"Potholes: {row.pothole_count}<br>Reported: {row.created_at}<br>{note}")
        folium.CircleMarker(
            location=(row.latitude, row.longitude),
            radius=9,
            color=SEVERITY_COLORS[row.severity],
            fill=True,
            fill_opacity=0.8,
            tooltip=f"{name} - {row.severity}",
            popup=folium.Popup(popup_html, max_width=260),
        ).add_to(bengaluru_map)
    st_folium(bengaluru_map, height=550, use_container_width=True, returned_objects=[])
    st.caption("Green = Low, orange = Medium, red = High, grey = Unclear (needs a human look). "
               "Severity is a rough estimate from the photo and never measures depth.")

    # Hotspots: split the city into ~500 m squares and count reports in each.
    st.subheader("Areas with repeated reports")
    grid = view.assign(cell_lat=view["latitude"] // 0.005, cell_lon=view["longitude"] // 0.005)
    hotspots = (
        grid.groupby(["cell_lat", "cell_lon"])
        .agg(area=("locality", "first"), reports=("id", "count"), potholes=("pothole_count", "sum"))
        .reset_index()
    )
    hotspots = hotspots[hotspots["reports"] >= 2].sort_values("reports", ascending=False)
    if hotspots.empty:
        st.caption("No area has more than one report yet.")
    else:
        st.dataframe(hotspots[["area", "reports", "potholes"]], hide_index=True)

    st.subheader("All reports")
    st.dataframe(
        view[["id", "created_at", "locality", "severity", "pothole_count",
              "max_confidence", "description"]],
        hide_index=True,
    )


# ----------------------------------------------------------------------
# 7. MAIN - runs every time the page loads or you click something
# ----------------------------------------------------------------------
init_db()

st.title(f"🕳️ {APP_NAME}")
st.caption("AI pothole detection and reporting for Bengaluru roads")

page = st.sidebar.radio("Go to", ["Detect & report", "Dashboard"])

with st.sidebar.expander("Demo data"):
    st.caption("The starter reports are fake sample data for demonstration.")
    if st.button("Reset to demo data"):
        db_execute("DELETE FROM reports")
        seed_demo_data()
        st.success("Demo data restored.")

if page == "Detect & report":
    page_detect()
else:
    page_dashboard()