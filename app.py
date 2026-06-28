import os
import json
import sqlite3
import uuid
import re
import statistics
import math
from datetime import datetime, timezone
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask, Limiter, and Groq Client
app = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["10 per minute", "100 per day"],
    storage_uri="memory://",
)

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("ERROR: GROQ_API_KEY not found in environment variables!")
else:
    print(f"Groq API Key loaded: {api_key[:10]}...")

groq_client = Groq(api_key=api_key)
DB_PATH = "audit_log.db"


# ==========================================
# Database Helper Functions
# ==========================================
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        # Note: Uncomment the DROP lines below to reset data during development
        # conn.execute("DROP TABLE IF EXISTS audit_log")
        # conn.execute("DROP TABLE IF EXISTS creator_certificates")
        # conn.execute("DROP TABLE IF EXISTS appeals")

        # Main audit log
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                content_id TEXT PRIMARY KEY,
                creator_id TEXT,
                timestamp TEXT,
                groq_score REAL,
                stylometric_score REAL,
                entropy_score REAL,
                final_score REAL,
                label TEXT,
                status TEXT,
                content_type TEXT
            )
        """)

        # Creator certificates (provenance)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS creator_certificates (
                creator_id TEXT PRIMARY KEY,
                certificate_level TEXT,
                human_submissions INTEGER DEFAULT 0,
                ai_submissions INTEGER DEFAULT 0,
                appeals_successful INTEGER DEFAULT 0,
                earned_date TEXT
            )
        """)

        # Appeals log
        conn.execute("""
            CREATE TABLE IF NOT EXISTS appeals (
                appeal_id TEXT PRIMARY KEY,
                content_id TEXT,
                creator_id TEXT,
                appeal_reason TEXT,
                timestamp TEXT,
                status TEXT
            )
        """)


def log_event(entry: dict):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO audit_log (
                content_id, creator_id, timestamp, groq_score,
                stylometric_score, entropy_score, final_score, label, status, content_type
            ) VALUES (
                :content_id, :creator_id, :timestamp, :groq_score,
                :stylometric_score, :entropy_score, :final_score, :label, :status, :content_type
            )
        """,
            entry,
        )


def read_log(limit=20):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def log_appeal(content_id: str, creator_id: str, reason: str):
    appeal_id = str(uuid.uuid4())
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO appeals (appeal_id, content_id, creator_id, appeal_reason, timestamp, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (appeal_id, content_id, creator_id, reason, datetime.now(timezone.utc).isoformat(), "pending"),
        )
        conn.execute(
            "UPDATE audit_log SET status = ? WHERE content_id = ?",
            ("under_review", content_id),
        )
    return appeal_id


def get_content_by_id(content_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM audit_log WHERE content_id = ?", (content_id,)).fetchone()
    return dict(row) if row else None


def update_creator_certificate(creator_id: str, label: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cert = conn.execute(
            "SELECT * FROM creator_certificates WHERE creator_id = ?", (creator_id,)
        ).fetchone()

        if not cert:
            conn.execute(
                """
                INSERT INTO creator_certificates
                (creator_id, certificate_level, human_submissions, earned_date)
                VALUES (?, ?, ?, ?)
                """,
                (creator_id, "verified_human", 1 if label == "human" else 0, datetime.now(timezone.utc).isoformat()),
            )
        else:
            cert = dict(cert)
            if label == "human":
                cert["human_submissions"] += 1
            else:
                cert["ai_submissions"] += 1

            if cert["human_submissions"] >= 3 and cert["ai_submissions"] == 0:
                cert["certificate_level"] = "verified_human"

            conn.execute(
                """
                UPDATE creator_certificates
                SET human_submissions = ?, ai_submissions = ?, certificate_level = ?
                WHERE creator_id = ?
                """,
                (cert["human_submissions"], cert["ai_submissions"], cert["certificate_level"], creator_id),
            )


def get_analytics() -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        total = conn.execute("SELECT COUNT(*) as count FROM audit_log").fetchone()["count"]
        human = conn.execute(
            "SELECT COUNT(*) as count FROM audit_log WHERE label = ?", ("Verified Human Creation",)
        ).fetchone()["count"]
        ai = conn.execute(
            "SELECT COUNT(*) as count FROM audit_log WHERE label = ?", ("High likelihood of AI generation",)
        ).fetchone()["count"]
        uncertain = conn.execute(
            "SELECT COUNT(*) as count FROM audit_log WHERE label = ?", ("Uncertain Origin - AI detection inconclusive",)
        ).fetchone()["count"]
        appeals_count = conn.execute("SELECT COUNT(*) as count FROM appeals").fetchone()["count"]
        certified = conn.execute(
            "SELECT COUNT(*) as count FROM creator_certificates WHERE certificate_level = ?",
            ("verified_human",),
        ).fetchone()["count"]

        avg_conf = conn.execute("SELECT AVG(final_score) as avg FROM audit_log").fetchone()["avg"]

    return {
        "total_submissions": total,
        "human_classifications": human,
        "ai_classifications": ai,
        "uncertain_classifications": uncertain,
        "appeal_rate": round(appeals_count / total * 100, 2) if total > 0 else 0,
        "certified_creators": certified,
        "avg_confidence": round(avg_conf, 2) if avg_conf else 0,
    }


# ==========================================
# Detection Pipeline (Signal 1: Semantic)
# ==========================================
def get_groq_semantic_score(text: str) -> float:
    """
    Sends text to Groq LLM to evaluate semantic coherence and predictability.
    Returns a float between 0.0 (Human) and 1.0 (AI).
    """
    prompt = f"""
    You are an AI detection system analyzing text.
    Evaluate the following text for stylistic predictability, uniform sentence structure, and lack of human burstiness.
    Score the likelihood it was AI-generated on a scale from 0.0 (certainly human) to 1.0 (certainly AI).
    Return ONLY a JSON object with a single key 'ai_score' containing the float value. No markdown, no explanation.

    Text to analyze:
    "{text}"
    """

    try:
        print("[Groq] Sending request to API...")
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=30,
        )

        # Safely extract the content string first
        content = completion.choices[0].message.content
        print(f"[Groq] Response received: {content}")

        # Safety net: If Groq returns None or an empty string, fallback to uncertain
        if not content:
            print("Warning: Groq returned empty content.")
            return 0.5

        # Parse the JSON response
        result = json.loads(content)
        score = float(result.get("ai_score", 0.5))
        print(f"[Groq] Parsed score: {score}")
        return score

    except Exception as e:
        print(f"[Groq] API Error: {type(e).__name__}: {e}")
        return 0.5  # Default to neutral if API fails


# ==========================================
# Detection Pipeline (Signal 2: Structural)
# ==========================================
def get_stylometric_score(text: str) -> float:
    """
    Calculates structural variance (burstiness) and vocabulary diversity (TTR).
    Returns a float between 0.0 (Human) and 1.0 (AI).
    """
    words = [w.lower() for w in re.findall(r"\b\w+\b", text)]
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    # Edge Case Mitigation: Too short to score accurately
    if len(words) < 20 or len(sentences) < 3:
        print("[Stylo] Text too short, returning neutral 0.5")
        return 0.5

    # 1. Type-Token Ratio (Vocabulary Diversity)
    ttr = len(set(words)) / len(words)
    # Assuming typical human TTR is ~0.6+, AI is ~0.4 or lower. Map to 0-1 scale.
    ttr_score = max(0.0, min(1.0, 1.0 - ((ttr - 0.4) / 0.2)))

    # 2. Burstiness (Sentence Length Variance)
    sentence_lengths = [len(s.split()) for s in sentences]
    mean_len = statistics.mean(sentence_lengths)
    stdev = statistics.stdev(sentence_lengths)

    cv = stdev / mean_len if mean_len > 0 else 0
    # Assuming human CV (Coefficient of Variation) is ~0.5+, AI is ~0.2 or lower
    burst_score = max(0.0, min(1.0, 1.0 - ((cv - 0.2) / 0.3)))

    combined_stylo = (ttr_score + burst_score) / 2
    print(
        f"[Stylo] TTR: {ttr:.2f} (Score: {ttr_score:.2f}) | CV: {cv:.2f} (Score: {burst_score:.2f})"
    )
    print(f"[Stylo] Final Stylometric Score: {combined_stylo:.2f}")

    return round(combined_stylo, 3)


# ==========================================
# Detection Pipeline (Signal 3: Entropy)
# ==========================================
def get_entropy_score(text: str) -> float:
    """
    Analyzes character and word entropy to detect AI patterns.
    AI models tend to have lower character entropy (more predictable letter sequences).
    Returns a float between 0.0 (Human) and 1.0 (AI).
    """
    if len(text) < 50:
        print("[Entropy] Text too short, returning neutral 0.5")
        return 0.5

    # Character entropy
    char_freq = {}
    for char in text.lower():
        if char.isalnum() or char == " ":
            char_freq[char] = char_freq.get(char, 0) + 1

    total_chars = sum(char_freq.values())
    char_entropy = -sum((count / total_chars) * math.log2(count / total_chars)
                        for count in char_freq.values() if count > 0)

    # Normalize character entropy (typical range: 3.5-5.5)
    char_entropy_score = max(0.0, min(1.0, (5.5 - char_entropy) / 2.0))

    # Word frequency entropy
    words = re.findall(r"\b\w+\b", text.lower())
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1

    total_words = len(words)
    word_entropy = -sum((count / total_words) * math.log2(count / total_words)
                        for count in word_freq.values() if count > 0)

    # Normalize word entropy (typical range: 5-9)
    word_entropy_score = max(0.0, min(1.0, (9.0 - word_entropy) / 4.0))

    combined_entropy = (char_entropy_score + word_entropy_score) / 2
    print(f"[Entropy] Char entropy: {char_entropy:.2f} (Score: {char_entropy_score:.2f}) | "
          f"Word entropy: {word_entropy:.2f} (Score: {word_entropy_score:.2f})")
    print(f"[Entropy] Final Entropy Score: {combined_entropy:.2f}")

    return round(combined_entropy, 3)


# ==========================================
# Confidence Scoring & Mapping Logic
# ==========================================
def calculate_confidence_score(groq_score: float, stylo_score: float, entropy_score: float) -> float:
    """
    Combines three signals using ensemble voting with asymmetric veto.
    Weights: Groq 40%, Stylometric 35%, Entropy 25%
    """
    weighted_average = (groq_score * 0.4) + (stylo_score * 0.35) + (entropy_score * 0.25)
    print(f"[Ensemble] Weighted score: Groq({groq_score:.2f})*0.4 + Stylo({stylo_score:.2f})*0.35 + Entropy({entropy_score:.2f})*0.25 = {weighted_average:.3f}")

    # The Asymmetric Veto:
    # If structural score shows strong human variance (<= 0.4) but Groq flags as AI (>= 0.7)
    if stylo_score <= 0.4 and groq_score >= 0.7:
        print("[Scoring] Veto triggered! Dragging score down into Uncertain range.")
        # Force the score down into the "Uncertain" bracket (max 0.79)
        vetoed_score = min(0.79, weighted_average - 0.15)
        return round(max(0.0, vetoed_score), 3)

    return round(weighted_average, 3)


def map_score_to_label(final_score: float) -> tuple:
    """
    Maps the final confidence score to the attribution status and transparency label.
    Thresholds defined in planning.md.
    """
    if final_score <= 0.35:
        return "human", "Verified Human Creation"
    elif final_score <= 0.79:
        return "uncertain", "Uncertain Origin - AI detection inconclusive"
    else:
        return "ai", "High likelihood of AI generation"


# ==========================================
# API Endpoints
# ==========================================
@app.route("/submit", methods=["POST"])
@limiter.limit("5 per minute")  # Specific limit for submission
def submit():
    print("\n[Submit] Request received")
    data = request.get_json()

    # Input validation
    if not data or "text" not in data or "creator_id" not in data:
        print("[Submit] Validation failed: Missing 'text' or 'creator_id'")
        return jsonify({"error": "Missing 'text' or 'creator_id'"}), 400

    text = data.get("text")
    creator_id = data.get("creator_id")
    content_id = str(uuid.uuid4())
    print(
        f"[Submit] Processing submission: content_id={content_id}, creator_id={creator_id}"
    )

    # Run Signal 1 (Semantic)
    groq_score = get_groq_semantic_score(text)

    # Run Signal 2 (Structural)
    stylo_score = get_stylometric_score(text)

    # Run Signal 3 (Entropy)
    entropy_score = get_entropy_score(text)

    # Calculate combined score with ensemble
    final_score = calculate_confidence_score(groq_score, stylo_score, entropy_score)
    print(f"[Submit] Final Confidence Score: {final_score}")

    # Map to Transparency Label
    attribution, final_label = map_score_to_label(final_score)
    print(f"[Submit] Label assigned: {final_label}")

    # Create structured audit log entry
    log_entry = {
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "groq_score": groq_score,
        "stylometric_score": stylo_score,
        "entropy_score": entropy_score,
        "final_score": final_score,
        "label": final_label,
        "status": "classified",
        "content_type": "text",
    }

    # Write to database
    print("[Submit] Writing to database...")
    log_event(log_entry)
    print("[Submit] Database write complete")

    # Update creator certificate
    update_creator_certificate(creator_id, attribution)

    # Return response to user
    response = {
        "content_id": content_id,
        "attribution": attribution,
        "confidence": final_score,
        "label": final_label,
    }
    print(f"[Submit] Returning response: {response}")
    return jsonify(response)


@app.route("/appeal", methods=["POST"])
@limiter.limit("10 per hour")
def appeal():
    print("\n[Appeal] Request received")
    data = request.get_json()

    if not data or "content_id" not in data or "reason" not in data:
        print("[Appeal] Validation failed: Missing 'content_id' or 'reason'")
        return jsonify({"error": "Missing 'content_id' or 'reason'"}), 400

    content_id = data.get("content_id")
    reason = data.get("reason")

    # Verify content exists
    content = get_content_by_id(content_id)
    if not content:
        print(f"[Appeal] Error: content_id {content_id} not found")
        return jsonify({"error": "content_id not found"}), 404

    # Log appeal and update status
    appeal_id = log_appeal(content_id, content["creator_id"], reason)
    print(f"[Appeal] Appeal logged: {appeal_id}")

    return jsonify(
        {
            "appeal_id": appeal_id,
            "content_id": content_id,
            "status": "under_review",
            "message": "Your appeal was received and is under review.",
        }
    )


@app.route("/log", methods=["GET"])
def view_log():
    print("[Log] Request received")
    entries = read_log()
    print(f"[Log] Returning {len(entries)} entries")
    return jsonify({"entries": entries})


@app.route("/analytics", methods=["GET"])
def analytics():
    print("[Analytics] Request received")
    stats = get_analytics()
    print("[Analytics] Returning statistics")
    return jsonify(stats)


@app.route("/certificates", methods=["GET"])
def certificates():
    print("[Certificates] Request received")
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        certs = conn.execute(
            "SELECT creator_id, certificate_level, human_submissions, earned_date FROM creator_certificates WHERE certificate_level = ?",
            ("verified_human",),
        ).fetchall()
    return jsonify(
        {
            "verified_creators": [dict(c) for c in certs],
            "total_verified": len(certs),
        }
    )


@app.route("/creator/<creator_id>/certificate", methods=["GET"])
def get_creator_certificate(creator_id):
    print(f"[Certificate] Request for creator: {creator_id}")
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cert = conn.execute(
            "SELECT * FROM creator_certificates WHERE creator_id = ?", (creator_id,)
        ).fetchone()

    if not cert:
        return jsonify({"certificate_level": "none", "creator_id": creator_id}), 200

    return jsonify(dict(cert))


@app.route("/", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/analytics", methods=["GET"])
def api_analytics():
    stats = get_analytics()
    return jsonify(stats)


@app.route("/api/appeals", methods=["GET"])
def api_appeals():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        appeals = conn.execute(
            "SELECT * FROM appeals ORDER BY timestamp DESC LIMIT 20"
        ).fetchall()
    return jsonify({"appeals": [dict(a) for a in appeals]})


@app.route("/api/dashboard", methods=["GET"])
def api_dashboard():
    """Comprehensive dashboard data endpoint"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        # Get analytics
        analytics = get_analytics()

        # Get recent submissions
        submissions = conn.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 20"
        ).fetchall()

        # Get appeals
        appeals = conn.execute(
            "SELECT * FROM appeals ORDER BY timestamp DESC LIMIT 20"
        ).fetchall()

        # Get certificates
        certs = conn.execute(
            "SELECT * FROM creator_certificates WHERE certificate_level = ? ORDER BY earned_date DESC",
            ("verified_human",),
        ).fetchall()

    return jsonify(
        {
            "analytics": analytics,
            "submissions": [dict(s) for s in submissions],
            "appeals": [dict(a) for a in appeals],
            "certified_creators": [dict(c) for c in certs],
        }
    )


@app.route("/api/dashboard/export", methods=["GET"])
def api_dashboard_export():
    """Export dashboard data as CSV"""
    import csv
    from io import StringIO

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        submissions = conn.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC"
        ).fetchall()

    output = StringIO()
    if submissions:
        writer = csv.DictWriter(output, fieldnames=dict(submissions[0]).keys())
        writer.writeheader()
        for row in submissions:
            writer.writerow(dict(row))

    return output.getvalue(), 200, {"Content-Disposition": "attachment; filename=audit_log.csv"}


@app.route("/api/submissions", methods=["GET"])
def api_submissions():
    """Get all submissions with optional filtering"""
    creator_id = request.args.get("creator_id")
    status = request.args.get("status")
    limit = request.args.get("limit", 50, type=int)

    query = "SELECT * FROM audit_log WHERE 1=1"
    params = []

    if creator_id:
        query += " AND creator_id = ?"
        params.append(creator_id)

    if status:
        query += " AND status = ?"
        params.append(status)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        submissions = conn.execute(query, params).fetchall()

    return jsonify({"submissions": [dict(s) for s in submissions], "count": len(submissions)})


@app.route("/api/certificates", methods=["GET"])
def api_certificates():
    """Get certificate data"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        certs = conn.execute(
            "SELECT * FROM creator_certificates ORDER BY earned_date DESC"
        ).fetchall()

    return jsonify({"certificates": [dict(c) for c in certs], "total": len(certs)})


@app.route("/api/creator/<creator_id>/stats", methods=["GET"])
def api_creator_stats(creator_id):
    """Get detailed stats for a specific creator"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        # Get creator submissions
        submissions = conn.execute(
            "SELECT * FROM audit_log WHERE creator_id = ? ORDER BY timestamp DESC",
            (creator_id,),
        ).fetchall()

        # Get creator appeals
        appeals = conn.execute(
            "SELECT * FROM appeals WHERE creator_id = ? ORDER BY timestamp DESC",
            (creator_id,),
        ).fetchall()

        # Get creator certificate
        cert = conn.execute(
            "SELECT * FROM creator_certificates WHERE creator_id = ?", (creator_id,)
        ).fetchone()

    submissions_list = [dict(s) for s in submissions]
    human_count = sum(1 for s in submissions_list if "Human" in s.get("label", ""))
    ai_count = sum(1 for s in submissions_list if "AI generation" in s.get("label", ""))
    uncertain_count = sum(1 for s in submissions_list if "Uncertain" in s.get("label", ""))

    return jsonify(
        {
            "creator_id": creator_id,
            "total_submissions": len(submissions_list),
            "classifications": {"human": human_count, "ai": ai_count, "uncertain": uncertain_count},
            "appeals": [dict(a) for a in appeals],
            "certificate": dict(cert) if cert else None,
            "submissions": submissions_list,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# Initialize DB on startup
init_db()

if __name__ == "__main__":
    app.run(port=5000, debug=False)
