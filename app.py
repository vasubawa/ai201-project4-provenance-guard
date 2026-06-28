import os
import json
import sqlite3
import uuid
import re
import statistics
from datetime import datetime, timezone
from flask import Flask, request, jsonify
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
        # Note: Dropping the table on startup is great for testing,
        # but remove the DROP line if you want logs to persist between restarts!
        conn.execute("DROP TABLE IF EXISTS audit_log")
        conn.execute("""
            CREATE TABLE audit_log (
                content_id TEXT PRIMARY KEY,
                creator_id TEXT,
                timestamp TEXT,
                groq_score REAL,
                stylometric_score REAL,
                final_score REAL,
                label TEXT,
                status TEXT,
                appeal_reason TEXT
            )
        """)


def log_event(entry: dict):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO audit_log (
                content_id, creator_id, timestamp, groq_score, 
                stylometric_score, final_score, label, status, appeal_reason
            ) VALUES (
                :content_id, :creator_id, :timestamp, :groq_score, 
                :stylometric_score, :final_score, :label, :status, :appeal_reason
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
# Confidence Scoring & Mapping Logic
# ==========================================
def calculate_confidence_score(groq_score: float, stylo_score: float) -> float:
    """
    Combines signals using the Asymmetric Veto logic defined in planning.md.
    """
    base_average = (groq_score + stylo_score) / 2

    # The Asymmetric Veto:
    # If structural score shows strong human variance (<= 0.4) but Groq flags as AI (>= 0.7)
    if stylo_score <= 0.4 and groq_score >= 0.7:
        print("[Scoring] Veto triggered! Dragging score down into Uncertain range.")
        # Force the score down into the "Uncertain" bracket (max 0.79)
        vetoed_score = min(0.79, base_average - 0.15)
        return round(max(0.0, vetoed_score), 3)

    return round(base_average, 3)


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

    # Calculate combined score
    final_score = calculate_confidence_score(groq_score, stylo_score)
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
        "final_score": final_score,
        "label": final_label,
        "status": "classified",
        "appeal_reason": None,
    }

    # Write to database
    print("[Submit] Writing to database...")
    log_event(log_entry)
    print("[Submit] Database write complete")

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
def appeal():
    print("\n[Appeal] Request received")
    data = request.get_json()

    # Validate input based on Milestone 5 spec
    if not data or "content_id" not in data or "creator_reasoning" not in data:
        print("[Appeal] Validation failed: Missing 'content_id' or 'creator_reasoning'")
        return jsonify({"error": "Missing 'content_id' or 'creator_reasoning'"}), 400

    content_id = data.get("content_id")
    reason = data.get("creator_reasoning")

    # Update the SQLite database status
    with sqlite3.connect(DB_PATH) as conn:
        # Check if the content_id exists
        cursor = conn.execute(
            "SELECT status FROM audit_log WHERE content_id = ?", (content_id,)
        )
        row = cursor.fetchone()

        if not row:
            print(f"[Appeal] Error: content_id {content_id} not found")
            return jsonify({"error": "content_id not found"}), 404

        print(f"[Appeal] Updating status for {content_id} to 'under_review'")
        conn.execute(
            """
            UPDATE audit_log 
            SET status = 'under_review', appeal_reason = ? 
            WHERE content_id = ?
        """,
            (reason, content_id),
        )

    return jsonify(
        {
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


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# Initialize DB on startup
init_db()

if __name__ == "__main__":
    app.run(port=5000, debug=False)
