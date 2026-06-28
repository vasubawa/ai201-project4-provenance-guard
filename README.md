# Provenance Guard: AI Text Detection API

## 1. System Architecture
Provenance Guard is a lightweight backend API built with Flask. It receives text submissions, routes them through a multi-signal detection pipeline, assigns a calibrated confidence score mapping to a transparency label, and logs the entire transaction to a local SQLite database. It also features a rate-limited endpoint array and a built-in appeals workflow for creators.

## 2. Detection Signals & Confidence Scoring
**Signal 1: Semantic Coherence (Groq API / Llama-3)**
This signal evaluates the holistic semantic flow and stylistic predictability of the text. Because LLMs operate by predicting the most probable next token, AI text is inherently highly cohesive and predictable. 

**Signal 2: Stylometric Heuristics (Python)**
This signal calculates pure structural variance without relying on machine learning. It measures Sentence Length Variance (Burstiness) and Vocabulary Diversity (Type-Token Ratio). 

**The Scoring Approach (The Asymmetric Veto):**
Simply averaging these two scores creates a dangerous vulnerability to false positives. To prioritize creator trust, I implemented an "Asymmetric Veto." If the stylometric engine detects strong human structural variance (score <= 0.4), but the LLM flags the semantic flow as AI (score >= 0.7), the stylometric score acts as a heavy mathematical drag. It actively vetoes the LLM and forces the final combined score safely down into the "Uncertain" bracket to protect the human creator from a false accusation. 

*If I were deploying this in a real production environment, I would add a third signal (such as calculating cross-entropy/perplexity against a massive web corpus) to catch texts that have been heavily edited by humans but originated from an LLM.*

### Scoring Examples (From Testing)
**Example 1: High-Confidence Human**
* **Input:** `"ok so i finally tried that new ramen place downtown and honestly? underwhelming. the broth was fine but they put WAY too much sodium in it and i was thirsty for like three hours after."`
* **Result:** Final Score: `0.10` *(Groq: 0.2, Stylo: 0.0)*

**Example 2: Uncertain / Borderline**
* **Input:** `"Testing the appeal pipeline manually. I wrote this myself from personal experience."`
* **Result:** Final Score: `0.65` *(Groq: 0.8, Stylo: 0.5 - Triggered short-text mitigation)*

## 3. Transparency Labels
The final confidence score explicitly maps to one of three exact transparency labels:
* **Score 0.0 - 0.35:** `"Verified Human Creation"`
* **Score 0.36 - 0.79:** `"Uncertain Origin - AI detection inconclusive"`
* **Score 0.80 - 1.0:** `"High likelihood of AI generation"`

## 4. Known Limitations
**The "Short Text" Blindspot**
This system struggles heavily with very short texts (under 20 words or 3 sentences). Because Signal 2 relies on calculating the statistical variance of sentence lengths and vocabulary diversity, it is mathematically impossible to find meaningful variance in just two sentences. To prevent division-by-zero errors or wild score swings, the system actively mitigates this by defaulting the stylometric score to a neutral `0.5`, which forces the pipeline to rely almost entirely on the LLM's semantic score. 

## 5. Spec Reflection
* **How the spec helped:** Writing out the exact mapping thresholds (`0.35`, `0.79`, etc.) and label strings in `planning.md` made implementing the production layer in Milestone 5 completely frictionless. There was no guesswork; I just transcribed my plan into code.
* **Where the implementation diverged:** I didn't originally plan for the "short text blindspot" in my spec. Once I started writing the pure Python stylometric math, I realized I had to forcefully inject an edge-case mitigation (returning `0.5` early) if the text was too short, otherwise the standard deviation math would crash. 

## 6. AI Usage Reflection
* **Instance 1:** I prompted the AI to write the Groq API call. It generated standard API fetching code, but my VS Code type-checker flagged that `completion.choices[0].message.content` could theoretically return `None`. I had to revise the AI's code to extract the content first and wrap it in a safety `if not content:` check before passing it to the JSON parser to prevent fatal app crashes.
* **Instance 2:** I prompted the AI for test commands to hit my endpoints. It initially provided multiline Bash `curl` commands, which threw nasty syntax errors in my Windows environment. I revised my prompt to explicitly request native PowerShell `Invoke-RestMethod` loops, which successfully tested the rate limiter without freezing my terminal.

---

## 7. Evidence of Production Features

### Rate Limiting Proof
**Chosen Limits:** `5 per minute, 100 per day`. 
**Reasoning:** This easily allows a human writer to publish serial chapters or edit posts naturally, but is strict enough to instantly block automated scripts trying to scrape or flood the detection engine. 

**Terminal Output (12 Rapid Requests):**
```text
200 OK
200 OK
200 OK
200 OK
200 OK
429 Too Many Requests
429 Too Many Requests
429 Too Many Requests
429 Too Many Requests
429 Too Many Requests
429 Too Many Requests
429 Too Many Requests