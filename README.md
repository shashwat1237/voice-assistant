> # ⚠️ **GROQ API RATE LIMIT WARNING**
> # **THIS APP USES THE GROQ FREE TIER. IF YOU EXCEED THE TOKEN LIMIT, THE APP WILL THROW A RATE LIMIT ERROR. THE ERROR MESSAGE WILL TELL YOU EXACTLY HOW MANY SECONDS YOU NEED TO WAIT BEFORE TRYING AGAIN. SIMPLY WAIT THAT AMOUNT OF TIME AND RETRY YOUR QUERY.**

---

# 🌾 किसान सहायक — Natural Farming Voice Assistant

A Hindi-first, voice-driven AI assistant for farmers in Maharashtra and surrounding regions. Farmers speak their questions in Hindi, and the system responds with spoken Hindi answers — covering natural farming techniques, crop rotation, live weather data, and local mandi market prices. No typing required.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Skeleton](#2-architecture-skeleton)
3. [Full Directory Structure](#3-full-directory-structure)
4. [How It Works — End-to-End Flow](#4-how-it-works--end-to-end-flow)
5. [Module Deep Dive](#5-module-deep-dive)
   - [app.py — Streamlit Frontend](#apppy--streamlit-frontend)
   - [voice/ — Speech Input & Output](#voice--speech-input--output)
   - [translator/ — Bidirectional Translation](#translator--bidirectional-translation)
   - [rag/ — Retrieval-Augmented Generation Pipeline](#rag--retrieval-augmented-generation-pipeline)
   - [orchestration/ — State & Error Management](#orchestration--state--error-management)
   - [prompts/ — LLM Instruction Files](#prompts--llm-instruction-files)
   - [data/ — The Knowledge Base](#data--the-knowledge-base)
6. [Configuration & Secrets](#6-configuration--secrets)
7. [Installation & Setup](#7-installation--setup)
8. [Running the App](#8-running-the-app)
9. [Key Design Decisions](#9-key-design-decisions)
10. [Known Limitations](#10-known-limitations)

---

## 1. Project Overview

This project solves a real accessibility problem: most AI assistants require users to type in English. Rural farmers in India speak Hindi and are often not comfortable with text interfaces. This assistant is built from the ground up to be voice-first and Hindi-first.

**What the assistant can answer:**
- Questions about natural/organic farming techniques (Jeevamrutha, Beejamrutha, mulching, pest control)
- Crop rotation strategies and multi-year farming models
- Government subsidies and farming schemes
- Current weather conditions (state-level: Haryana, Punjab, Uttar Pradesh)
- Current mandi (market) prices for wheat, rice, mustard, and chickpea

**What it strictly will NOT do:**
- Recommend chemical fertilizers or pesticides
- Invent government schemes or prices not in its knowledge base
- Answer questions outside the agricultural domain

**Core tech stack:**
- **Frontend:** Streamlit
- **Speech-to-Text:** faster-whisper (OpenAI Whisper, small model, CPU)
- **LLM & Translation:** Groq API (LLaMA 3.3 70B)
- **Embeddings:** BAAI/bge-small-en via sentence-transformers
- **Vector Database:** ChromaDB (persistent, local)
- **Text-to-Speech:** gTTS (Google TTS, Hindi)

---

## 2. Architecture Skeleton

At the highest level, every user interaction follows this pipeline:

```
[User Speaks Hindi]
       ↓
[Whisper STT] → Hindi Text
       ↓
[Groq Translate] → English Text
       ↓
[RAG Pipeline] → Relevant Context + LLM Answer (English)
       ↓
[Groq Translate] → Hindi Answer Text
       ↓
[gTTS] → Hindi Audio
       ↓
[Streamlit UI] → Plays Audio + Shows Text
```

There are two parallel paths for input — voice (microphone) and text (keyboard). Both converge at the translation step and follow the same pipeline from there.

---

## 3. Full Directory Structure

```
voice-assistant-main/
│
├── app.py                          # Main Streamlit application entry point
├── requirements.txt                # Python package dependencies
├── packages.txt                    # System-level dependencies (ffmpeg)
│
├── voice/
│   ├── whisper.py                  # Speech-to-text using faster-whisper
│   └── tts.py                      # Text-to-speech using gTTS
│
├── translator/
│   ├── hindi_to_english.py         # Hindi → English via Groq LLaMA
│   └── english_to_hindi.py         # English → Hindi via Groq LLaMA
│
├── rag/
│   ├── embeddings.py               # Loads BAAI/bge-small-en embedding model
│   ├── ingestion.py                # Chunks documents and stores in ChromaDB
│   ├── retrieval.py                # Queries ChromaDB + calls Groq LLM
│   └── chroma_db/                  # Auto-created persistent vector store
│
├── orchestration/
│   ├── state_manager.py            # Manages conversation history in session state
│   └── error_handlers.py           # Custom exceptions + bilingual error messages
│
├── prompts/
│   ├── expert_prompt.txt           # System prompt for the farming expert LLM
│   └── translator_prompt.txt       # System prompt for translation calls
│
└── data/
    ├── crop_rotation.md            # Knowledge: crop rotation science & models
    ├── natural_farming.md          # Knowledge: natural farming pillars & practices
    ├── subsidies.md                # Knowledge: government schemes & subsidies
    ├── market.json                 # Live data: mandi prices per quintal (INR)
    └── weather.json                # Live data: state-level weather parameters
```

---

## 4. How It Works — End-to-End Flow

Here is a detailed walkthrough of what happens from the moment a farmer taps the microphone to when they hear the answer.

### Step 1: Audio Capture
The user taps the microphone button rendered by `audio-recorder-streamlit`. This is a third-party Streamlit component that uses the browser's native `MediaRecorder` API. When the user stops recording, it returns raw audio bytes to the Python backend.

### Step 2: Saving the Audio File
The raw bytes are written to a temporary `.wav` file with a UUID-based name (e.g., `temp_a3f2...wav`). The UUID prevents file collisions in concurrent sessions.

### Step 3: Speech-to-Text (Whisper)
`voice/whisper.py` loads the `faster-whisper` small model (quantized to int8 for CPU performance). It transcribes the `.wav` file with:
- `language="hi"` — forces Hindi recognition, no language detection overhead
- `vad_filter=True` — Voice Activity Detection strips silent/noisy segments
- `min_silence_duration_ms=500` — segments shorter than 500ms of silence are merged

Validation checks run after transcription:
- File smaller than 1000 bytes → `EmptyAudioError`
- Empty transcript → `EmptyAudioError`
- Fewer than 2 words → `PartialRecordingError`

### Step 4: Hindi → English Translation
`translator/hindi_to_english.py` sends the Hindi text to Groq's LLaMA 3.3 70B model with a minimalist system prompt: "translate only, return only translation." Using an LLM for translation (rather than a library like `deep-translator`) gives significantly better results for agricultural terminology and informal rural Hindi dialects.

### Step 5: RAG Retrieval
`rag/retrieval.py` is the core of the system. It does the following:

**5a. Lazy DB Initialization:** Checks if the ChromaDB collection exists and has documents. If not, it automatically runs `ingest_knowledge()` to build the vector store on first run.

**5b. Intent Detection:** Keyword matching checks if the query is about weather or market prices. This bypasses the distance threshold check for these queries, since JSON data doesn't embed well in a text-focused vector store.

**5c. Vector Search:** The English query is embedded using `BAAI/bge-small-en` and the top 5 most semantically similar document chunks are retrieved from ChromaDB. A distance threshold of `1.1` (L2 space) blocks off-topic queries and raises `OutOfDomainError`.

**5d. JSON Data Injection:** For weather and market queries, the relevant `.json` files are read, parsed, and formatted into natural-language strings that are injected directly into the LLM context — bypassing the vector store entirely for these data types.

**5e. LLM Answer Generation:** The expert system prompt (`prompts/expert_prompt.txt`), conversation history, and full context are assembled and sent to Groq's LLaMA 3.3 70B. Temperature is set to `0.3` for factual consistency.

### Step 6: Response Sanitization
`app.py` runs `clean_final_response()` on the English answer before translation. This regex-based function strips any JSON artifacts (`{...}`), API-related strings, and stray colons that the LLM might occasionally include despite instructions. A second pass runs on the final Hindi output.

### Step 7: English → Hindi Translation
The cleaned English answer goes through `translator/english_to_hindi.py`, which uses the same Groq LLaMA model to translate back to Hindi Devanagari script.

### Step 8: Text-to-Speech
`voice/tts.py` uses `gTTS` to convert the final Hindi text to a `.mp3` audio file. It uses `lang='hi'` for natural Hindi pronunciation.

### Step 9: Output
Streamlit displays:
- The Hindi text answer
- An audio player with the spoken response
- The list of source documents used
- An expandable section showing the raw retrieved context chunks

### Step 10: Cleanup
The `finally` block in `app.py` deletes both temporary files (input `.wav` and output `.mp3`) regardless of whether the request succeeded or failed.

---

## 5. Module Deep Dive

### `app.py` — Streamlit Frontend

The application entry point. Responsibilities:

- **Page config:** Sets title and layout via `st.set_page_config`
- **Session init:** Calls `init_session_state()` on every render to ensure `chat_history` exists
- **Input handling:** Handles two input modes (audio bytes from recorder, text from `st.text_input`) and unifies them into a single `hindi_query` string
- **Pipeline orchestration:** Calls each module in sequence with spinner feedback shown to the user at each step
- **`clean_final_response(text)`:** A sanitization utility that uses regex to remove `{...}` JSON blocks, known bad phrases (in both Hindi and English), stray colons, and extra whitespace. It runs twice — once on the English answer, once on the final Hindi translation — acting as a double shield against LLM formatting leakage
- **Error handling:** A broad `try/except` block catches all custom exceptions and maps them to bilingual error messages
- **File cleanup:** `finally` block always removes temp audio files

### `voice/` — Speech Input & Output

**`whisper.py`**
- Loads `WhisperModel("small", device="cpu", compute_type="int8")` once and caches it with `@st.cache_resource` — subsequent requests reuse the loaded model without reloading from disk
- Uses `vad_filter=True` to prevent the model from "hallucinating" text from background noise or silence
- Three-tier validation: file size check → empty transcript check → minimum word count check

**`tts.py`**
- Thin wrapper around `gTTS`
- Returns the output path on success, empty string on failure (silent fail — the UI will still show the text even if audio generation fails)

### `translator/` — Bidirectional Translation

Both translation modules (`hindi_to_english.py`, `english_to_hindi.py`) follow the same pattern:
- Pull the Groq API key from `st.secrets`
- Call LLaMA 3.3 70B with `temperature=0.1` (near-deterministic for translation accuracy)
- System prompt instructs the model to return only the translation — no preamble, no explanation
- On any exception, raises a generic `Exception` with the Groq error message (not `LLMTimeoutError` directly — this is a known area for improvement)

### `rag/` — Retrieval-Augmented Generation Pipeline

**`embeddings.py`**
- Loads `BAAI/bge-small-en` via `sentence-transformers`, cached with `@st.cache_resource`
- `normalize_embeddings=True` is critical — it forces all vectors onto the unit hypersphere, which makes L2 distances mathematically equivalent to cosine similarity and ensures the `1.1` distance threshold works correctly

**`ingestion.py`**
- Scans the `data/` directory for all `.md` files (JSON files are excluded from the vector store intentionally — they're handled separately at retrieval time)
- `chunk_text(text, word_limit=500, overlap=100)`: splits documents into overlapping 500-word chunks. The 100-word overlap prevents context loss at chunk boundaries
- Uses `collection.upsert()` instead of `add()` — this makes ingestion idempotent, so rerunning it on an already-populated database doesn't throw `IDAlreadyExistsError`
- ChromaDB collection uses `"hnsw:space": "l2"` — L2 (Euclidean) distance space, aligned with the normalized embeddings

**`retrieval.py`**
- `DISTANCE_THRESHOLD = 1.1`: the maximum L2 distance allowed for a result to be considered "in domain." Queries about topics not in the knowledge base will produce distant results and raise `OutOfDomainError`
- Intent detection for weather/market queries bypasses this threshold because JSON content doesn't embed meaningfully alongside markdown text
- JSON data (weather, market) is parsed with defensive `isinstance(data, dict)` checks to support both flat and nested structures
- The final LLM call includes a hardcoded second system message as an absolute rule to prevent JSON output — defense in depth alongside the prompt instructions
- Returns a dict: `{"answer": str, "sources": list, "context_used": list}`

### `orchestration/` — State & Error Management

**`state_manager.py`**
- `init_session_state()`: safely initializes `chat_history` list in `st.session_state` if it doesn't exist (Streamlit re-runs the entire script on every interaction, so this guard is essential)
- `get_history_string()`: returns the last 3 interactions formatted as `User: ... \nSystem: ...` for injection into the LLM prompt. Limiting to 3 keeps the context window manageable
- `update_history(user_query, system_response)`: appends to the history list
- `clear_history()`: resets the list to empty

**`error_handlers.py`**
- Exception hierarchy: all custom exceptions inherit from `VoiceAssistantError(Exception)`
- `EmptyAudioError`, `NoisyAudioError`, `PartialRecordingError`: voice pipeline failures
- `LLMTimeoutError`, `VectorDBError`, `OutOfDomainError`: backend failures
- `ERROR_MESSAGES` dict: all messages are bilingual (Hindi + English in parentheses), ensuring the app is usable for both farmers and developers

### `prompts/` — LLM Instruction Files

**`expert_prompt.txt`**
The system prompt for the farming expert LLM. Key rules encoded in the prompt:
- Use only retrieved context (no hallucination of schemes or prices)
- Never recommend chemical farming solutions
- If info is unavailable, say exactly: `[Information not available in knowledge base.]`
- Routing logic: weather/market queries → use only environmental data, answer in 1-2 sentences. Farming technique queries → use documents, answer in detail
- Never output JSON, dict brackets, or the word "API"
- Template variables: `{history}` and `{context}` are replaced at runtime in `retrieval.py`

**`translator_prompt.txt`**
Minimal 4-line prompt: translate accurately, do not explain, do not summarize, return only translation.

### `data/` — The Knowledge Base

**Markdown files (ingested into ChromaDB):**

| File | Content |
|------|---------|
| `crop_rotation.md` | Biological nitrogen fixation, 3-year rotation models (wheat-pulse-oilseed, paddy cycle), weed suppression science, root architecture mechanics |
| `natural_farming.md` | Four pillars of ZBNF/SPNF: Jeevamrutha (microbial culture), Beejamrutha (seed treatment), Acchadana (mulching), Waaphasa (soil aeration); botanical pest management (Neemastra, Agniastra) |
| `subsidies.md` | Government farming schemes and subsidy information |

**JSON files (read directly at query time, not embedded):**

| File | Structure | Purpose |
|------|-----------|---------|
| `market.json` | `{"wheat": "2400", "rice": "2600", ...}` | Mandi prices in INR per quintal |
| `weather.json` | `{"haryana": {"temperature": 32, "humidity": 60, "rain": "Low"}, ...}` | State-level weather parameters |

JSON files support both flat (`commodity: price`) and nested (`commodity: {key: value}`) formats — the retrieval code handles both.

---

## 6. Configuration & Secrets

The app uses **Streamlit Secrets** for API key management. No `.env` file is needed.

For local development, create `.streamlit/secrets.toml` in the project root:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

For Streamlit Cloud deployment, add the secret via the app's Settings → Secrets panel in the dashboard.

The key is accessed in code via:
```python
api_key = st.secrets.get("GROQ_API_KEY")
```

**Getting a Groq API key:** Sign up at [console.groq.com](https://console.groq.com) — the free tier provides generous rate limits sufficient for this application.

---

## 7. Installation & Setup

### Prerequisites
- Python 3.9 or higher
- `ffmpeg` installed on the system (required by faster-whisper for audio decoding)

### System dependency
On Ubuntu/Debian:
```bash
sudo apt-get install ffmpeg
```
On macOS:
```bash
brew install ffmpeg
```
On Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

The `packages.txt` file in the repo is used by Streamlit Cloud to install `ffmpeg` automatically on deployment.

### Python dependencies
```bash
pip install -r requirements.txt
```

The `requirements.txt` uses `--extra-index-url https://download.pytorch.org/whl/cpu` to ensure the CPU-only build of PyTorch is installed, avoiding a multi-gigabyte GPU build that isn't needed here.

Pinned versions for stability:
```
faster-whisper==1.2.1
chromadb==0.4.24
sentence-transformers==2.5.1
gTTS==2.5.1
huggingface-hub==0.21.4
numpy<2.0.0          # sentence-transformers compatibility
protobuf==3.20.3     # ChromaDB compatibility
requests==2.31.0
```

### First-run database build
The ChromaDB vector store is built automatically on the first query. If you prefer to pre-build it:
```bash
python rag/ingestion.py
```
This creates `rag/chroma_db/` and embeds all `.md` files. Subsequent runs are idempotent.

---

## 8. Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

**First run:** The Whisper model (`~150MB`) and embedding model (`~130MB`) will be downloaded from HuggingFace on first use and cached locally. Expect a 1-2 minute wait on first query.

**Streamlit Cloud deployment:** Push the repository to GitHub, connect it on [share.streamlit.io](https://share.streamlit.io), and add the `GROQ_API_KEY` secret in the dashboard. The `packages.txt` handles `ffmpeg` installation automatically.

---

## 9. Key Design Decisions

**Why Groq for translation instead of `deep-translator`?**
LLaMA 3.3 70B produces dramatically better translations for informal rural Hindi, agricultural jargon, and mixed Hindi-English (Hinglish) queries. Library-based translators struggle with domain-specific terminology.

**Why `BAAI/bge-small-en` for embeddings?**
It's a small (90MB), fast model that performs well on English text with normalized embeddings. Since all queries are translated to English before embedding, an English embedding model is appropriate. `normalize_embeddings=True` aligns L2 distance with cosine similarity.

**Why are JSON files not embedded in ChromaDB?**
Embedding structured data like `{"wheat": "2400"}` produces poor semantic vectors. A text query about "wheat price" won't reliably match against a JSON snippet. Direct file parsing at query time (with intent detection) is more reliable for this data type.

**Why `upsert` instead of `add` in ingestion?**
`add()` throws `IDAlreadyExistsError` if the same document IDs already exist. `upsert()` makes the pipeline idempotent — safe to run multiple times without clearing the database first.

**Why two sanitization passes in `app.py`?**
The LLM occasionally leaks JSON-formatted data or technical strings despite prompt instructions. Running `clean_final_response()` on the English output before translation prevents bad strings from being translated into Hindi and corrupting the final answer.

**Why limit conversation history to 3 turns?**
Groq's LLaMA 3.3 70B has a large context window, but including long histories increases cost and latency. Three turns captures enough conversational context for follow-up questions without bloat.

---

> # ⚠️ GROQ API TOKEN LIMIT WARNING
> # **THE GROQ FREE TIER HAS A TOKEN-PER-MINUTE (TPM) AND TOKEN-PER-DAY (TPD) LIMIT. IF YOU SEND TOO MANY REQUESTS IN A SHORT PERIOD, THE APP WILL THROW A RATE LIMIT / TOKEN EXHAUSTION ERROR FROM THE GROQ API. THIS IS NOT A BUG — IT IS A GROQ FREE TIER RESTRICTION. UPGRADE YOUR GROQ PLAN OR WAIT FOR THE LIMIT TO RESET IF THIS HAPPENS.**

---

> # 📁 KNOWLEDGE BASE IS STORED AS `.md` FILES
> # **ALL FARMING KNOWLEDGE (crop rotation, natural farming techniques, subsidies) IS STORED AS PLAIN MARKDOWN (`.md`) FILES INSIDE THE `data/` FOLDER. TO ADD OR UPDATE KNOWLEDGE, SIMPLY EDIT OR ADD `.md` FILES IN THAT DIRECTORY AND RE-RUN THE INGESTION PIPELINE (`python rag/ingestion.py`) TO REBUILD THE VECTOR DATABASE.**

---

> # 🧪 WEATHER AND MARKET DATA IS MOCKED
> # **THE `weather.json` AND `market.json` FILES IN THE `data/` FOLDER CONTAIN HARDCODED, MANUALLY MAINTAINED (MOCK) DATA. THERE IS NO LIVE API INTEGRATION. THE WEATHER CONDITIONS AND MANDI PRICES SHOWN TO USERS ARE STATIC VALUES AND DO NOT REFLECT REAL-TIME INFORMATION.**

---

## 10. Known Limitations

- **Weather and market data is static.** The `weather.json` and `market.json` files are manually maintained. There is no live API integration — prices and weather reflect whatever values are currently in those files.
- **`subsidies.md` contains duplicate content.** The file currently has the same content as `natural_farming.md`. This should be replaced with actual government scheme data.
- **Single-language STT.** Whisper is hardcoded to `language="hi"`. Farmers who mix Hindi with regional languages (Marathi, Punjabi) may get poor transcription results.
- **CPU-only inference.** The Whisper model runs on CPU with int8 quantization. On low-end hardware, transcription can take 3-5 seconds per query.
- **No persistent user history across sessions.** `st.session_state` is reset when the browser tab is closed. There is no database-backed conversation history.
- **Groq API dependency.** Both translation steps and the LLM answer generation require an active internet connection and a valid Groq API key. The app has no offline fallback.
