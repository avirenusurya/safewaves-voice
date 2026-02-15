import base64
import binascii
import json
import os
import shutil
import uuid
from pathlib import Path

import librosa
import numpy as np
import torch
import uvicorn
from fastapi import Body, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

try:
    import gradio as gr
except Exception:
    gr = None

IS_HF_SPACE = bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST") or os.getenv("SPACE_AUTHOR_NAME"))
DEFAULT_REMOTE_MODEL_REPO = os.getenv("HF_MODEL_REPO", "avirenusurya/guvi-model")


def is_df_arena_model(path: str) -> bool:
    config_path = Path(path) / "config.json"
    if not config_path.exists():
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception:
        return False
    custom_pipelines = config_data.get("custom_pipelines", {})
    return "antispoofing" in custom_pipelines


def resolve_default_model_path() -> str:
    # Prefer local model paths first (keeps local workflow unchanged).
    for candidate in ("./models/DF_Arena", ".", "./model"):
        if is_df_arena_model(candidate):
            return candidate

    # On Space, fall back to model repo id if local files are not present.
    if IS_HF_SPACE:
        return DEFAULT_REMOTE_MODEL_REPO

    # Legacy local fallback.
    return "./models/DF_Arena" if Path("./models/DF_Arena").exists() else "./model"


MODEL_PATH = os.getenv("MODEL_PATH", resolve_default_model_path())
FALLBACK_MODEL_PATH = os.getenv("FALLBACK_MODEL_PATH", ".")
ENABLE_STRICT_CLONE_HEURISTIC = os.getenv("ENABLE_STRICT_CLONE_HEURISTIC", "0") == "1"
ENABLE_GRADIO_UI = os.getenv("ENABLE_GRADIO_UI", "1") == "1"
PREFERRED_DEVICE = os.getenv("PREFERRED_DEVICE", "cpu" if IS_HF_SPACE else "cuda").lower()
ALLOW_CPU_FALLBACK = os.getenv("ALLOW_CPU_FALLBACK", "1" if IS_HF_SPACE else "0") == "1"
MODEL_DTYPE_NAME = os.getenv("MODEL_DTYPE", "float32").lower()
TEMP_DIR = "/tmp/temp_audio"
ID2LABEL = {0: "HUMAN", 1: "AI_GENERATED"}

HF_HOME = os.getenv("HF_HOME", str(Path(".hf_cache").resolve()))
HF_HUB_CACHE = os.getenv("HF_HUB_CACHE", str(Path(HF_HOME) / "hub"))
HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", str(Path(HF_HOME) / "modules"))
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", str(Path(HF_HOME) / "transformers"))

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(HF_HOME, exist_ok=True)
os.makedirs(HF_HUB_CACHE, exist_ok=True)
os.makedirs(HF_MODULES_CACHE, exist_ok=True)
os.makedirs(TRANSFORMERS_CACHE, exist_ok=True)

os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", HF_HUB_CACHE)
os.environ.setdefault("HF_MODULES_CACHE", HF_MODULES_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", TRANSFORMERS_CACHE)

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, pipeline

MODEL_BACKEND = "uninitialized"
LOAD_ERROR_MESSAGE = None
if PREFERRED_DEVICE == "cpu":
    device = "cpu"
elif PREFERRED_DEVICE == "cuda":
    if torch.cuda.is_available():
        device = "cuda"
    elif ALLOW_CPU_FALLBACK:
        device = "cpu"
    else:
        device = "cuda"
        LOAD_ERROR_MESSAGE = (
            "CUDA is required (PREFERRED_DEVICE=cuda) but no GPU is visible to PyTorch. "
            "Set ALLOW_CPU_FALLBACK=1 only if you explicitly want CPU."
        )
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline_device = 0 if device == "cuda" else -1
if MODEL_DTYPE_NAME == "float16":
    dtype = torch.float16
elif MODEL_DTYPE_NAME == "bfloat16":
    dtype = torch.bfloat16
else:
    dtype = torch.float32

model = None
feature_extractor = None
antispoof_pipe = None

def map_df_label(raw_label: str) -> str:
    label = (raw_label or "").strip().lower()
    if label in {"spoof", "ai_generated", "ai-generated", "fake"}:
        return "AI_GENERATED"
    if label in {"bonafide", "human", "real"}:
        return "HUMAN"
    if "spoof" in label or "fake" in label:
        return "AI_GENERATED"
    if "bona" in label or "human" in label or "real" in label:
        return "HUMAN"
    return raw_label.upper() if raw_label else "UNKNOWN"


def normalize_output(label: str, score: float) -> tuple[str, float]:
    # Enforce contest schema: classification must be strictly AI_GENERATED or HUMAN.
    if label not in {"AI_GENERATED", "HUMAN"}:
        label = "AI_GENERATED" if float(score) >= 0.5 else "HUMAN"

    safe_score = float(score)
    if np.isnan(safe_score) or np.isinf(safe_score):
        safe_score = 0.0
    safe_score = max(0.0, min(1.0, safe_score))
    return label, safe_score


def stage_local_model_code_for_dynamic_loader(model_path: str) -> None:
    source_dir = Path(model_path)
    target_dir = Path(HF_MODULES_CACHE) / "transformers_modules" / source_dir.name
    target_dir.mkdir(parents=True, exist_ok=True)
    for py_file in source_dir.glob("*.py"):
        shutil.copy2(py_file, target_dir / py_file.name)
    for json_file in source_dir.glob("*.json"):
        shutil.copy2(json_file, target_dir / json_file.name)


print("Starting DeepFake Detection Server...")
print(f"Model path preference: {MODEL_PATH}")
print(f"Device preference: {PREFERRED_DEVICE}; active device: {device}")
print(f"Model dtype: {MODEL_DTYPE_NAME}")
print(f"HF Space detected: {IS_HF_SPACE}")
print(f"HF modules cache: {HF_MODULES_CACHE}")
print("Loading Model...")
try:
    if LOAD_ERROR_MESSAGE:
        raise RuntimeError(LOAD_ERROR_MESSAGE)

    model_source_exists_local = Path(MODEL_PATH).exists()

    if model_source_exists_local and is_df_arena_model(MODEL_PATH):
        stage_local_model_code_for_dynamic_loader(MODEL_PATH)
        antispoof_pipe = pipeline(
            task="antispoofing",
            model=MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
            device=pipeline_device,
            dtype=dtype,
        )
        MODEL_BACKEND = "df_arena_antispoofing"
        print(f"Loaded DF Arena antispoofing pipeline from {MODEL_PATH} on {device}")
    elif not model_source_exists_local:
        antispoof_pipe = pipeline(
            task="antispoofing",
            model=MODEL_PATH,
            trust_remote_code=True,
            local_files_only=False,
            device=pipeline_device,
            dtype=dtype,
        )
        MODEL_BACKEND = "df_arena_antispoofing"
        print(f"Loaded DF Arena antispoofing pipeline from remote repo {MODEL_PATH} on {device}")
    else:
        resolved_fallback = FALLBACK_MODEL_PATH if Path(FALLBACK_MODEL_PATH).exists() else "."
        source_path = MODEL_PATH if Path(MODEL_PATH).exists() else resolved_fallback
        model = AutoModelForAudioClassification.from_pretrained(source_path, local_files_only=True)
        feature_extractor = AutoFeatureExtractor.from_pretrained(source_path, local_files_only=True)
        model.to(device)
        model.eval()
        MODEL_BACKEND = "legacy_audio_classifier"
        print(f"Loaded legacy classifier from {source_path} on {device}")
except Exception as e:
    print(f"Failed to load model: {e}")
    LOAD_ERROR_MESSAGE = str(e)
    MODEL_BACKEND = "load_failed"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"status": "error", "message": "Invalid request body"},
    )

@app.get("/", response_class=HTMLResponse)
async def read_root():
    ui_link = "<p><a href='/ui'>Open Demo UI</a></p>" if ENABLE_GRADIO_UI and gr is not None else ""
    return (
        "<h1>DeepFake Detector API</h1>"
        "<p>Active.</p>"
        "<p><a href='/docs'>View API Documentation (Swagger UI)</a></p>"
        f"{ui_link}"
    )

def generate_explanation(classification, score):
    score_pct = round(score * 100, 2)
    if classification == "AI_GENERATED":
        if score > 0.98: return f"Strong synthetic artifacts (Confidence: {score_pct}%)"
        elif score > 0.85: return f"Unnatural spectral consistency (Confidence: {score_pct}%)"
        else: return f"Possible digital manipulation (Confidence: {score_pct}%)"
    else:
        return f"Natural human speech pattern (Confidence: {score_pct}%)"

def process_audio_array(audio_path):
    try:
        speech, sr = librosa.load(audio_path, sr=16000, duration=10)
        return speech
    except Exception as e:
        print(f"Error loading: {e}")
        return np.array([])


def infer_from_speech_array(speech: np.ndarray) -> tuple[str, float]:
    if MODEL_BACKEND == "df_arena_antispoofing":
        raw_output = antispoof_pipe(speech)
        score = float(raw_output.get("score", 0.0))
        raw_label = str(raw_output.get("label", ""))
        label = map_df_label(raw_label)
    else:
        inputs = feature_extractor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16000 * 10,
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_id = torch.argmax(logits, dim=-1).item()
        probabilities = torch.softmax(logits, dim=-1)
        score = probabilities[0][predicted_id].item()
        label = ID2LABEL[predicted_id]
    return label, score


def build_success_payload(language: str, label: str, score: float) -> dict:
    label, score = normalize_output(label, score)
    explanation = generate_explanation(label, score)

    if ENABLE_STRICT_CLONE_HEURISTIC and label == "HUMAN" and score < 0.92:
        label = "AI_GENERATED"
        explanation = (
            f"High-quality voice clone detected (Confidence in Human: {round(score * 100, 1)}% < 92%)"
        )
        score = 1.0 - score

    return {
        "status": "success",
        "language": language,
        "classification": label,
        "confidenceScore": round(score, 4),
        "explanation": explanation,
    }

@app.post("/api/voice-detection")
def detect_voice(
    x_api_key: str = Header(None, alias="x-api-key"),
    body: dict = Body(...)
):
    if MODEL_BACKEND == "load_failed":
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": LOAD_ERROR_MESSAGE or "Model failed to load. Check local model files and restart the server."
            },
        )

    # Retrieve key from environment variable only (no hardcoded secret).
    EXPECTED_API_KEY = os.getenv("API_KEY")

    if not EXPECTED_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Server API key is not configured"},
        )

    if not x_api_key or x_api_key != EXPECTED_API_KEY:
        return JSONResponse(status_code=401, content={"status": "error", "message": "Invalid API key"})

    # Strict Requirements Check: Language must be one of the 5 supported (Tamil, English, Hindi, Malayalam, Telugu)
    ALLOWED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    language = body.get("language")
    
    if language not in ALLOWED_LANGUAGES:
        return JSONResponse(
            status_code=400, 
            content={
                "status": "error", 
                "message": f"Invalid language. Must be one of: {', '.join(ALLOWED_LANGUAGES)}"
            }
       )
    
    # Strict Requirements Check: Audio format must be exactly 'mp3'
    audio_format = body.get("audioFormat")
    if str(audio_format).lower() != "mp3":
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid format. Requirement: audioFormat must be 'mp3'."})

    audio_base64 = body.get("audioBase64")
    
    if not audio_base64:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Missing audioBase64"})
    
    # Validation for MP3 format based on Problem Statement "Audio format: MP3"
    # Note: We trust the problem statement says "MP3", but librosa can handle others.
    # However, we save it as .mp3 to be consistent.
    filename = f"{uuid.uuid4()}.mp3"
    filepath = os.path.join(TEMP_DIR, filename)
    
    try:
        try:
            audio_data = base64.b64decode(audio_base64, validate=True)
        except binascii.Error:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Malformed Base64 audio data"})

        with open(filepath, "wb") as f:
            f.write(audio_data)

        speech = process_audio_array(filepath)
        
        if len(speech) == 0:
             return JSONResponse(status_code=400, content={"status": "error", "message": "Could not process audio data."})

        label, score = infer_from_speech_array(speech)
        return build_success_payload(language=language, label=label, score=score)

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def _run_demo_detection(language: str, audio_base64: str):
    try:
        payload = {"audioBase64": audio_base64, "language": language, "audioFormat": "mp3"}
        expected_key = os.getenv("API_KEY")
        if not expected_key:
            error_obj = {"status": "error", "message": "Server API key is not configured"}
            return "Error: Server API key is not configured", error_obj
        result = detect_voice(x_api_key=expected_key, body=payload)

        if isinstance(result, JSONResponse):
            data = json.loads(result.body.decode("utf-8"))
        else:
            data = result

        if data.get("status") == "success":
            summary = (
                f"Classification: {data.get('classification')} | "
                f"Confidence: {data.get('confidenceScore')} | "
                f"Language: {data.get('language')}"
            )
        else:
            summary = f"Error: {data.get('message', 'Unknown error')}"
        return summary, data
    except Exception as e:
        error_obj = {"status": "error", "message": str(e)}
        return f"Error: {e}", error_obj


def _run_demo_detection_from_speech(language: str, speech: np.ndarray):
    try:
        label, score = infer_from_speech_array(speech)
        data = build_success_payload(language=language, label=label, score=score)
        summary = (
            f"Classification: {data.get('classification')} | "
            f"Confidence: {data.get('confidenceScore')} | "
            f"Language: {data.get('language')}"
        )
        return summary, data
    except Exception as e:
        error_obj = {"status": "error", "message": str(e)}
        return f"Error: {e}", error_obj


def gradio_detect_from_file(audio_file_path, language):
    if not audio_file_path:
        error_obj = {"status": "error", "message": "Please upload an MP3 file."}
        return "Upload an MP3 file to begin.", error_obj

    if not str(audio_file_path).lower().endswith(".mp3"):
        error_obj = {"status": "error", "message": "Only mp3 files are allowed."}
        return "Invalid file type. Upload an .mp3 file.", error_obj

    try:
        with open(audio_file_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")
        return _run_demo_detection(language=language, audio_base64=audio_base64)
    except Exception as e:
        error_obj = {"status": "error", "message": str(e)}
        return f"Error: {e}", error_obj


def gradio_detect_from_base64(audio_base64_input, language):
    if not audio_base64_input or not str(audio_base64_input).strip():
        error_obj = {"status": "error", "message": "Please paste Base64 audio string."}
        return "Paste Base64 audio to begin.", error_obj

    audio_base64 = "".join(str(audio_base64_input).split())
    return _run_demo_detection(language=language, audio_base64=audio_base64)


def build_gradio_ui():
    with gr.Blocks(title="SafeWaves Demo") as demo:
        gr.Markdown("# SafeWaves Demo UI")
        gr.Markdown("Demo either by uploading an MP3 file or by pasting Base64 audio.")

        with gr.Tab("Upload MP3 File"):
            with gr.Row():
                audio_file = gr.File(label="MP3 Audio File", type="filepath", file_types=[".mp3"])
                language_file = gr.Dropdown(
                    choices=["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
                    value="English",
                    label="Language",
                )
            run_file_btn = gr.Button("Analyze File")
            file_summary = gr.Textbox(label="Summary", interactive=False)
            file_json = gr.JSON(label="Raw API Response")
            run_file_btn.click(
                fn=gradio_detect_from_file,
                inputs=[audio_file, language_file],
                outputs=[file_summary, file_json],
            )

        with gr.Tab("Paste Base64"):
            with gr.Row():
                language_b64 = gr.Dropdown(
                    choices=["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
                    value="English",
                    label="Language",
                )
            base64_text = gr.Textbox(
                label="Audio Base64",
                lines=10,
                placeholder="Paste raw Base64 MP3 audio string here (no data: prefix).",
            )
            run_b64_btn = gr.Button("Analyze Base64")
            b64_summary = gr.Textbox(label="Summary", interactive=False)
            b64_json = gr.JSON(label="Raw API Response")
            run_b64_btn.click(
                fn=gradio_detect_from_base64,
                inputs=[base64_text, language_b64],
                outputs=[b64_summary, b64_json],
            )

    return demo


if ENABLE_GRADIO_UI and gr is not None:
    app = gr.mount_gradio_app(app, build_gradio_ui(), path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
