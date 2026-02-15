# SafeWaves: AI DeepFake Voice Detection

SafeWaves is an API-first voice authenticity system that detects whether an input voice sample is:

- `AI_GENERATED`
- `HUMAN`

The solution is built for the **India AI Impact Buildathon** voice-detection problem and follows the required request/response contract.

## Problem We Solve

AI-generated voices are now realistic enough to imitate human speech in scam and impersonation scenarios.  
SafeWaves helps protect trust by classifying audio as synthetic or real, with a confidence score and explanation.

## Supported Languages

The API is designed for these 5 required languages:

- Tamil
- English
- Hindi
- Malayalam
- Telugu

## API Contract

Endpoint:

`POST /api/voice-detection`

Headers:

- `Content-Type: application/json`
- `x-api-key: <your_api_key>`

Request body:

```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "BASE64_ENCODED_MP3"
}
```

Success response:

```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Unnatural spectral consistency (Confidence: 91.0%)"
}
```

Error response:

```json
{
  "status": "error",
  "message": "Invalid API key or malformed request"
}
```

## Simple Flow

1. Input audio arrives as Base64 MP3.
2. API validates key, language, and format.
3. Audio is decoded and converted to model-ready waveform.
4. Deepfake detector predicts class and confidence.
5. Output is normalized to strict labels: `AI_GENERATED` or `HUMAN`.

## Design Choices

- API-first architecture for automated evaluation.
- Strict output normalization for contest compliance.
- Accuracy prioritized over minimal latency.
- Confidence + explanation returned for transparent decisions.

## Current Limitations

- Very noisy or very short clips can reduce confidence.
- Heavily compressed/transcoded audio may degrade reliability.

## Model Attribution

This project uses the `Speech-Arena-2025/DF_Arena_1B_V_1` model and local model files derived from DF_Arena.

## Citation

If you use this project or model setup in your work, please cite:

```bibtex
@misc{kulkarni_2024_df_arena_1b,
  author       = {Ajinkya Kulkarni and Atharva Kulkarni and Sandipana Dowerah and Matthew Magimai Doss and Tanel Alumäe},
  title        = {DF_Arena_1B_V_1 - Universal Audio Deepfake Detection},
  year         = {2025},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1/}
}
```

## Local Run

```bash
cd /home/a/Documents/A/Hackathon/guvi-model
source venv/bin/activate
export MODEL_PATH=./models/DF_Arena
export PREFERRED_DEVICE=cuda
export ALLOW_CPU_FALLBACK=0
export MODEL_DTYPE=float32
export ENABLE_GRADIO_UI=1
export API_KEY=YOUR_API_KEY
python app.py
```

- API docs: `http://localhost:7860/docs`
- Demo UI: `http://localhost:7860/ui`

## Raw cURL Example

```bash
curl -X POST "http://localhost:7860/api/voice-detection" \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{
    "language":"English",
    "audioFormat":"mp3",
    "audioBase64":"BASE64_ENCODED_MP3"
  }'
```
