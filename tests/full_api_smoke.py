"""Exercise real model capabilities against a running API; no secrets in reports."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid
import wave
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8070")
    parser.add_argument("--access-file")
    parser.add_argument("--output", default="/tmp/full-api")
    args = parser.parse_args()
    key = os.environ.get("TTS_API_KEY")
    if args.access_file:
        access = dict(line.split("=", 1) for line in Path(args.access_file).read_text(encoding="utf-8-sig").splitlines() if "=" in line and not line.startswith("#"))
        key = access["TTS_API_KEY"]
    assert key, "An API key is required"
    output = Path(args.output)
    output.mkdir(exist_ok=True, parents=True)
    results = []

    def request(path, payload=None, fields=None, file_field=None, file_bytes=None):
        headers = {"Authorization": "Bearer " + key}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload, ensure_ascii=False).encode()
        elif fields is not None or file_field:
            boundary = "text-speak-" + uuid.uuid4().hex
            chunks = []
            for name, value in (fields or {}).items():
                chunks.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n').encode())
            if file_field:
                chunks.extend([(f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="reference.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode(), file_bytes, b"\r\n"])
            chunks.append((f"--{boundary}--\r\n").encode())
            data = b"".join(chunks)
            headers["Content-Type"] = "multipart/form-data; boundary=" + boundary
        req = urllib.request.Request(args.url.rstrip("/") + path, data=data, headers=headers)
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=200) as response:
                status, body, response_headers = response.status, response.read(), response.headers
        except urllib.error.HTTPError as error:
            status, body, response_headers = error.code, error.read(), error.headers
        return status, body, response_headers, round(time.monotonic() - started, 3)

    def json_call(label, path, **kwargs):
        status, body, headers, elapsed = request(path, **kwargs)
        assert status in (200, 201), (label, status, body[:300])
        result = json.loads(body)
        results.append({"case": label, "status": status, "elapsed_seconds": elapsed})
        print(json.dumps(results[-1]), flush=True)
        return result

    def wav_call(label, path="/v1/audio/speech", **kwargs):
        status, body, headers, elapsed = request(path, **kwargs)
        assert status == 200, (label, status, body[:300])
        assert headers["Content-Type"].startswith("audio/wav")
        with wave.open(io.BytesIO(body)) as audio:
            assert audio.getnchannels() == 1 and audio.getsampwidth() == 2
            assert audio.getframerate() == int(headers["X-Audio-Sample-Rate"])
            seconds = audio.getnframes() / audio.getframerate()
            pcm = audio.readframes(audio.getnframes())
        assert 0.2 < seconds < 600 and any(pcm), label
        (output / (label + ".wav")).write_bytes(body)
        results.append({"case": label, "status": status, "seconds": round(seconds, 3), "elapsed_seconds": elapsed, "sha256": hashlib.sha256(body).hexdigest()})
        print(json.dumps(results[-1]), flush=True)
        return body, seconds

    deadline = time.monotonic() + 90
    while True:
        try:
            if request("/readyz")[0] == 200:
                break
        except urllib.error.URLError:
            pass
        assert time.monotonic() < deadline, "Readiness deadline exceeded"
        time.sleep(1)
    caps = json_call("capabilities", "/v1/capabilities")
    assert set(caps["modes"]) == {"preset", "design", "auto", "clone"}
    assert caps["asr_available"] and {"en", "zh"} <= set(caps["normalization_languages"])
    assert "uz" not in caps["normalization_languages"]
    original = json_call("voices", "/v1/voices")["data"]
    assert {"female_1", "female_2", "male_1", "male_2"} <= {item["id"] for item in original}

    wav_call("legacy", payload={"text": "Assalomu alaykum. Bu eski so'rov shaklining sinovi.", "language": "uz", "voice": "female_1", "gender": "female"})
    wav_call("design", payload={"text": "This is a clear example of a newly designed speaking voice.", "language": "en", "mode": "design", "instruct": "male, middle-aged, low pitch", "seed": 101})
    wav_call("auto", payload={"text": "Today we are testing an automatically selected speaking voice.", "language": "en", "mode": "auto", "seed": 102})
    _, fixed = wav_call("fixed-duration", payload={"text": "This sentence has a fixed duration.", "language": "en", "voice": "male_1", "duration": 6.0, "speed": 2.0, "generation": {"postprocess_output": False, "pad_duration": 0.0, "fade_duration": 0.0, "audio_chunk_duration": 3.0, "audio_chunk_threshold": 3.0}})
    assert abs(fixed - 6) < 0.1
    phrase = "Today we are comparing two different speaking speeds using the same sentence."
    _, slow = wav_call("slow", payload={"text": phrase, "language": "en", "voice": "female_1", "speed": 0.7, "seed": 100})
    _, fast = wav_call("fast", payload={"text": phrase, "language": "en", "voice": "female_1", "speed": 1.5, "seed": 100})
    assert slow > fast * 1.2
    wav_call("normalized-en", payload={"text": "Today I have 25 apples and 12 oranges.", "language": "en", "voice": "female_1", "normalize_text": True})
    wav_call("nonverbal", payload={"text": "[laughter] Well, that was a surprise. [sigh] Let us try again.", "language": "en", "voice": "female_1"})
    wav_call("phonemes-en", payload={"text": "He plays the [B EY1 S] guitar while catching a [B AE1 S] fish.", "language": "en", "voice": "male_1"})
    wav_call("pinyin-zh", payload={"text": "这批货物打ZHE2出售后他严重SHE2本了。", "language": "zh", "voice": "female_1"})
    reference, _ = wav_call("reference", "/v1/voices/female_1/sample")
    reference_text = "Assalomu alaykum. Bugun siz bilan ravon va tushunarli ovozda suhbatlashamiz. Har bir so'z aniq va tabiiy talaffuz qilinadi."
    wav_call("clone-with-transcript", "/v1/audio/clone", fields={"text": "Assalomu alaykum. Bu ovoz audio namunadan yaratildi.", "language": "uz", "ref_text": reference_text, "options": json.dumps({"speed": 1.1, "generation": {"num_step": 16}})}, file_field="ref_audio", file_bytes=reference)
    transcription = json_call("transcription", "/v1/audio/transcriptions", file_field="audio", file_bytes=reference)
    assert isinstance(transcription["text"], str) and transcription["text"].strip()
    wav_call("clone-auto-transcript", "/v1/audio/clone", fields={"text": "Assalomu alaykum. Namuna matni avtomatik aniqlandi.", "language": "uz"}, file_field="ref_audio", file_bytes=reference)
    clone = json_call("saved-clone", "/v1/voices/clone", fields={"name": "API sinov klon ovozi", "gender": "female", "reference_language": "uz", "ref_text": reference_text, "preprocess_prompt": "false"}, file_field="ref_audio", file_bytes=reference)
    design = json_call("saved-design", "/v1/voices/design", payload={"name": "API designed test voice", "gender": "male", "language": "en", "instruct": "male, young adult, moderate pitch", "reference_text": "Hello everyone. Today we are creating a clear and natural speaking voice for this service.", "seed": 303, "generation": {"postprocess_output": False}})
    wav_call("saved-clone-reuse", payload={"text": "Salom. Bu saqlangan ovozdan qayta foydalanish sinovi.", "language": "uz", "voice": clone["id"]})
    wav_call("saved-design-reuse", payload={"text": "Hello again. This saved voice is now available for reuse.", "language": "en", "voice": design["id"]})
    wav_call("saved-clone-sample", clone["sample_url"])
    status, zipped, headers, elapsed = request("/v1/audio/batch", payload={"items": [{"text": "Hello. This is the first batch item.", "language": "en", "voice": "female_1"}, {"text": "Здравствуйте. Это второй пример.", "language": "ru", "mode": "design", "instruct": "male, young adult"}]})
    assert status == 200 and headers["Content-Type"].startswith("application/zip"), (status, zipped[:300])
    with zipfile.ZipFile(io.BytesIO(zipped)) as archive:
        assert set(archive.namelist()) == {"001.wav", "002.wav", "manifest.json"}
        assert len(json.loads(archive.read("manifest.json"))["items"]) == 2
        assert archive.testzip() is None
    (output / "batch.zip").write_bytes(zipped)
    results.append({"case": "batch", "status": status, "elapsed_seconds": elapsed})
    long_text = ("Assalomu alaykum. Bugun siz bilan yangi xabarlarni tinglaymiz. " * 20)[:1000]
    _, long_seconds = wav_call("long-form", payload={"text": long_text, "language": "uz", "voice": "male_1"})
    assert long_seconds > 30
    status, body, _, _ = request("/v1/audio/speech", payload={"text": "Menda 25 ta olma bor.", "language": "uz", "voice": "female_1", "normalize_text": True})
    assert status == 422
    status, body, _, _ = request("/v1/audio/clone", fields={"text": "Hello there.", "language": "en", "ref_text": "Hello"}, file_field="ref_audio", file_bytes=b"not an audio file")
    assert status == 422
    caps = json_call("capabilities-after-asr", "/v1/capabilities")
    assert caps["asr_loaded"]
    report = {"passed": True, "cases": results, "saved_voice_ids": [clone["id"], design["id"]], "transcription": transcription["text"], "listening_review": "Not performed; inspect saved WAVs for pronunciation and voice quality."}
    (output / "full-api-result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FULL_API_SMOKE_PASS", flush=True)


if __name__ == "__main__":
    main()
