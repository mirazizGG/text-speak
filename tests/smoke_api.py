"""Real deployment check; writes returned WAVs for listening, never prints keys."""

import argparse
from array import array
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
import wave


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--access-file", default="API-ACCESS.env")
    parser.add_argument("--output", default="verification")
    args = parser.parse_args()
    access = dict(line.split("=", 1) for line in Path(args.access_file).read_text(encoding="utf-8-sig").splitlines() if line and not line.startswith("#") and "=" in line)
    base = access["TTS_API_URL"].rstrip("/")
    key = access["TTS_API_KEY"]
    output = Path(args.output)
    output.mkdir(exist_ok=True, parents=True)

    def request(path, payload=None, auth=True):
        headers = {"Authorization": "Bearer " + key} if auth else {}
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(base + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=200) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()

    def json_request(path):
        status, _, body = request(path)
        assert status == 200, (path, status, body[:300])
        return json.loads(body)

    assert request("/healthz", auth=False)[0] == 200
    assert request("/readyz", auth=False)[0] == 200
    assert request("/v1/languages", auth=False)[0] == 401
    assert request("/docs", auth=False)[0] == 200
    assert request("/static/swagger-ui-bundle.js", auth=False)[0] == 200
    languages = json_request("/v1/languages")["data"]
    assert {"uz", "en", "ru"} <= {item["code"] for item in languages}
    voices = json_request("/v1/voices")["data"]
    assert {"female_1", "female_2", "male_1", "male_2"} <= {item["id"] for item in voices}
    assert all(item["gender"] == "female" for item in json_request("/v1/voices?gender=female")["data"])
    invalid = {"text": "Sinov", "language": "invalid-language", "voice": "female_1"}
    assert request("/v1/audio/speech", invalid)[0] == 422
    assert request("/v1/audio/speech", {**invalid, "language": "uz", "gender": "male"})[0] == 422

    def generate(voice, language, text, filename):
        started = time.monotonic()
        status, headers, content = request("/v1/audio/speech", {"text": text, "language": language, "voice": voice})
        elapsed = time.monotonic() - started
        assert status == 200, (voice, language, status, content[:300])
        assert headers["Content-Type"].startswith("audio/wav")
        with wave.open(io.BytesIO(content), "rb") as audio:
            assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 24000)
            duration = audio.getnframes() / audio.getframerate()
            samples = array("h", audio.readframes(audio.getnframes()))
        if sys.byteorder != "little":
            samples.byteswap()
        assert 0.3 < duration < 600
        peak = max(abs(sample) for sample in samples) / 32768
        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples)) / 32768
        assert peak > 0.001 and rms > 0.0001
        (output / filename).write_bytes(content)
        result = {"voice": voice, "language": language, "seconds": round(duration, 3), "elapsed_seconds": round(elapsed, 3), "peak": round(peak, 5), "rms": round(rms, 5), "sha256": hashlib.sha256(content).hexdigest(), "file": filename}
        print(json.dumps(result), flush=True)
        return result

    results = [generate(voice["id"], "uz", "Assalomu alaykum. Bu matnni ovozga aylantirish xizmatining sinovi.", voice["id"] + "-uz.wav") for voice in voices]
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [
            pool.submit(generate, "female_1", "en", "Hello. This is a test of the text to speech service.", "female_1-en.wav"),
            pool.submit(generate, "male_1", "ru", "Здравствуйте. Это проверка сервиса преобразования текста в речь.", "male_1-ru.wav"),
        ]
        results.extend(future.result() for future in pending)
    report = {"url": base, "language_count": len(languages), "voice_count": len(voices), "http_validation_passed": True, "audio": results, "listening_review": "Not performed; inspect the saved WAV files."}
    (output / "smoke-result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PASS: authenticated remote API, 4 Uzbek voices, English/Russian requests, concurrent clients and WAV validation.")


if __name__ == "__main__":
    main()
