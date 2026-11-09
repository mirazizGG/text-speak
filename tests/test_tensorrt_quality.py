"""Private benchmark provenance checks; no Torch, models, downloads, or GPU."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import struct
from types import SimpleNamespace
import wave

import pytest

from verification.tensorrt import audio_probe, quality_check


@dataclass
class Generation:
    num_step: int = 32
    guidance_scale: float = 2.0
    t_shift: float = .1
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    denoise: bool = True
    preprocess_prompt: bool = True
    postprocess_output: bool = True
    audio_chunk_duration: float = 15.0
    audio_chunk_threshold: float = 30.0
    pad_duration: float = .1
    fade_duration: float = .1


@pytest.fixture
def evidence(tmp_path):
    voices = tmp_path / "voices"
    profile = voices / "female_1"
    profile.mkdir(parents=True)
    for name in ("prompt.pt", "reference.wav", "manifest.json"):
        (profile / name).write_bytes(name.encode())
    engine = SimpleNamespace(model_id=audio_probe.MODEL_ID, model_revision=audio_probe.MODEL_REVISION,
                             voices_dir=voices, _prompts={"female_1": object()},
                             _generation=lambda options: Generation(**options))
    provenance = audio_probe.experiment_provenance(engine, "female_1", {"short": "Salom dunyo."},
                                                  {"num_step": 32, "guidance_scale": 2.0})
    waveform = tmp_path / "short.wav"
    with wave.open(str(waveform), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(struct.pack("<hh", 1024, -1024) * 2400)
    row = {"label": "short", **provenance["samples"]["short"], "warmup": False,
           "repetition": 1, "file": waveform.name,
           "wav_sha256": hashlib.sha256(waveform.read_bytes()).hexdigest()}
    report = {"status": "private_audio_experiment_complete_unvalidated_quality",
              "voice": "female_1", "seed": 1234, "steps": 32, "provenance": provenance,
              "runs": [row]}
    return tmp_path, report, engine


def write_report(directory, report, name="report.json"):
    target = directory / name
    target.write_text(json.dumps(report), encoding="utf8")
    return target


def test_profile_fingerprint_includes_extra_inputs_and_detects_changes(evidence):
    directory, report, engine = evidence
    before = report["provenance"]["reference_voice"]["files"]
    extra = engine.voices_dir / "female_1" / "prompt-extra.pt"
    extra.write_bytes(b"alternate prompt")
    after = audio_probe.voice_file_hashes(engine.voices_dir, "female_1")
    assert set(after) == set(before) | {"prompt-extra.pt"}
    assert after["prompt-extra.pt"] == hashlib.sha256(b"alternate prompt").hexdigest()
    extra.write_bytes(b"changed prompt")
    assert audio_probe.voice_file_hashes(engine.voices_dir, "female_1") != after
    with pytest.raises(ValueError, match="Unsafe"):
        audio_probe.voice_file_hashes(engine.voices_dir, "../female_1")


@pytest.mark.parametrize("status", [None, "running_private_audio_experiment", "failed_private_audio_experiment"])
def test_incomplete_or_failed_report_cannot_be_compared(evidence, status):
    directory, report, _ = evidence
    report["status"] = status
    with pytest.raises(ValueError, match="incomplete or failed"):
        quality_check.measured_runs(write_report(directory, report), directory)


def test_legacy_report_requires_regeneration(evidence):
    directory, report, _ = evidence
    del report["provenance"]
    with pytest.raises(ValueError, match="rerun"):
        quality_check.measured_runs(write_report(directory, report), directory)


@pytest.mark.parametrize("warmup", [False, True])
def test_every_run_must_match_its_captured_source_text(evidence, warmup):
    directory, report, _ = evidence
    extra = {**report["runs"][0], **audio_probe.text_record("Different sentence."), "warmup": warmup}
    report["runs"].append(extra)
    with pytest.raises(ValueError, match="conflicting source text"):
        quality_check.measured_runs(write_report(directory, report), directory)


@pytest.mark.parametrize("mutation", ["missing_text", "wrong_text_digest", "missing_run_digest"])
def test_source_text_hash_is_verified(evidence, mutation):
    directory, report, _ = evidence
    if mutation == "missing_text":
        del report["runs"][0]["text"]
    elif mutation == "wrong_text_digest":
        report["provenance"]["samples"]["short"]["text_sha256"] = "0" * 64
    else:
        del report["runs"][0]["text_sha256"]
    with pytest.raises(ValueError, match="text provenance"):
        quality_check.measured_runs(write_report(directory, report), directory)


@pytest.mark.parametrize("difference", ["model", "reference_voice", "language", "generation", "text"])
def test_pair_requires_identical_generation_inputs(evidence, difference):
    directory, baseline, _ = evidence
    candidate = deepcopy(baseline)
    provenance = candidate["provenance"]
    if difference == "model":
        provenance["model"]["revision"] = "b" * 40
    elif difference == "reference_voice":
        provenance["reference_voice"]["files"]["reference.wav"] = "b" * 64
    elif difference == "language":
        provenance["language"] = "en"
    elif difference == "generation":
        provenance["generation"]["t_shift"] = .2
    else:
        provenance["samples"]["short"] = audio_probe.text_record("Different sentence.")
        candidate["runs"][0].update(provenance["samples"]["short"])
    left, measurements = quality_check.measured_runs(write_report(directory, baseline, "left.json"), directory)
    right, _ = quality_check.measured_runs(write_report(directory, candidate, "right.json"), directory)
    with pytest.raises(ValueError, match="different text, model"):
        quality_check.paired_sources(left, right, measurements)


def test_supplied_source_must_match_recorded_text_exactly(evidence):
    directory, report, _ = evidence
    loaded, measurements = quality_check.measured_runs(write_report(directory, report), directory)
    assert quality_check.paired_sources(loaded, loaded, measurements) == {"short": "Salom dunyo."}
    for supplied in ({"short": "Wrong source."}, {}, {"another": "Salom dunyo."}):
        with pytest.raises(ValueError, match="exactly match"):
            quality_check.paired_sources(loaded, loaded, measurements, supplied)


def test_declared_but_unmeasured_sample_is_rejected(evidence):
    directory, report, _ = evidence
    report["provenance"]["samples"]["absent"] = audio_probe.text_record("Unmeasured.")
    with pytest.raises(ValueError, match="every declared sample"):
        quality_check.measured_runs(write_report(directory, report), directory)


def test_complete_metadata_pair_retains_source_and_never_certifies_quality(evidence):
    directory, report, _ = evidence
    source = write_report(directory, report)
    target = directory / "quality.json"
    quality_check.main(["--baseline-report", str(source), "--candidate-report", str(source),
                        "--output", str(target), "--metadata-only"])
    result = json.loads(target.read_text(encoding="utf8"))
    assert result["status"] == "signal_only_requires_listening"
    assert result["provenance_matched"] is True
    assert result["pairs"][0]["source_text"] == "Salom dunyo."
    assert result["pairs"][0]["identical_pcm"] is True
    assert result["quality_validated"] is False
    assert result["speaker_identity_validated"] is False
    assert result["listening_required"] is True
