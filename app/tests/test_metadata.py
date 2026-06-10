"""Tests for metadata persistence: JSONL stream, CSV, manifest and key-safety.

These build :class:`GenerationResult` objects by hand (no provider/engine) so
they exercise the writer in isolation, including the rule that *failures are
recorded, never discarded*, and that no API-key material ever reaches disk.
"""

from __future__ import annotations

import csv
import json

from app.core.config import AppConfig
from app.core.metadata import MetadataWriter
from app.core.models import (
    BatchGenerationRequest,
    CostEstimate,
    DistributionMode,
    GenerationResult,
    ItemStatus,
    ModelInfo,
    PlannedItem,
    PromptOptions,
    Run,
    RunStatus,
)
from app.core.storage import Storage


def _success_result(item_id: str = "portrait_000001") -> GenerationResult:
    return GenerationResult(
        id=item_id,
        filename=f"{item_id}.png",
        provider="mock",
        model="mock-image",
        prompt="a fully rendered prompt",
        age_bucket="adult, 26 to 40",
        gender_bucket="female-presenting",
        ethnicity_bucket="East Asian",
        variation_level=0,
        size="1024x1024",
        seed=7,
        estimated_cost_usd=0.0,
        actual_cost_usd=0.0,
        cost_is_estimated=False,
        status=ItemStatus.SUCCESS,
        retries=0,
    )


def _failed_result(item_id: str = "portrait_000002") -> GenerationResult:
    return GenerationResult(
        id=item_id,
        filename=None,
        provider="mock",
        model="mock-image",
        prompt="another rendered prompt",
        age_bucket="adult, 26 to 40",
        gender_bucket="male-presenting",
        ethnicity_bucket="White European",
        variation_level=0,
        size="1024x1024",
        seed=8,
        estimated_cost_usd=0.0,
        actual_cost_usd=None,
        cost_is_estimated=True,
        status=ItemStatus.FAILED,
        error="ProviderError: transient upstream failure",
        retries=3,
    )


def test_jsonl_keeps_success_and_failure_rows(tmp_path):
    """Both a SUCCESS and a FAILED result land in metadata.jsonl as valid JSON;
    failures are NOT discarded and carry status/error/retries."""
    storage = Storage(tmp_path)
    run_dir = storage.prepare(tmp_path / "run_meta")
    writer = MetadataWriter(storage, run_dir, save_prompt=True)

    writer.append_result(_success_result())
    writer.append_result(_failed_result())

    jsonl_path = storage.metadata_jsonl_path(run_dir)
    lines = [ln for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 2

    records = [json.loads(ln) for ln in lines]  # every line must be valid JSON
    by_status = {rec["status"]: rec for rec in records}

    assert by_status["success"]["status"] == "success"

    failed = by_status["failed"]
    assert failed["status"] == "failed"
    assert failed["error"] == "ProviderError: transient upstream failure"
    assert failed["retries"] == 3


def test_csv_has_uniform_columns_across_success_and_failure(tmp_path):
    """write_csv yields a DictReader-readable file whose success and failure rows
    share an identical key-set."""
    storage = Storage(tmp_path)
    run_dir = storage.prepare(tmp_path / "run_csv")
    writer = MetadataWriter(storage, run_dir, save_prompt=True)

    csv_path = writer.write_csv([_success_result(), _failed_result()])

    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        header = reader.fieldnames

    assert len(rows) == 2
    assert header is not None
    # Same column set for every row, matching the header exactly.
    for row in rows:
        assert set(row.keys()) == set(header)
    assert set(rows[0].keys()) == set(rows[1].keys())

    # Required reporting columns are present.
    for col in ("id", "status", "error", "retries"):
        assert col in header


def _build_run(tmp_path) -> Run:
    request = BatchGenerationRequest(
        provider="mock",
        model_id="mock-image",
        age_buckets=["adult, 26 to 40"],
        gender_buckets=["female-presenting"],
        ethnicity_buckets=["East Asian"],
        distribution_mode=DistributionMode.EVEN,
        total_count=2,
        size="1024x1024",
        output_dir=str(tmp_path / "run_manifest"),
    )
    model_info = ModelInfo(
        provider="mock",
        model_id="mock-image",
        display_name="Mock provider (no network, free)",
        supports_size=["1024x1024"],
        price_per_image_usd=0.0,
    )
    estimate = CostEstimate(
        provider="mock",
        model_id="mock-image",
        total_count=2,
        price_per_image_usd=0.0,
        estimated_total_usd=0.0,
        pricing_available=True,
    )
    plan = [
        PlannedItem(
            index=0,
            id="portrait_000001",
            filename="portrait_000001.png",
            prompt_options=PromptOptions(
                age_bucket="adult, 26 to 40",
                gender_bucket="female-presenting",
                ethnicity_bucket="East Asian",
            ),
        ),
        PlannedItem(
            index=1,
            id="portrait_000002",
            filename="portrait_000002.png",
            prompt_options=PromptOptions(
                age_bucket="adult, 26 to 40",
                gender_bucket="female-presenting",
                ethnicity_bucket="East Asian",
            ),
        ),
    ]
    run = Run(
        run_id="run_test",
        request=request,
        model_info=model_info,
        estimate=estimate,
        output_dir=tmp_path / "run_manifest",
        plan=plan,
        status=RunStatus.COMPLETED,
    )
    run.record(_success_result("portrait_000001"))
    run.record(_failed_result("portrait_000002"))
    return run


def test_manifest_is_valid_json_with_summary(tmp_path):
    """write_manifest writes a manifest.json that json.loads cleanly and contains a
    summary with planned/succeeded/failed counts."""
    storage = Storage(tmp_path)
    run = _build_run(tmp_path)
    run_dir = storage.prepare(run.output_dir)
    writer = MetadataWriter(storage, run_dir, save_prompt=True)

    manifest_path = writer.write_manifest(run)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "summary" in manifest
    summary = manifest["summary"]
    assert summary["planned"] == 2
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1


def test_manifest_does_not_leak_api_keys(tmp_path, monkeypatch):
    """Security: build a manifest from a real config/Run and assert the serialized
    manifest text contains no API-key field values nor a fake key from the env."""
    fake_key = "sk-test-LEAKED-SECRET-DO-NOT-PERSIST-1234567890"
    monkeypatch.setenv("OPENAI_API_KEY", fake_key)
    monkeypatch.setenv("FAL_KEY", fake_key)
    monkeypatch.setenv("REPLICATE_API_TOKEN", fake_key)

    config = AppConfig.load()
    # Sanity: the config really did pick up the fake key (so the assertion is meaningful).
    assert config.settings.api_key_for("openai") == fake_key

    storage = Storage(tmp_path)
    run = _build_run(tmp_path)
    run_dir = storage.prepare(run.output_dir)
    writer = MetadataWriter(storage, run_dir, save_prompt=True)

    manifest_path = writer.write_manifest(run)
    text = manifest_path.read_text(encoding="utf-8")

    assert fake_key not in text
    assert "api_key" not in text
    # The SecretStr placeholder must not leak either.
    assert "get_secret_value" not in text
