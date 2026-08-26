#!/usr/bin/env python3
"""Independently audit all session Cube.AI bundles and their manifests."""

from __future__ import annotations

import binascii
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEPLOY_DIR = Path(__file__).resolve().parent
INDY_ROOT = DEPLOY_DIR.parent
MIDSIZE_DIR = INDY_ROOT / "models" / "midsize"
SESSIONS = (
    "indy_20160622_01",
    "indy_20160630_01",
    "indy_20170131_02",
    "loco_20170210_03",
    "loco_20170215_02",
    "loco_20170301_05",
)
HEADER_FORMAT = "<8sHHII32sHHHHHH32sIIIIIIIII32s32sI32s"
HEADER_CRC_OFFSET = 196
HEADER_SIZE = 256
MAGIC = b"BCIAIB1\0"
EXPECTED_STATUS = "deployment_candidate_replay_complete"
EXPECTED_GRAPH_ABI_ID = "tcn64-gru64-xcubeai10.2-f32-v1"


def sha256_bytes(values: bytes) -> str:
    return hashlib.sha256(values).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def crc32_bytes(values: bytes) -> int:
    return binascii.crc32(values) & 0xFFFFFFFF


def crc32_hex(values: bytes) -> str:
    return f"{crc32_bytes(values):08x}"


def text_field(values: bytes) -> str:
    return values.split(b"\0", 1)[0].decode("utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_artifact(
    path: Path, expected: dict[str, Any], context: str
) -> None:
    values = path.read_bytes()
    require(len(values) == expected["bytes"], f"{context}: size mismatch")
    require(crc32_hex(values) == expected["crc32"], f"{context}: CRC32 mismatch")
    require(sha256_bytes(values) == expected["sha256"], f"{context}: SHA256 mismatch")


def verify_session(session: str) -> dict[str, Any]:
    session_dir = MIDSIZE_DIR / session
    cubeai_dir = session_dir / "cubeai"
    manifest_path = cubeai_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["model_id"] == session, f"{session}: manifest model ID")
    require(manifest["session"] == session, f"{session}: manifest session")
    require(manifest["status"] == EXPECTED_STATUS, f"{session}: status")
    require(not manifest["promotion_claimed"], f"{session}: promotion must be false")

    checkpoint_path = session_dir / "deployment_candidate.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(checkpoint["model_id"] == session, f"{session}: checkpoint model ID")
    require(checkpoint["status"] == EXPECTED_STATUS, f"{session}: checkpoint status")
    require(
        sha256(checkpoint_path) == manifest["checkpoint"]["sha256"],
        f"{session}: checkpoint SHA256",
    )

    constants_path = session_dir / "deployment_constants.npz"
    verify_artifact(
        constants_path, manifest["deployment_constants"], f"{session}: constants"
    )
    with np.load(constants_path, allow_pickle=False) as constants_file:
        constants = {key: constants_file[key] for key in constants_file.files}
    constant_model_id = constants["model_id"].item()
    if isinstance(constant_model_id, bytes):
        constant_model_id = constant_model_id.decode("utf-8")
    require(constant_model_id == session, f"{session}: constants model ID")

    golden_path = session_dir / "deployment_golden_vectors.npz"
    verify_artifact(
        golden_path,
        manifest["deployment_golden_vectors"],
        f"{session}: golden vectors",
    )
    encoder_path = cubeai_dir / "encoder.weights.bin"
    gru_path = cubeai_dir / "gru_head.weights.bin"
    verify_artifact(
        encoder_path, manifest["components"]["encoder"], f"{session}: encoder"
    )
    verify_artifact(
        gru_path, manifest["components"]["gru_head"], f"{session}: GRU/head"
    )
    encoder = encoder_path.read_bytes()
    gru = gru_path.read_bytes()

    bundle_path = cubeai_dir / f"{session}.aibundle"
    bundle = bundle_path.read_bytes()
    bundle_manifest = manifest["bundle"]
    require(len(bundle) == bundle_manifest["total_bytes"], f"{session}: bundle size")
    require(crc32_hex(bundle) == bundle_manifest["crc32"], f"{session}: bundle CRC32")
    require(sha256_bytes(bundle) == bundle_manifest["sha256"], f"{session}: bundle SHA256")
    fields = struct.unpack_from(HEADER_FORMAT, bundle, 0)
    (
        magic,
        format_version,
        header_size,
        flags,
        total_size,
        model_id_raw,
        source_channel_count,
        selected_channel_count,
        feature_count,
        window_bins,
        read_timestep,
        alignment,
        checkpoint_sha_raw,
        encoder_offset,
        encoder_size,
        encoder_crc,
        gru_offset,
        gru_size,
        gru_crc,
        params_offset,
        params_size,
        params_crc,
        body_sha_raw,
        bundle_version_raw,
        header_crc,
        graph_abi_id_raw,
    ) = fields
    require(magic == MAGIC, f"{session}: bundle magic")
    require(format_version == 1, f"{session}: format version")
    require(header_size == HEADER_SIZE, f"{session}: header size")
    require(flags == 1, f"{session}: flags")
    require(total_size == len(bundle), f"{session}: header total size")
    require(text_field(model_id_raw) == session, f"{session}: header model ID")
    require(
        source_channel_count == int(constants["source_channel_count"]),
        f"{session}: source channel count",
    )
    require(selected_channel_count == 96, f"{session}: selected channel count")
    require(feature_count == 192, f"{session}: feature count")
    require(window_bins == 50, f"{session}: window bins")
    require(read_timestep == 49, f"{session}: read timestep")
    require(alignment == 32, f"{session}: alignment")
    require(
        checkpoint_sha_raw.hex() == manifest["checkpoint"]["sha256"],
        f"{session}: header checkpoint SHA256",
    )
    require(text_field(bundle_version_raw) == "bci-cubeai-bundle-v1", f"{session}: bundle version")
    require(
        text_field(graph_abi_id_raw) == EXPECTED_GRAPH_ABI_ID,
        f"{session}: graph ABI ID",
    )
    require(
        bundle_manifest["graph_abi_id"] == EXPECTED_GRAPH_ABI_ID,
        f"{session}: manifest graph ABI ID",
    )
    for name, offset in (
        ("encoder", encoder_offset),
        ("GRU/head", gru_offset),
        ("parameters", params_offset),
    ):
        require(offset % alignment == 0, f"{session}: {name} is not aligned")
    require(encoder_offset == HEADER_SIZE, f"{session}: encoder offset")
    require(encoder_size == len(encoder), f"{session}: encoder size")
    require(gru_size == len(gru), f"{session}: GRU/head size")
    require(params_size == 976, f"{session}: parameter size")
    require(gru_offset >= encoder_offset + encoder_size, f"{session}: encoder overlap")
    require(params_offset >= gru_offset + gru_size, f"{session}: GRU overlap")
    require(params_offset + params_size == total_size, f"{session}: trailing size")
    encoder_in_bundle = bundle[encoder_offset : encoder_offset + encoder_size]
    gru_in_bundle = bundle[gru_offset : gru_offset + gru_size]
    params = bundle[params_offset : params_offset + params_size]
    require(encoder_in_bundle == encoder, f"{session}: encoder payload")
    require(gru_in_bundle == gru, f"{session}: GRU/head payload")
    require(crc32_bytes(encoder_in_bundle) == encoder_crc, f"{session}: encoder header CRC")
    require(crc32_bytes(gru_in_bundle) == gru_crc, f"{session}: GRU header CRC")
    require(crc32_bytes(params) == params_crc, f"{session}: params header CRC")
    require(sha256_bytes(bundle[HEADER_SIZE:]) == body_sha_raw.hex(), f"{session}: body SHA256")
    header_for_crc = bytearray(bundle[:HEADER_SIZE])
    struct.pack_into("<I", header_for_crc, HEADER_CRC_OFFSET, 0)
    require(crc32_bytes(header_for_crc) == header_crc, f"{session}: header CRC32")
    require(f"{header_crc:08x}" == bundle_manifest["header_crc32"], f"{session}: manifest header CRC")

    floor = np.frombuffer(params, dtype="<f4", count=192, offset=0)
    target_mean = np.frombuffer(params, dtype="<f4", count=2, offset=768)
    target_std = np.frombuffer(params, dtype="<f4", count=2, offset=776)
    channels = np.frombuffer(params, dtype="<u2", count=96, offset=784)
    require(np.array_equal(floor, constants["feature_std_floor"]), f"{session}: floor payload")
    require(np.array_equal(target_mean, constants["target_mean"]), f"{session}: target mean payload")
    require(np.array_equal(target_std, constants["target_std"]), f"{session}: target std payload")
    require(np.array_equal(channels, constants["selected_channel_indices"]), f"{session}: mapping payload")
    require(
        channels.astype(int).tolist() == manifest["selected_channel_indices"],
        f"{session}: manifest mapping",
    )
    for name, report in manifest["parity"].items():
        require(report["max_abs_error"] <= 1.0e-5, f"{session}: parity {name}")
    return {
        "model_id": session,
        "source_channels": source_channel_count,
        "encoder_bytes": encoder_size,
        "gru_head_bytes": gru_size,
        "bundle_bytes": total_size,
        "bundle_crc32": crc32_hex(bundle),
        "bundle_sha256": sha256_bytes(bundle),
        "chain_max_abs_error": manifest["parity"]["generated_c_chain_vs_pytorch"]["max_abs_error"],
    }


def main() -> None:
    results = [verify_session(session) for session in SESSIONS]
    index_path = MIDSIZE_DIR / "cubeai_bundles.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    indexed = {entry["model_id"]: entry for entry in index["sessions"]}
    require(set(indexed) == set(SESSIONS), "bundle index sessions")
    for result in results:
        entry = indexed[result["model_id"]]
        require(entry["bundle_bytes"] == result["bundle_bytes"], "index bundle size")
        require(entry["bundle_crc32"] == result["bundle_crc32"], "index bundle CRC32")
        require(entry["bundle_sha256"] == result["bundle_sha256"], "index bundle SHA256")
    print(json.dumps({"verified": True, "sessions": results}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"verification failed: {error}", file=sys.stderr)
        raise
