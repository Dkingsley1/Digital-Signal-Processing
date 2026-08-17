#!/usr/bin/env python3
"""Canonical fixed-rig identities shared by production and research training."""

from __future__ import annotations

import hashlib
import json
import re


RIG_IDENTITY_FIELDS = (
    "amp",
    "amp_channel",
    "cabinet",
    "microphone",
    "mic_position",
    "boost_pedal",
    "sample_rate_hz",
)


def _plain(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower())
    return " ".join(text.split())


def _unlabeled(value: object) -> bool:
    return _plain(value) in {"", "unknown", "unlabeled", "none unlabeled"}


def canonical_rig_label(value: object, field: str, profile_family: object = "") -> str:
    text = _plain(value)
    family = _plain(profile_family)

    if field == "sample_rate_hz":
        try:
            return str(int(value or 0))
        except (TypeError, ValueError):
            return "0"

    if field == "boost_pedal":
        if text in {"", "none", "off", "bypassed", "none unlabeled"}:
            return "none" if text in {"none", "off", "bypassed"} else "unlabeled"
        if "maxon" in text and "808" in text:
            drive_zero = "drive 0" in text or "drive zero" in text
            tone_half = any(token in text for token in ("tone 5", "tone 50", "tone half", "tone halfway"))
            balance_full = any(token in text for token in ("balance 10", "balance full", "level 10", "level full"))
            if drive_zero and tone_half and balance_full:
                return "maxon od808 drive 0 tone 5 balance 10"

    if field == "amp_channel" and _unlabeled(value) and "rhythm" in family:
        return "rhythm"
    if field == "microphone" and "sm57" in text:
        return "shure sm57"
    if field == "amp" and "peavey" in text and "6505" in text:
        return "peavey 6505 mini head" if "mini" in text else "peavey 6505"
    if field == "cabinet" and "egnater" in text and "tweaker" in text:
        return "egnater tweaker 1x12 celestion"
    if field == "mic_position":
        if any(token in text for token in ("grille touch", "grill touch", "touching grille")):
            return "sm57 grille touch directly in front of speaker"
        if "close" in text and "front" in text:
            return "sm57 close directly in front of speaker"

    return "unlabeled" if _unlabeled(value) else text


def rig_identity_from_manifest(manifest: dict) -> dict:
    di_box = dict(manifest.get("di_box", {}))
    metadata = dict(manifest.get("take_metadata", {}))
    interface = dict(manifest.get("audio_interface", {}))
    rig = dict(manifest.get("rig", {}))
    profile_family = metadata.get("profile_family", "")
    raw = {
        "amp": di_box.get("amp_name") or rig.get("amp") or manifest.get("amp"),
        "amp_channel": metadata.get("amp_channel") or rig.get("amp_settings") or manifest.get("amp_settings"),
        "cabinet": di_box.get("cabinet_name") or rig.get("cabinet") or manifest.get("cabinet"),
        "microphone": di_box.get("mic_name") or rig.get("microphone") or rig.get("mic") or manifest.get("mic"),
        "mic_position": metadata.get("mic_position") or rig.get("mic_position") or manifest.get("mic_position"),
        "boost_pedal": metadata.get("boost_pedal") or rig.get("pedal") or manifest.get("pedal"),
        "sample_rate_hz": interface.get("sample_rate_hz") or manifest.get("sample_rate_hz") or 0,
    }
    identity = {
        field: canonical_rig_label(raw[field], field, profile_family=profile_family)
        for field in RIG_IDENTITY_FIELDS
    }
    identity["sample_rate_hz"] = int(identity["sample_rate_hz"])
    return identity


def rig_fingerprint(identity: dict) -> str:
    selected = {field: identity.get(field) for field in RIG_IDENTITY_FIELDS}
    encoded = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]

