"""Stage A deterministic validation (synthetic images + a sample of real bundled images)."""
import random

import numpy as np
import pytest
from PIL import Image, ImageFilter

import config
from src.gatekeeper import rejection_manager as rm
from src.gatekeeper.image_quality import modality_blur_check, validate_image
from tests.conftest import png_bytes, xray_like


def code(data, name="x.png"):
    r = validate_image(data, name)
    return r.rejection.code if r.rejection else None


def test_valid_png_passes():
    r = validate_image(png_bytes(xray_like()), "xray.png")
    assert r.ok and r.image is not None and len(r.sha256) == 64


@pytest.mark.parametrize("name", ["scan.gif", "scan.dcm", "scan", "scan.pdf"])
def test_unsupported_extension(name):
    assert code(png_bytes(xray_like()), name) == rm.IMAGE_UNSUPPORTED_FORMAT


def test_actual_format_checked_not_just_extension():
    gif = png_bytes(xray_like().convert("P"), "GIF")
    assert code(gif, "renamed.png") == rm.IMAGE_UNSUPPORTED_FORMAT


def test_extension_mismatch_is_warning_only():
    r = validate_image(png_bytes(xray_like(), "PNG"), "actually_png.jpg")
    assert r.ok and any("does not match" in w for w in r.warnings)


@pytest.mark.parametrize("data", [b"", b"not an image", png_bytes(xray_like())[:200]])
def test_corrupt_rejected(data):
    assert code(data) == rm.IMAGE_CORRUPTED


def test_too_small():
    r = validate_image(png_bytes(xray_like((100, 300))), "s.png")
    assert r.rejection.code == rm.IMAGE_TOO_SMALL
    assert "100" in r.rejection.explanation()


def test_extreme_aspect():
    assert code(png_bytes(xray_like((1200, 200)))) == rm.IMAGE_EXTREME_ASPECT


def test_blurry_rejected():
    img = xray_like().filter(ImageFilter.GaussianBlur(10))
    assert code(png_bytes(img)) == rm.IMAGE_TOO_BLURRY


def test_uniform_image_rejected():
    assert code(png_bytes(Image.new("L", (300, 300), 128))) == rm.IMAGE_TOO_BLURRY


def test_underexposed_and_overexposed():
    dark = Image.fromarray(np.random.default_rng(0).integers(0, 30, (300, 300), dtype=np.uint8))
    bright = Image.fromarray(np.random.default_rng(0).integers(240, 256, (300, 300), dtype=np.uint8))
    assert code(png_bytes(dark)) == rm.IMAGE_UNDEREXPOSED
    assert code(png_bytes(bright)) == rm.IMAGE_OVEREXPOSED


def test_naturally_dark_image_with_content_is_not_rejected():
    """Mostly black background (like a fundus surround) but real bright structure → passes."""
    a = np.zeros((320, 320), np.uint8)
    y, x = np.mgrid[0:320, 0:320]
    disc = (x - 160) ** 2 + (y - 160) ** 2 < 70 ** 2
    a[disc] = (120 + 60 * np.sin(x / 4.0))[disc].astype(np.uint8)
    assert validate_image(png_bytes(Image.fromarray(a)), "dark.png").ok


def test_sixteen_bit_image_converted_with_warning():
    a = (np.random.default_rng(0).random((300, 300)) * 60000).astype(np.uint16)
    r = validate_image(png_bytes(Image.fromarray(a, "I;16")), "x16.png")
    assert r.ok and any("16-bit" in w for w in r.warnings)


def test_modality_specific_blur_floor():
    assert modality_blur_check({"blur_laplacian_var": 10.0}, "fundus").code == rm.IMAGE_TOO_BLURRY
    assert modality_blur_check({"blur_laplacian_var": 10.0}, "chest_xray") is None


def test_rejection_has_reason_explanation_action_timestamp():
    r = validate_image(b"junk", "x.png").rejection.to_dict("en")
    assert r["code"] and r["explanation"] and r["suggested_action"] and r["timestamp"].endswith("Z")


def test_real_bundled_images_pass_quality_checks():
    """Thresholds must not reject valid training images (random sample per disease)."""
    rnd = random.Random(0)
    for disease, modality in (("tb", "chest_xray"), ("dr", "fundus"), ("oral", "histopathology")):
        files = sorted(p for p in config.DISEASES[disease]["data_dir"].rglob("*")
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if not files:
            pytest.skip("bundled data not present")
        for f in rnd.sample(files, 15):
            r = validate_image(f.read_bytes(), f.name)
            assert r.ok, (f, r.rejection and r.rejection.code)
            assert modality_blur_check(r.metrics, modality) is None, f
