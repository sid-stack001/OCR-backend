import numpy as np
import pytest

import main


def test_clean_text_for_llm_removes_noise():
    raw = "__TEST__\n| value |\n[]\n(a)"
    cleaned = main.clean_text_for_llm(raw)
    assert "|" not in cleaned
    assert "__" not in cleaned


def test_adjust_gamma_shape_preserved():
    image = np.full((12, 12, 3), 128, dtype=np.uint8)
    adjusted = main.adjust_gamma(image, gamma=0.5)
    assert adjusted.shape == image.shape


def test_advanced_preprocess_rejects_bad_image_bytes():
    with pytest.raises(main.ValidationError):
        main.advanced_preprocess(np.frombuffer(b"not-an-image", np.uint8))
