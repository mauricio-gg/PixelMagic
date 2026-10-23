import os
import sys
import textwrap

import numpy as np
import pytest
from PIL import Image

from pixelmagic.model import Point
from pixelmagic.segmenter.base import (
    EXIT_NOT_RUN,
    EXIT_REFUSED_IO,
    Mask,
    SegmenterError,
    SegmentFailure,
    SegmentRequest,
)
from pixelmagic.segmenter.cli_adapter import SegmenterCli

#: Stands in for `pixelmagic-segmenter`: answers `masks` the way the contract
#: says, writing an L-shaped mask PNG for the request "a" and refusing "b".
HELPER = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    from PIL import Image

    argv = sys.argv[1:]
    out_dir = argv[argv.index("--out-dir") + 1]
    requests = json.load(open(argv[argv.index("--requests") + 1]))
    crop = np.zeros((3, 4), np.uint8)
    crop[:, 0] = 255   # the L: a left column ...
    crop[2, :] = 255   # ... and a bottom row
    Image.fromarray(crop, "L").save(out_dir + "/0000.png")
    print(json.dumps({"segmenter": "sam2.1-small-mlx", "results": [
        {"id": "a", "box": [10, 20, 14, 23], "area": 6, "score": 0.9, "mask": "0000.png"},
        {"id": "b", "error": json.dumps(requests[1])},
    ]}))
    sys.exit(1)   # a masks call with a failed request exits 1 but the reply stands
    """
)


def adapter_running(tmp_path, source: str) -> SegmenterCli:
    """A `SegmenterCli` whose helper is this little Python script."""
    script = tmp_path / "helper.py"
    script.write_text(source)
    return SegmenterCli([sys.executable, str(script)])


def test_masks_land_at_the_reported_box_and_a_refused_request_is_its_own_failure(tmp_path):
    requests = [
        SegmentRequest("a", points=[Point(x=11, y=21, label=1)]),
        SegmentRequest("b", points=[Point(x=1, y=2, label=0)], box=(5, 6, 50, 60)),
    ]

    answers = adapter_running(tmp_path, HELPER).segment(tmp_path / "picture.png", requests)

    assert isinstance(answers[0], Mask)
    assert answers[0].box == (10, 20, 14, 23)
    assert answers[0].pixels.tolist() == [
        [True, False, False, False],
        [True, False, False, False],
        [True, True, True, True],
    ]
    assert answers[0].score == 0.9
    assert isinstance(answers[1], SegmentFailure) and answers[1].id == "b"
    # The helper echoed what it was sent: points as [x, y, label], the box as four numbers.
    assert '"points": [[1, 2, 0]]' in answers[1].message
    assert '"box": [5, 6, 50, 60]' in answers[1].message


@pytest.mark.parametrize(
    ("script", "exit_code", "code"),
    [
        (
            'import json, sys\nprint(json.dumps({"failure": {"code": "MODEL_NOT_FOUND",'
            ' "message": "No weights.", "hint": "run pixelmagic-segmenter fetch-model"}}))\n'
            "sys.exit(3)",
            EXIT_REFUSED_IO,
            "MODEL_NOT_FOUND",
        ),
        ('import sys\nprint("Traceback: boom", file=sys.stderr)\nsys.exit(1)', 1, None),
        (
            'print("{\\"suggestions\\": []}")\nprint("x", end="")\nimport sys\nsys.exit(0)',
            EXIT_NOT_RUN,
            None,
        ),
        (
            'import json\nprint(json.dumps({"failure": {"code": "ODD", "message": "odd"}}))',
            EXIT_NOT_RUN,
            "ODD",
        ),
    ],
    ids=[
        "failure reply",
        "crash with no reply",
        "exit 0 but stdout is not one JSON object",
        "exit 0 but a failure object",
    ],
)
def test_a_failed_helper_is_an_error_never_an_empty_answer(tmp_path, script, exit_code, code):
    adapter = adapter_running(tmp_path, script)
    with pytest.raises(SegmenterError) as caught:
        adapter.suggest(tmp_path / "picture.png")
    # A crash must not read as "no objects on the picture".
    assert caught.value.exit_code == exit_code
    assert caught.value.code == code


def test_a_missing_helper_names_how_to_set_it_up(tmp_path):
    with pytest.raises(SegmenterError) as caught:
        SegmenterCli([str(tmp_path / "nowhere")]).suggest(tmp_path / "picture.png")
    assert caught.value.exit_code == EXIT_NOT_RUN
    assert "uv sync" in str(caught.value) and "--segmenter stub" in str(caught.value)


#: Set this to run the one test in this file that starts the real helper.
LIVE_ENV = "PIXELMAGIC_LIVE_SEGMENTER"


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get(LIVE_ENV),
    reason=f"set {LIVE_ENV}=1 (and set up segmenter/) to run the real Segment Anything helper",
)
def test_the_real_helper_finds_a_drawn_rectangle(tmp_path):
    """Opt-in: the gate on the helper's environment, weights and point scaling."""
    picture = np.full((256, 320, 3), 235, np.uint8)
    picture[90:170, 100:220] = (200, 40, 40)
    path = tmp_path / "picture.png"
    Image.fromarray(picture).save(path)
    truth = np.zeros((256, 320), bool)
    truth[90:170, 100:220] = True

    def iou_with_truth(mask: Mask) -> float:
        placed = np.zeros_like(truth)
        x0, y0, x1, y1 = mask.box
        placed[y0:y1, x0:x1] = mask.pixels
        return (placed & truth).sum() / (placed | truth).sum()

    segmenter = SegmenterCli()
    suggested = segmenter.suggest(path)
    (answer,) = segmenter.segment(
        path, [SegmentRequest("rect", points=[Point(x=160, y=130, label=1)])]
    )

    assert any(iou_with_truth(mask) > 0.9 for mask in suggested)
    assert isinstance(answer, Mask) and iou_with_truth(answer) > 0.9
