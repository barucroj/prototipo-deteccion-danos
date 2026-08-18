import torch

from src.detection.model import build_model


def test_build_model_output_head_matches_num_classes():
    num_classes = 48
    model = build_model(num_classes=num_classes, pretrained=False)

    assert model.roi_heads.box_predictor.cls_score.out_features == num_classes
    assert model.roi_heads.box_predictor.bbox_pred.out_features == num_classes * 4


def test_model_forward_eval_mode_runs_on_random_image():
    model = build_model(num_classes=5, pretrained=False)
    model.eval()

    image = torch.rand(3, 100, 100)
    with torch.no_grad():
        output = model([image])[0]

    assert set(output.keys()) == {"boxes", "labels", "scores"}
    assert output["boxes"].shape[1] == 4


def test_model_forward_train_mode_computes_losses():
    model = build_model(num_classes=3, pretrained=False)
    model.train()

    image = torch.rand(3, 100, 100)
    target = {
        "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        "labels": torch.tensor([1], dtype=torch.int64),
    }

    loss_dict = model([image], [target])

    assert "loss_classifier" in loss_dict
    assert all(torch.is_tensor(v) for v in loss_dict.values())
