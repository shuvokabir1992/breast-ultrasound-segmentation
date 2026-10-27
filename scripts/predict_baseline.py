"""Generate a native-size lesion mask from one image, without a target mask."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from breastseg.preprocess import letterbox
from breastseg.torch_pipeline import build_model, restore_probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("smoke", False):
        raise ValueError("Use a completed baseline checkpoint instead of a smoke checkpoint")
    config = checkpoint["config"]
    device = torch.device(args.device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with Image.open(args.image) as source:
        item = letterbox(source, size=int(config["image_size"]))
    tensor = torch.from_numpy(item["image"][None, None]).to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor).float())[0]
        native = restore_probability(probability, item["layout"]).cpu().numpy()
    mask = (native >= float(config["threshold"])).astype(np.uint8) * 255
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(args.output)
    print(f"Saved native-size binary mask: {args.output}")


if __name__ == "__main__":
    main()
