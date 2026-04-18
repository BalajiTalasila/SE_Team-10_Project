#CH.SC.U4AIE23056
# YOLOv5 Currency Detection with Stable Audio + Text Output

import argparse
import time
from pathlib import Path
import torch
import cv2
import pygame

from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.plots import Annotator, colors

# ---------------- AUDIO SETUP ----------------
AUDIO_DIR = Path("audio_wav")
pygame.mixer.init()

AUDIO_COOLDOWN = 3.0        # seconds
CONF_THRESHOLD = 0.55       # avoid random detection

last_spoken = {}            # label -> last time spoken


def play_audio(label):
    audio_file = AUDIO_DIR / f"{label}.wav"
    if not audio_file.exists():
        return

    pygame.mixer.music.load(str(audio_file))
    pygame.mixer.music.play()


# ---------------- MAIN DETECTION ----------------
def run(weights, source, data, imgsz=640, conf_thres=0.25, device="", view_img=False):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data)
    stride, names, pt = model.stride, model.names, model.pt

    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    model.warmup(imgsz=(1, 3, imgsz, imgsz))

    for path, im, im0s, _, _ in dataset:
        im = torch.from_numpy(im).to(device).float() / 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        preds = model(im)
        preds = non_max_suppression(preds, conf_thres, 0.45)

        for i, det in enumerate(preds):
            im0 = im0s[i].copy()
            annotator = Annotator(im0, line_width=3)

            current_time = time.time()

            if det is not None and len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in det:
                    conf = float(conf)
                    if conf < CONF_THRESHOLD:
                        continue   # ❌ Ignore weak detections

                    label = names[int(cls)]

                    # ---------- TEXT OUTPUT ----------
                    text = f"{label} ({conf:.2f})"
                    print(text)

                    annotator.box_label(
                        xyxy,
                        text,
                        color=colors(int(cls), True)
                    )

                    # ---------- AUDIO OUTPUT ----------
                    last_time = last_spoken.get(label, 0)
                    if current_time - last_time >= AUDIO_COOLDOWN:
                        play_audio(label)
                        last_spoken[label] = current_time

            result = annotator.result()

            if view_img:
                cv2.imshow("Currency Detection", result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return


# ---------------- CLI ----------------
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--device", default="")
    parser.add_argument("--view-img", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))