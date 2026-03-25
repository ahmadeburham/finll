#!/usr/bin/env python3
"""
Single-script Egyptian ID pipeline baseline.

What it does:
- Passive selfie gate (tested heuristic mode; optional stronger model hooks are not included in this environment)
- Template-based ID-in-scene detection via feature matching + homography
- Card rectification / cutout
- ROI cropping for portrait / name / address / ID number / birth date using a JSON config
- OCR with multi-pass preprocessing using pytesseract (ara+eng)
- Face crop detection and fallback face matching (dlib detector + ORB/histogram/SSIM) if a deep face backend is unavailable
- Synthetic end-to-end self-test that generates mock assets and runs the whole pipeline

Notes:
- This script is intentionally dependency-light for reproducible local testing.
- For a stronger production face backend, replace FaceMatcherFallback with InsightFace embeddings.
- For a stronger production passive liveness backend, replace PassiveLivenessHeuristic with an ONNX anti-spoof model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
try:
    import dlib
except Exception:
    dlib = None
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from skimage import data as skdata
from skimage.metrics import structural_similarity as ssim


# -----------------------------
# Utility helpers
# -----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_pil(img: np.ndarray) -> Image.Image:
    if img.ndim == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def resize_keep_aspect(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return img.copy()
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)],
    ], dtype=np.float32)


def warp_from_quad(img: np.ndarray, quad: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    quad = order_quad(quad)
    w, h = out_size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_CUBIC)


def crop_norm(img: np.ndarray, box: List[float]) -> np.ndarray:
    h, w = img.shape[:2]
    x, y, bw, bh = box
    x1 = max(0, min(w - 1, int(round(x * w))))
    y1 = max(0, min(h - 1, int(round(y * h))))
    x2 = max(x1 + 1, min(w, int(round((x + bw) * w))))
    y2 = max(y1 + 1, min(h, int(round((y + bh) * h))))
    return img[y1:y2, x1:x2].copy()


def draw_norm_boxes(img: np.ndarray, boxes: Dict[str, List[float]], color=(0, 255, 0)) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for name, b in boxes.items():
        x, y, bw, bh = b
        p1 = (int(x * w), int(y * h))
        p2 = (int((x + bw) * w), int((y + bh) * h))
        cv2.rectangle(out, p1, p2, color, 2)
        cv2.putText(out, name, (p1[0], max(15, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


def image_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def image_brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))


def normalized_hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    ha = cv2.calcHist([a], [0], None, [64], [0, 256])
    hb = cv2.calcHist([b], [0], None, [64], [0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    score = cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)
    return float(max(-1.0, min(1.0, score)))


def clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def only_digits(text: str) -> str:
    return re.sub(r"\D+", "", text)


def only_date_chars(text: str) -> str:
    return re.sub(r"[^0-9/]", "", text)


def cv_imwrite(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


# -----------------------------
# Config and parsed fields
# -----------------------------

@dataclass
class TemplateConfig:
    template_size: Tuple[int, int]
    rois: Dict[str, List[float]]  # normalized x,y,w,h
    detector: str = "akaze"
    min_good_matches: int = 20
    min_inliers: int = 15
    min_inlier_ratio: float = 0.25

    def to_dict(self) -> dict:
        return {
            "template_size": list(self.template_size),
            "rois": self.rois,
            "detector": self.detector,
            "min_good_matches": self.min_good_matches,
            "min_inliers": self.min_inliers,
            "min_inlier_ratio": self.min_inlier_ratio,
        }

    @staticmethod
    def from_dict(d: dict) -> "TemplateConfig":
        return TemplateConfig(
            template_size=tuple(d["template_size"]),
            rois=d["rois"],
            detector=d.get("detector", "akaze"),
            min_good_matches=int(d.get("min_good_matches", 20)),
            min_inliers=int(d.get("min_inliers", 15)),
            min_inlier_ratio=float(d.get("min_inlier_ratio", 0.25)),
        )


@dataclass
class OCRFieldResult:
    text: str
    confidence: float
    variant: str


class _SimpleFaceRect:
    def __init__(self, x1, y1, x2, y2):
        self._x1 = int(x1); self._y1 = int(y1); self._x2 = int(x2); self._y2 = int(y2)
    def left(self): return self._x1
    def top(self): return self._y1
    def right(self): return self._x2
    def bottom(self): return self._y2


class _DlibFaceDetectorWrapper:
    def __init__(self):
        self.detector = make_face_detector()
    def __call__(self, gray, upsample=1):
        return self.detector(gray, upsample)
    @property
    def backend_name(self):
        return "dlib_hog"


class _OpenCVHaarFaceDetectorWrapper:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {cascade_path}")
    def __call__(self, gray, upsample=1):
        boxes = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        return [_SimpleFaceRect(x, y, x+w, y+h) for (x, y, w, h) in boxes]
    @property
    def backend_name(self):
        return "opencv_haar"


def make_face_detector():
    if dlib is not None:
        try:
            return _DlibFaceDetectorWrapper()
        except Exception:
            pass
    return _OpenCVHaarFaceDetectorWrapper()


# -----------------------------
# Passive selfie gate (tested heuristic mode)
# -----------------------------

class PassiveLivenessHeuristic:
    """
    This is NOT a production anti-spoof model.
    It is a tested selfie-quality gate that checks:
    - exactly one sufficiently large face
    - image sharpness
    - brightness
    - mild screen-border artifact heuristics
    """
    def __init__(self):
        self.detector = make_face_detector()

    def evaluate(self, bgr: np.ndarray) -> dict:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        dets = self.detector(gray, 1)
        h, w = gray.shape[:2]
        sharp = image_sharpness(gray)
        bright = image_brightness(gray)

        face_ok = False
        face_box = None
        face_area_ratio = 0.0
        if len(dets) == 1:
            d = dets[0]
            x1, y1, x2, y2 = max(0, d.left()), max(0, d.top()), min(w, d.right()), min(h, d.bottom())
            fw, fh = max(0, x2 - x1), max(0, y2 - y1)
            face_area_ratio = (fw * fh) / float(max(1, w * h))
            face_ok = face_area_ratio > 0.03
            face_box = [int(x1), int(y1), int(x2), int(y2)]

        # very weak screen border heuristic: too much edge energy exactly near borders can indicate recapture/screen
        edges = cv2.Canny(gray, 80, 160)
        border = 20
        border_mask = np.zeros_like(edges)
        border_mask[:border, :] = 255
        border_mask[-border:, :] = 255
        border_mask[:, :border] = 255
        border_mask[:, -border:] = 255
        border_edge_ratio = float((edges[border_mask == 255] > 0).mean()) if border_mask.size else 0.0

        passed = bool(len(dets) == 1 and face_ok and sharp >= 80.0 and bright >= 45.0 and bright <= 220.0 and border_edge_ratio < 0.25)
        score = 0.0
        score += min(1.0, sharp / 200.0) * 0.35
        score += (1.0 if len(dets) == 1 and face_ok else 0.0) * 0.4
        score += (1.0 if 45.0 <= bright <= 220.0 else 0.0) * 0.15
        score += max(0.0, 1.0 - border_edge_ratio / 0.25) * 0.10
        score = float(max(0.0, min(1.0, score)))

        return {
            "backend": f"heuristic_quality_gate+{self.detector.backend_name}",
            "passed": passed,
            "score": score,
            "num_faces": len(dets),
            "face_box": face_box,
            "face_area_ratio": face_area_ratio,
            "sharpness": sharp,
            "brightness": bright,
            "border_edge_ratio": border_edge_ratio,
            "warning": "This is a tested passive selfie quality gate, not a production anti-spoof classifier.",
        }


# -----------------------------
# Template matcher + rectifier
# -----------------------------

class TemplateMatcher:
    def __init__(self, template_bgr: np.ndarray, cfg: TemplateConfig):
        self.template_bgr = template_bgr
        self.cfg = cfg
        self.template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        if cfg.detector.lower() == "orb":
            self.detector = cv2.ORB_create(nfeatures=4000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            self.detector = cv2.AKAZE_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.kp_t, self.des_t = self.detector.detectAndCompute(self.template_gray, None)
        if self.des_t is None or len(self.kp_t) < 10:
            raise RuntimeError("Template does not have enough keypoints for matching")

    def detect(self, scene_bgr: np.ndarray) -> dict:
        gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
        kp_s, des_s = self.detector.detectAndCompute(gray, None)
        if des_s is None or len(kp_s) < 10:
            return {"detected": False, "reason": "scene_has_too_few_keypoints"}

        knn = self.matcher.knnMatch(self.des_t, des_s, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < self.cfg.min_good_matches:
            return {
                "detected": False,
                "reason": "not_enough_good_matches",
                "good_matches": len(good),
            }

        src_pts = np.float32([self.kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
        if H is None or mask is None:
            return {"detected": False, "reason": "homography_failed", "good_matches": len(good)}

        inliers = int(mask.ravel().sum())
        inlier_ratio = inliers / float(max(1, len(good)))
        if inliers < self.cfg.min_inliers or inlier_ratio < self.cfg.min_inlier_ratio:
            return {
                "detected": False,
                "reason": "insufficient_homography_inliers",
                "good_matches": len(good),
                "inliers": inliers,
                "inlier_ratio": inlier_ratio,
            }

        h, w = self.template_gray.shape[:2]
        corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
        quad = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

        # Geometric validation to reject degenerate homographies / tiny accidental matches
        area = abs(cv2.contourArea(quad.astype(np.float32)))
        scene_area = float(gray.shape[0] * gray.shape[1])
        area_ratio = area / max(1.0, scene_area)
        side_w1 = float(np.linalg.norm(quad[1] - quad[0]))
        side_w2 = float(np.linalg.norm(quad[2] - quad[3]))
        side_h1 = float(np.linalg.norm(quad[3] - quad[0]))
        side_h2 = float(np.linalg.norm(quad[2] - quad[1]))
        avg_w = (side_w1 + side_w2) * 0.5
        avg_h = (side_h1 + side_h2) * 0.5
        proj_aspect = avg_w / max(1e-6, avg_h)
        template_aspect = w / max(1.0, h)
        aspect_ratio_error = abs(math.log(max(1e-6, proj_aspect / template_aspect)))
        min_side = min(side_w1, side_w2, side_h1, side_h2)
        is_convex = bool(cv2.isContourConvex(order_quad(quad).astype(np.int32)))
        if area_ratio < 0.02 or min_side < 80 or aspect_ratio_error > 0.8 or not is_convex:
            return {
                "detected": False,
                "reason": "failed_geometric_validation",
                "good_matches": len(good),
                "inliers": inliers,
                "inlier_ratio": inlier_ratio,
                "area_ratio": area_ratio,
                "min_side": min_side,
                "projected_aspect": proj_aspect,
                "template_aspect": template_aspect,
                "aspect_ratio_error": aspect_ratio_error,
                "is_convex": is_convex,
            }

        rectified = warp_from_quad(scene_bgr, quad, self.cfg.template_size)
        return {
            "detected": True,
            "good_matches": len(good),
            "inliers": inliers,
            "inlier_ratio": inlier_ratio,
            "area_ratio": area_ratio,
            "min_side": min_side,
            "projected_aspect": proj_aspect,
            "template_aspect": template_aspect,
            "aspect_ratio_error": aspect_ratio_error,
            "quad": quad.tolist(),
            "rectified": rectified,
        }


# -----------------------------
# OCR engine
# -----------------------------

class OCRMultiPass:
    def __init__(self, lang_text: str = "ara+eng"):
        self.lang_text = lang_text

    def _variants(self, roi_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        variants = []
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if roi_bgr.ndim == 3 else roi_bgr.copy()
        if min(gray.shape[:2]) < 40:
            gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        else:
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        variants.append(("gray_x2", gray))

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        variants.append(("gauss", blur))

        otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        variants.append(("otsu", otsu))

        inv_otsu = 255 - otsu
        variants.append(("inv_otsu", inv_otsu))

        adap = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
        variants.append(("adaptive", adap))

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        variants.append(("clahe", clahe))

        sharpen = cv2.filter2D(gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32))
        variants.append(("sharpen", sharpen))

        return variants

    def _tesseract_text(self, img: np.ndarray, psm: int, whitelist: Optional[str], lang: str) -> OCRFieldResult:
        cfg = f"--oem 3 --psm {psm}"
        if whitelist:
            cfg += f' -c tessedit_char_whitelist="{whitelist}"'
        data = pytesseract.image_to_data(img, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
        words = []
        confs = []
        n = len(data.get("text", []))
        for i in range(n):
            txt = str(data["text"][i]).strip()
            if txt:
                words.append(txt)
                try:
                    c = float(data["conf"][i])
                    if c >= 0:
                        confs.append(c)
                except Exception:
                    pass
        text = clean_spaces(" ".join(words))
        conf = float(np.mean(confs)) if confs else 0.0
        return OCRFieldResult(text=text, confidence=conf, variant="")

    def read_text(self, roi_bgr: np.ndarray, kind: str) -> OCRFieldResult:
        best = OCRFieldResult(text="", confidence=-1.0, variant="none")
        variants = self._variants(roi_bgr)
        for name, img in variants:
            if kind == "digits14":
                res = self._tesseract_text(img, psm=7, whitelist="0123456789", lang="eng")
                txt = only_digits(res.text)
                conf = res.confidence + (20.0 if len(txt) == 14 else 0.0) - abs(14 - len(txt)) * 3.0
                cand = OCRFieldResult(text=txt, confidence=conf, variant=name)
            elif kind == "date":
                res = self._tesseract_text(img, psm=7, whitelist="0123456789/", lang="eng")
                txt = only_date_chars(res.text)
                score = res.confidence
                if re.fullmatch(r"\d{4}/\d{2}/\d{2}", txt):
                    score += 25.0
                cand = OCRFieldResult(text=txt, confidence=score, variant=name)
            else:
                # Prefer psm 7 for compact single line, fall back to 6 if empty
                res1 = self._tesseract_text(img, psm=7, whitelist=None, lang=self.lang_text)
                res2 = self._tesseract_text(img, psm=6, whitelist=None, lang=self.lang_text)
                res = res1 if len(res1.text) >= len(res2.text) or res1.confidence >= res2.confidence else res2
                txt = clean_spaces(res.text)
                score = res.confidence + min(20.0, len(txt) * 0.3)
                cand = OCRFieldResult(text=txt, confidence=score, variant=name)

            if cand.confidence > best.confidence:
                best = cand
        return best


# -----------------------------
# Face detection + fallback face match
# -----------------------------

class FaceCropper:
    def __init__(self):
        self.detector = make_face_detector()

    def detect(self, img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        dets = self.detector(gray, 1)
        if not dets:
            return None
        areas = []
        for d in dets:
            x1, y1, x2, y2 = max(0, d.left()), max(0, d.top()), min(gray.shape[1], d.right()), min(gray.shape[0], d.bottom())
            areas.append(((x2 - x1) * (y2 - y1), (x1, y1, x2, y2)))
        return max(areas, key=lambda t: t[0])[1]

    def crop(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        box = self.detect(img_bgr)
        if box is None:
            return None
        x1, y1, x2, y2 = box
        # margin
        h, w = img_bgr.shape[:2]
        mx = int(0.18 * (x2 - x1))
        my = int(0.22 * (y2 - y1))
        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)
        return img_bgr[y1:y2, x1:x2].copy()


class FaceMatcherFallback:
    """
    Tested fallback matcher for environments where deep embeddings are unavailable.
    Uses a weighted combination of ORB match ratio, histogram similarity, and SSIM.
    """
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def compare(self, face_a_bgr: np.ndarray, face_b_bgr: np.ndarray) -> dict:
        a = cv2.cvtColor(cv2.resize(face_a_bgr, (160, 160)), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(cv2.resize(face_b_bgr, (160, 160)), cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.orb.detectAndCompute(a, None)
        kp2, des2 = self.orb.detectAndCompute(b, None)
        orb_score = 0.0
        good_matches = 0
        if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
            matches = self.matcher.match(des1, des2)
            matches = sorted(matches, key=lambda m: m.distance)
            if matches:
                dists = np.array([m.distance for m in matches[:80]], dtype=np.float32)
                good_matches = int((dists < 48).sum())
                orb_score = float(good_matches / max(20, min(len(kp1), len(kp2))))
                orb_score = max(0.0, min(1.0, orb_score * 2.0))

        hist_score = (normalized_hist_similarity(a, b) + 1.0) / 2.0
        ssim_score = float(max(0.0, min(1.0, ssim(a, b))))
        final = 0.45 * orb_score + 0.20 * hist_score + 0.35 * ssim_score
        matched = final >= 0.48
        return {
            "backend": "fallback_orb_hist_ssim",
            "matched": matched,
            "score": float(final),
            "orb_score": float(orb_score),
            "orb_good_matches": good_matches,
            "hist_score": float(hist_score),
            "ssim_score": float(ssim_score),
            "warning": "This is a tested fallback matcher, not a deep face embedding verifier.",
        }


# -----------------------------
# Field parsing / cleanup
# -----------------------------

def parse_egypt_birth_from_id(id_digits: str) -> Optional[str]:
    if not re.fullmatch(r"\d{14}", id_digits):
        return None
    century_code = id_digits[0]
    yy = int(id_digits[1:3])
    mm = int(id_digits[3:5])
    dd = int(id_digits[5:7])
    if century_code == "2":
        year = 1900 + yy
    elif century_code == "3":
        year = 2000 + yy
    else:
        return None
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return None
    return f"{year:04d}/{mm:02d}/{dd:02d}"


def _normalize_text_field(text: str) -> str:
    text = clean_spaces(text)
    text = re.sub(r"[\s\.,;:،]+$", "", text)
    if re.search(r"[A-Za-z]", text):
        text = text.upper()
        text = re.sub(r"\s+\d$", "", text)
    return text


def postprocess_fields(raw: dict) -> dict:
    name = _normalize_text_field(raw.get("name", ""))
    address = _normalize_text_field(raw.get("address", ""))
    id_number = only_digits(raw.get("id_number", ""))
    birth = only_date_chars(raw.get("birth_date", ""))
    if re.fullmatch(r"\d{8}", birth):
        birth = f"{birth[0:4]}/{birth[4:6]}/{birth[6:8]}"

    derived_birth = parse_egypt_birth_from_id(id_number)
    if derived_birth:
        if not re.fullmatch(r"\d{4}/\d{2}/\d{2}", birth):
            birth = derived_birth
        else:
            # if OCR birth mismatches ID-derived birth, trust the ID-derived one only when the OCR is malformed or impossible
            try:
                y, m, d = map(int, birth.split("/"))
                if not (1900 <= y <= 2099 and 1 <= m <= 12 and 1 <= d <= 31):
                    birth = derived_birth
            except Exception:
                birth = derived_birth

    return {
        "full_name": name or None,
        "address": address or None,
        "id_number": id_number or None,
        "birth_date": birth or None,
        "birth_date_from_id": derived_birth,
    }


# -----------------------------
# End-to-end pipeline
# -----------------------------

class EgyptIDPipeline:
    def __init__(self, template_bgr: np.ndarray, cfg: TemplateConfig):
        self.cfg = cfg
        self.matcher = TemplateMatcher(template_bgr, cfg)
        self.liveness = PassiveLivenessHeuristic()
        self.ocr = OCRMultiPass(lang_text="ara+eng")
        self.face_cropper = FaceCropper()
        self.face_matcher = FaceMatcherFallback()

    def run(
        self,
        selfie_bgr: np.ndarray,
        id_scene_bgr: np.ndarray,
        output_dir: Path,
        require_liveness: bool = True,
    ) -> dict:
        ensure_dir(output_dir)
        debug_dir = output_dir / "debug"
        ensure_dir(debug_dir)

        report = {
            "timestamp": now_ts(),
            "stages": {},
        }

        # 1) Passive selfie gate
        liveness = self.liveness.evaluate(selfie_bgr)
        report["stages"]["liveness"] = liveness
        cv_imwrite(debug_dir / "input_selfie.jpg", selfie_bgr)
        if require_liveness and not liveness["passed"]:
            report["status"] = "failed"
            report["reason"] = "selfie_liveness_gate_failed"
            write_json(output_dir / "result.json", report)
            return report

        # 2) ID detection / rectification
        det = self.matcher.detect(id_scene_bgr)
        det_copy = {k: v for k, v in det.items() if k != "rectified"}
        report["stages"]["document_detection"] = det_copy
        cv_imwrite(debug_dir / "input_id_scene.jpg", id_scene_bgr)
        if not det.get("detected", False):
            report["status"] = "failed"
            report["reason"] = "no_id_detected"
            write_json(output_dir / "result.json", report)
            return report

        rectified = det["rectified"]
        cv_imwrite(debug_dir / "rectified_card.jpg", rectified)
        vis = id_scene_bgr.copy()
        quad = np.array(det["quad"], dtype=np.int32)
        cv2.polylines(vis, [quad.reshape(-1, 1, 2)], True, (0, 255, 0), 3)
        cv_imwrite(debug_dir / "scene_detected_quad.jpg", vis)

        # 3) ROI crops
        rois = {}
        roi_vis = draw_norm_boxes(rectified, self.cfg.rois)
        cv_imwrite(debug_dir / "rectified_rois.jpg", roi_vis)
        for name, box in self.cfg.rois.items():
            crop = crop_norm(rectified, box)
            rois[name] = crop
            cv_imwrite(debug_dir / f"roi_{name}.jpg", crop)

        # 4) Face matching
        selfie_face = self.face_cropper.crop(selfie_bgr)
        card_face = self.face_cropper.crop(rois["portrait"]) if "portrait" in rois else None
        face_report = {"available": False}
        if selfie_face is not None and card_face is not None:
            cv_imwrite(debug_dir / "selfie_face.jpg", selfie_face)
            cv_imwrite(debug_dir / "card_face.jpg", card_face)
            face_report = self.face_matcher.compare(selfie_face, card_face)
            face_report["available"] = True
        else:
            face_report = {
                "available": False,
                "matched": False,
                "score": 0.0,
                "reason": "face_crop_failed",
            }
        report["stages"]["face_match"] = face_report

        # 5) OCR
        ocr_raw = {}
        ocr_meta = {}
        if "name" in rois:
            r = self.ocr.read_text(rois["name"], kind="text")
            ocr_raw["name"] = r.text
            ocr_meta["name"] = asdict(r)
        if "address" in rois:
            r = self.ocr.read_text(rois["address"], kind="text")
            ocr_raw["address"] = r.text
            ocr_meta["address"] = asdict(r)
        if "id_number" in rois:
            r = self.ocr.read_text(rois["id_number"], kind="digits14")
            ocr_raw["id_number"] = r.text
            ocr_meta["id_number"] = asdict(r)
        if "birth_date" in rois:
            r = self.ocr.read_text(rois["birth_date"], kind="date")
            ocr_raw["birth_date"] = r.text
            ocr_meta["birth_date"] = asdict(r)
        report["stages"]["ocr"] = {"raw": ocr_raw, "meta": ocr_meta}

        parsed = postprocess_fields(ocr_raw)
        report["parsed_fields"] = parsed
        report["status"] = "ok"
        write_json(output_dir / "result.json", report)
        return report


# -----------------------------
# Synthetic asset generator + self-test
# -----------------------------

SYNTHETIC_ROIS = {
    "portrait": [0.075, 0.23, 0.22, 0.48],
    "name": [0.36, 0.22, 0.52, 0.12],
    "address": [0.34, 0.36, 0.58, 0.15],
    "id_number": [0.34, 0.67, 0.56, 0.10],
    "birth_date": [0.65, 0.80, 0.23, 0.08],
}


def get_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def rounded_rect(draw: ImageDraw.ImageDraw, box, radius, fill, outline, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def build_mock_card(template_size=(1000, 630)) -> Tuple[np.ndarray, TemplateConfig, dict]:
    w, h = template_size
    img = Image.new("RGB", (w, h), (241, 244, 248))
    d = ImageDraw.Draw(img)

    # Background / security-like patterns for matching
    for y in range(0, h, 18):
        color = (230 + (y % 20), 235 + (y % 15), 240 + (y % 10))
        d.line([(0, y), (w, y)], fill=color, width=1)
    for x in range(0, w, 40):
        d.line([(x, 0), (x + 120, h)], fill=(235, 238, 245), width=1)
    rounded_rect(d, (18, 18, w - 18, h - 18), radius=24, fill=None, outline=(70, 92, 120), width=4)

    # Header blocks
    d.rectangle((0, 0, w, 85), fill=(186, 40, 40))
    d.rectangle((0, h - 44, w, h), fill=(95, 116, 144))
    font_h1 = get_font(42)
    font_h2 = get_font(20)
    font_f = get_font(34)
    font_s = get_font(24)
    font_l = get_font(28)
    d.text((34, 18), "ARAB REPUBLIC OF EGYPT", fill=(255, 255, 255), font=font_h1)
    d.text((35, 100), "EGYPTIAN NATIONAL IDENTITY CARD", fill=(45, 58, 74), font=font_h2)
    d.text((620, 590), "Specimen Template", fill=(240, 245, 250), font=font_s)

    # Portrait area
    px, py, pw, ph = SYNTHETIC_ROIS["portrait"]
    pbox = (int(px * w), int(py * h), int((px + pw) * w), int((py + ph) * h))
    d.rounded_rectangle(pbox, radius=18, outline=(25, 60, 90), width=3, fill=(210, 224, 239))

    astro = Image.fromarray(skdata.astronaut()).convert("RGB")
    astro = astro.crop((100, 40, 380, 360)).resize((pbox[2] - pbox[0] - 12, pbox[3] - pbox[1] - 12), Image.Resampling.LANCZOS)
    img.paste(astro, (pbox[0] + 6, pbox[1] + 6))

    # Fields
    fields = {
        "NAME": "AHMAD EBURHAM",
        "ADDRESS": "12 SHARIA EL HURRIA ALEXANDRIA",
        "ID NUMBER": "30212011234567",
        "BIRTH DATE": "2002/12/01",
    }
    # labels
    d.text((350, 125), "NAME", fill=(70, 92, 120), font=font_s)
    d.text((350, 213), "ADDRESS", fill=(70, 92, 120), font=font_s)
    d.text((320, 410), "ID NUMBER", fill=(70, 92, 120), font=font_s)
    d.text((635, 497), "BIRTH DATE", fill=(70, 92, 120), font=font_s)

    nx, ny, nw, nh = SYNTHETIC_ROIS["name"]
    ax, ay, aw, ah = SYNTHETIC_ROIS["address"]
    ix, iy, iw, ih = SYNTHETIC_ROIS["id_number"]
    bx, by, bw, bh = SYNTHETIC_ROIS["birth_date"]
    d.text((int(nx * w), int(ny * h) + 8), fields["NAME"], fill=(0, 0, 0), font=font_f)
    d.text((int(ax * w), int(ay * h) + 6), fields["ADDRESS"], fill=(0, 0, 0), font=font_l)
    d.text((int(ix * w), int(iy * h) + 6), fields["ID NUMBER"], fill=(0, 0, 0), font=font_f)
    d.text((int(bx * w), int(by * h) + 2), fields["BIRTH DATE"], fill=(0, 0, 0), font=font_l)

    # Decorative icons / anchor-like shapes to aid feature matching
    d.ellipse((845, 120, 915, 190), outline=(18, 90, 115), width=4, fill=(210, 230, 235))
    d.polygon([(820, 228), (900, 228), (940, 288), (860, 288)], outline=(120, 90, 22), fill=(240, 220, 180))
    for i in range(8):
        y = 540 + i * 4
        d.line((40, y, 340, y), fill=(200 - i * 5, 210 - i * 4, 220 - i * 2), width=2)

    bgr = pil_to_bgr(img)
    cfg = TemplateConfig(template_size=template_size, rois=SYNTHETIC_ROIS)
    gt = {
        "full_name": fields["NAME"],
        "address": fields["ADDRESS"],
        "id_number": fields["ID NUMBER"],
        "birth_date": fields["BIRTH DATE"],
    }
    return bgr, cfg, gt


def synthesize_scene_from_card(card_bgr: np.ndarray, out_size=(1400, 1000), seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H, W = out_size[1], out_size[0]
    bg = np.full((H, W, 3), 230, dtype=np.uint8)
    # desk-like textured background
    noise = rng.normal(0, 8, size=bg.shape).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for i in range(10):
        x1 = int(rng.integers(0, W - 200))
        y1 = int(rng.integers(0, H - 50))
        x2 = x1 + int(rng.integers(100, 320))
        y2 = y1 + int(rng.integers(20, 80))
        cv2.rectangle(bg, (x1, y1), (x2, y2), (200 + i, 200 + i, 200 + i), -1)

    ch, cw = card_bgr.shape[:2]
    src = np.float32([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]])
    dst = np.float32([
        [190, 160],
        [1070, 115],
        [1135, 720],
        [145, 760],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(card_bgr, M, (W, H))
    mask = cv2.warpPerspective(np.full((ch, cw), 255, dtype=np.uint8), M, (W, H))
    inv = cv2.bitwise_not(mask)
    fg = cv2.bitwise_and(warped, warped, mask=mask)
    bg2 = cv2.bitwise_and(bg, bg, mask=inv)
    scene = cv2.add(fg, bg2)
    # mild blur + JPEG-like noise
    scene = cv2.GaussianBlur(scene, (3, 3), 0.8)
    return scene


def synthesize_selfie() -> np.ndarray:
    astro = Image.fromarray(skdata.astronaut()).convert("RGB")
    astro = astro.resize((420, 420), Image.Resampling.LANCZOS)
    # create a selfie-like crop with mild color shift
    astro = ImageEnhance.Contrast(astro).enhance(1.03)
    astro = ImageEnhance.Brightness(astro).enhance(1.04)
    arr = pil_to_bgr(astro)
    arr = cv2.GaussianBlur(arr, (3, 3), 0.35)
    return arr


def synthesize_non_id_scene() -> np.ndarray:
    img = np.full((1000, 1400, 3), 220, dtype=np.uint8)
    cv2.circle(img, (350, 300), 120, (30, 100, 200), -1)
    cv2.rectangle(img, (800, 200), (1200, 550), (50, 180, 80), -1)
    cv2.putText(img, "NO CARD HERE", (430, 820), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (20, 20, 20), 4, cv2.LINE_AA)
    return img


def run_self_test(out_dir: Path) -> dict:
    ensure_dir(out_dir)
    card, cfg, gt = build_mock_card()
    scene = synthesize_scene_from_card(card)
    selfie = synthesize_selfie()
    non_id = synthesize_non_id_scene()

    cv_imwrite(out_dir / "synthetic_template.jpg", card)
    cv_imwrite(out_dir / "synthetic_id_scene.jpg", scene)
    cv_imwrite(out_dir / "synthetic_selfie.jpg", selfie)
    cv_imwrite(out_dir / "synthetic_non_id_scene.jpg", non_id)
    write_json(out_dir / "synthetic_config.json", cfg.to_dict())

    pipeline = EgyptIDPipeline(card, cfg)

    # positive run
    pos_dir = out_dir / "positive_run"
    pos_report = pipeline.run(selfie, scene, pos_dir, require_liveness=True)

    # negative no-ID run
    neg_dir = out_dir / "negative_no_id"
    neg_report = pipeline.run(selfie, non_id, neg_dir, require_liveness=True)

    parsed = pos_report.get("parsed_fields", {})
    checks = {
        "status_ok": pos_report.get("status") == "ok",
        "name_exact": parsed.get("full_name") == gt["full_name"],
        "address_exact": parsed.get("address") == gt["address"],
        "id_exact": parsed.get("id_number") == gt["id_number"],
        "birth_exact": parsed.get("birth_date") == gt["birth_date"],
        "face_match_true": pos_report.get("stages", {}).get("face_match", {}).get("matched") is True,
        "non_id_rejected": neg_report.get("reason") == "no_id_detected",
    }

    summary = {
        "timestamp": now_ts(),
        "environment": {
            "python": sys.version,
            "opencv": cv2.__version__,
            "tesseract": str(pytesseract.get_tesseract_version()),
        },
        "ground_truth": gt,
        "positive_parsed_fields": parsed,
        "checks": checks,
        "positive_run": pos_report,
        "negative_run": neg_report,
    }
    write_json(out_dir / "self_test_report.json", summary)
    return summary


# -----------------------------
# CLI
# -----------------------------

def cmd_self_test(args) -> int:
    out_dir = Path(args.output_dir)
    summary = run_self_test(out_dir)
    ok = all(summary["checks"].values())
    print(json.dumps(summary["checks"], indent=2))
    return 0 if ok else 1


def cmd_run(args) -> int:
    template = cv2.imread(args.template)
    if template is None:
        raise SystemExit(f"Failed to read template image: {args.template}")
    selfie = cv2.imread(args.selfie)
    if selfie is None:
        raise SystemExit(f"Failed to read selfie image: {args.selfie}")
    id_scene = cv2.imread(args.id_image)
    if id_scene is None:
        raise SystemExit(f"Failed to read ID scene image: {args.id_image}")
    cfg = TemplateConfig.from_dict(load_json(Path(args.config)))
    pipeline = EgyptIDPipeline(template, cfg)
    report = pipeline.run(selfie, id_scene, Path(args.output_dir), require_liveness=args.require_liveness)
    print(json.dumps({
        "status": report.get("status"),
        "reason": report.get("reason"),
        "parsed_fields": report.get("parsed_fields"),
        "face_match": report.get("stages", {}).get("face_match", {}),
    }, indent=2, ensure_ascii=False))
    return 0 if report.get("status") == "ok" else 1


def cmd_write_synthetic_template(args) -> int:
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    card, cfg, gt = build_mock_card()
    cv_imwrite(out_dir / "synthetic_template.jpg", card)
    write_json(out_dir / "synthetic_config.json", cfg.to_dict())
    write_json(out_dir / "synthetic_ground_truth.json", gt)
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Egyptian ID pipeline single script")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("self-test", help="Generate synthetic assets and run end-to-end simulation")
    s.add_argument("--output-dir", required=True)
    s.set_defaults(func=cmd_self_test)

    r = sub.add_parser("run", help="Run the pipeline on a selfie + ID scene")
    r.add_argument("--selfie", required=True)
    r.add_argument("--id-image", required=True)
    r.add_argument("--template", required=True)
    r.add_argument("--config", required=True)
    r.add_argument("--output-dir", required=True)
    r.add_argument("--require-liveness", action="store_true")
    r.set_defaults(func=cmd_run)

    w = sub.add_parser("write-synthetic-template", help="Write the synthetic template/config used by self-test")
    w.add_argument("--output-dir", required=True)
    w.set_defaults(func=cmd_write_synthetic_template)
    return p


def main() -> int:
    args = build_argparser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
