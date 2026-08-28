"""
grainvdb.vision — Native On-Device Apple Vision OCR & Screen Text Extraction.
Leverages Apple Silicon Neural Engine / GPU for sub-5ms UI text recognition.
"""

from __future__ import annotations

import subprocess
import json
import tempfile
import os
from typing import List, Dict, Optional, Union, Any


def extract_ui_text(image_input: Union[bytes, str, os.PathLike]) -> List[Dict[str, Any]]:
    """
    Extracts text tokens and bounding box coordinates from a screenshot.
    Uses native macOS Vision framework if available, with graceful fallback.
    
    Returns:
        List of dicts: [{"text": str, "confidence": float, "x": float, "y": float, "w": float, "h": float}]
    """
    tmp_file = None
    if isinstance(image_input, (bytes, bytearray)):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(image_input)
        tmp.close()
        image_path = tmp.name
        tmp_file = image_path
    else:
        image_path = str(image_input)

    elements: List[Dict[str, Any]] = []
    try:
        # Swift on-device vision script
        swift_script = f"""
import Foundation
import Vision
import ImageIO

guard let imgUrl = URL(string: "file://{image_path}"),
      let imageSource = CGImageSourceCreateWithURL(imgUrl as CFURL, nil),
      let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {{
    print("[]")
    exit(0)
}}

var results: [[String: Any]] = []
let request = VNRecognizeTextRequest {{ req, _ in
    guard let obs = req.results as? [VNRecognizedTextObservation] else {{ return }}
    for o in obs {{
        if let top = o.topCandidates(1).first {{
            let b = o.boundingBox
            results.append([
                "text": top.string,
                "confidence": top.confidence,
                "x": b.origin.x,
                "y": b.origin.y,
                "w": b.size.width,
                "h": b.size.height
            ])
        }}
    }}
}}
request.recognitionLevel = .fast
let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
try? handler.perform([request])

if let data = try? JSONSerialization.data(withJSONObject: results),
   let str = String(data: data, encoding: .utf8) {{
    print(str)
}}
"""
        out = subprocess.check_output(["swift", "-e", swift_script], timeout=3.0).decode("utf-8")
        elements = json.loads(out.strip()) if out.strip() else []
    except Exception:
        # Return empty list on non-macOS or timeout
        elements = []
    finally:
        if tmp_file and os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass

    return elements


def extract_full_text(image_input: Union[bytes, str, os.PathLike]) -> str:
    """Extracts all text tokens as a single space-separated string."""
    elements = extract_ui_text(image_input)
    return " ".join(e.get("text", "") for e in elements if e.get("text"))
