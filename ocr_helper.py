#!/usr/bin/env python3
"""
Helper OCR: recibe líneas "frame_num /ruta/imagen.png" por stdin,
devuelve líneas "frame_num valor confianza" por stdout.
Usado por leer_display.cpp vía popen.
"""
import sys, cv2, easyocr, os

reader = easyocr.Reader(['en'], gpu=False, verbose=False)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    parts = line.split()
    if len(parts) != 2:
        continue
    frame_num, img_path = parts[0], parts[1]

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"{frame_num} 0 0.0", flush=True)
        continue

    res = reader.readtext(img, allowlist='0123456789', detail=1)
    nums = [x for x in res if len(x[1]) >= 2]
    txt  = ''.join(x[1] for x in nums)
    conf = round(sum(x[2] for x in nums) / len(nums), 2) if nums else 0.0

    try:
        val = int(txt)
        if not (10 <= val <= 400):
            val = 0
    except Exception:
        val = 0

    print(f"{frame_num} {val} {conf}", flush=True)

os.remove(img_path) if img_path.startswith('/tmp/') else None
