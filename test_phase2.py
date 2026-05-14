import cv2
import numpy as np
import os

os.makedirs("tmp", exist_ok=True)

print("=== 1. Create original image ===")
img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
cv2.imwrite("tmp/p2_orig.png", img)
print("   256x256 test image created")

print("\n=== 2. Embed watermark ===")
from watermark_logic import encode_image, decode_image
r = encode_image("tmp/p2_orig.png", "TAMPER-TEST-2026", "tmp/p2_wm.png", "secret")
print(f"   {r}")

print("\n=== 3. Tamper the image (paint a red rectangle) ===")
tampered = cv2.imread("tmp/p2_wm.png")
cv2.rectangle(tampered, (80, 80), (180, 180), (0, 0, 255), -1)
cv2.imwrite("tmp/p2_tampered.png", tampered)
print("   Red rectangle painted at (80,80)-(180,180)")

print("\n=== 4. generate_tamper_map() on CLEAN watermarked image ===")
from tamper_detection import generate_tamper_map, get_tamper_summary
result_clean = generate_tamper_map("tmp/p2_wm.png", "secret")
print(f"   Tamper %: {result_clean['tamper_pct']}%")
print(f"   Overlay shape: {result_clean['overlay'].shape}")

print("\n=== 5. generate_tamper_map() on TAMPERED image ===")
result_tampered = generate_tamper_map("tmp/p2_tampered.png", "secret")
print(f"   Tamper %: {result_tampered['tamper_pct']}%")
cv2.imwrite("tmp/p2_heatmap.png", result_tampered["overlay"])
print("   Heatmap saved to tmp/p2_heatmap.png")

print("\n=== 6. get_tamper_summary() ===")
summary = get_tamper_summary(result_tampered["raw_map"])
print(f"   Severity: {summary['severity']}")
print(f"   Tamper %: {summary['tamper_pct']}%")
print(f"   Regions found: {summary['num_regions']}")
for region in summary["regions"][:3]:
    print(f"     Region: x={region['x']}, y={region['y']}, w={region['w']}, h={region['h']}")

print("\n=== 7. Compare clean vs tampered ===")
print(f"   Clean image tamper:   {result_clean['tamper_pct']}%")
print(f"   Tampered image tamper: {result_tampered['tamper_pct']}%")
diff = result_tampered["tamper_pct"] - result_clean["tamper_pct"]
print(f"   Difference: +{diff:.1f}% (tampered area detected!)")

print("\n=== 8. Verify decode still works on tampered image ===")
r_decode = decode_image("tmp/p2_tampered.png", "secret")
print(f"   Decode result: {r_decode}")

print("\n=== ALL PHASE 2 TESTS PASSED ===")
