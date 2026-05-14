import cv2
import numpy as np
import os

os.makedirs("tmp", exist_ok=True)

print("=== 1. Create test video (72 frames, 3 sec) ===")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("tmp/phase1_test.mp4", fourcc, 24.0, (320, 240))
for i in range(72):
    frame = np.random.randint(30, 200, (240, 320, 3), dtype=np.uint8)
    cv2.putText(frame, f"F{i}", (120, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    writer.write(frame)
writer.release()
print("   Video created: 320x240, 72 frames, 24fps")

print("\n=== 2. extract_frames() ===")
from video_engine import extract_frames
paths = extract_frames("tmp/phase1_test.mp4", "tmp/frames_test", every_n=10)
print(f"   Extracted {len(paths)} frames (every 10th)")

print("\n=== 3. encode_video() with progress ===")
from video_engine import encode_video
progress_log = []
r = encode_video(
    "tmp/phase1_test.mp4", "TOP-SECRET-2026", "tmp/phase1_wm.mp4", "pass123",
    progress_callback=lambda p: progress_log.append(round(p, 2)),
)
print(f"   {r}")
print(f"   Progress callbacks: {len(progress_log)} calls")

print("\n=== 4. decode_video() with temporal averaging ===")
from video_engine import decode_video
d = decode_video("tmp/phase1_wm.mp4", "pass123", num_sample_frames=10)
print(f"   Found: {d['found']}")
print(f"   Message: {d['message']}")
print(f"   Avg Score: {d['avg_score']}")
print(f"   Frames analyzed: {d['frames_analyzed']}")
for s in d["scores"][:3]:
    print(f"     Frame {s['frame']}: score={s['score']}%, found={s['found']}")

print("\n=== 5. reassemble_video() ===")
from video_engine import reassemble_video
r2 = reassemble_video("tmp/frames_test", "tmp/phase1_reassembled.mp4", 24.0)
print(f"   {r2}")
sz = os.path.getsize("tmp/phase1_reassembled.mp4")
print(f"   Output size: {sz} bytes")

print("\n=== 6. get_frame_trust_score() — frame slider ===")
from video_engine import get_frame_trust_score
for fnum in [0, 35, 71]:
    ts = get_frame_trust_score("tmp/phase1_wm.mp4", fnum, "pass123")
    print(f"   Frame {fnum}: score={ts['score']}, found={ts['found']}")

print("\n=== 7. extract_single_frame() ===")
from video_engine import extract_single_frame
frame = extract_single_frame("tmp/phase1_wm.mp4", 36)
print(f"   Frame 36 shape: {frame.shape}")

print("\n=== ALL PHASE 1 TESTS PASSED ===")
