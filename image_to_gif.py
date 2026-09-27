from pathlib import Path
import imageio.v2 as imageio

root = Path(".")

for camera in ["front", "wrist"]:
    paths = sorted(root.glob(f"episode_000_{camera}_step_*.png"))

    if not paths:
        print(f"No images found for {camera}")
        continue

    frames = [
        imageio.imread(path)
        for path in paths
    ]

    output = root / f"episode_000_{camera}.gif"

    imageio.mimsave(
        output,
        frames,
        duration=0.6,
        loop=0,
    )

    print("Saved:", output)