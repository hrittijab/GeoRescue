from pathlib import Path
from google.cloud import storage

BUCKET_NAME = "sen1floods11"
BASE = "v1.1/data/flood_events/HandLabeled"
OUTPUT = Path("data/raw")
SAMPLE_COUNT = 5

client = storage.Client.create_anonymous_client()
bucket = client.bucket(BUCKET_NAME)

print("Finding satellite images...")

images = sorted(
    blob.name
    for blob in client.list_blobs(
        BUCKET_NAME,
        prefix=f"{BASE}/S1Hand/"
    )
    if blob.name.endswith("_S1Hand.tif")
)

if not images:
    raise RuntimeError("No images found in the dataset.")

# Prefer images from different countries/events.
selected = []
seen_regions = set()

for image in images:
    filename = Path(image).name
    region = filename.rsplit("_", 2)[0]

    if region not in seen_regions:
        selected.append(image)
        seen_regions.add(region)

    if len(selected) == SAMPLE_COUNT:
        break

# Fill remaining slots if fewer than 5 regions exist.
for image in images:
    if len(selected) >= SAMPLE_COUNT:
        break
    if image not in selected:
        selected.append(image)

for index, image in enumerate(selected, start=1):
    filename = Path(image).name
    label_filename = filename.replace(
        "_S1Hand.tif", "_LabelHand.tif"
    )

    files = [
        (image, OUTPUT / "S1Hand" / filename),
        (
            f"{BASE}/LabelHand/{label_filename}",
            OUTPUT / "LabelHand" / label_filename
        ),
    ]

    for source, destination in files:
        destination.parent.mkdir(
            parents=True, exist_ok=True
        )

        if destination.exists():
            print(f"Already exists: {destination.name}")
            continue

        print(f"Downloading {destination.name}")
        bucket.blob(source).download_to_filename(
            str(destination)
        )

    print(f"Completed pair {index}/{len(selected)}")

print("\nSatellite dataset download complete!")
