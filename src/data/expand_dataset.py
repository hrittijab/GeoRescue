from pathlib import Path
from collections import defaultdict
from google.cloud import storage

BUCKET_NAME = "sen1floods11"
BASE = "v1.1/data/flood_events/HandLabeled"
MAX_PER_EVENT = 10

OUTPUT = Path("data/raw")

client = storage.Client.create_anonymous_client()
bucket = client.bucket(BUCKET_NAME)

print("Discovering available satellite imagery...")

image_blobs = sorted(
    blob.name
    for blob in client.list_blobs(
        BUCKET_NAME,
        prefix=f"{BASE}/S1Hand/"
    )
    if blob.name.endswith("_S1Hand.tif")
)

label_blobs = {
    blob.name
    for blob in client.list_blobs(
        BUCKET_NAME,
        prefix=f"{BASE}/LabelHand/"
    )
    if blob.name.endswith("_LabelHand.tif")
}

events = defaultdict(list)

for image in image_blobs:
    filename = Path(image).name
    event = filename.split("_")[0]

    label_name = filename.replace(
        "_S1Hand.tif", "_LabelHand.tif"
    )
    label_blob = f"{BASE}/LabelHand/{label_name}"

    if label_blob in label_blobs:
        events[event].append((image, label_blob))

print(f"Found {len(events)} event groups")

downloaded = 0
existing = 0
failed = 0

for event, pairs in sorted(events.items()):
    print(f"\nEvent: {event} | Available: {len(pairs)}")

    # Include previously downloaded examples.
    already_local = []

    for image_blob, label_blob in pairs:
        image_path = OUTPUT / "S1Hand" / Path(image_blob).name
        label_path = OUTPUT / "LabelHand" / Path(label_blob).name

        if image_path.exists() and label_path.exists():
            already_local.append((image_blob, label_blob))

    remaining = [
        pair for pair in pairs
        if pair not in already_local
    ]

    selected = (
        already_local[:MAX_PER_EVENT] +
        remaining[
            :max(0, MAX_PER_EVENT - len(already_local))
        ]
    )

    for image_blob, label_blob in selected:
        pair_ok = True

        for remote, folder in [
            (image_blob, "S1Hand"),
            (label_blob, "LabelHand")
        ]:
            destination = OUTPUT / folder / Path(remote).name
            destination.parent.mkdir(
                parents=True, exist_ok=True
            )

            if destination.exists():
                existing += 1
                continue

            try:
                bucket.blob(remote).download_to_filename(
                    str(destination)
                )
                downloaded += 1
            except Exception as exc:
                failed += 1
                pair_ok = False
                print(f"Failed: {destination.name}: {exc}")
                destination.unlink(missing_ok=True)

        if pair_ok:
            print(f"  Ready: {Path(image_blob).name}")

print("\n" + "=" * 50)
print("GEORESCUE DATASET EXPANSION COMPLETE")
print("=" * 50)
print(f"New files downloaded: {downloaded}")
print(f"Existing files reused: {existing}")
print(f"Failed downloads: {failed}")

image_count = len(list((OUTPUT / "S1Hand").glob("*.tif")))
label_count = len(list((OUTPUT / "LabelHand").glob("*.tif")))

print(f"Total satellite files: {image_count}")
print(f"Total label files: {label_count}")
