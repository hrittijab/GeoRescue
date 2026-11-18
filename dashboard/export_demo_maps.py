from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

source = (
    ROOT
    / "dashboard/pages/1_Model_Comparison.py"
)

code = source.read_text(encoding="utf-8-sig")

# Load the existing geographic processing functions
# without executing the dashboard interface.
marker = '# Dashboard'

position = code.find(marker)

if position == -1:
    raise RuntimeError(
        "Could not locate dashboard section."
    )

namespace = {
    "__file__": str(source),
    "__name__": "georescue_map_export",
}

exec(
    compile(code[:position], str(source), "exec"),
    namespace,
)

load_exposure = namespace["load_exposure"]

destination = ROOT / "dashboard/demo_data"
destination.mkdir(parents=True, exist_ok=True)

for version, filename in [
    ("U-Net V1", "exposure_v1.json"),
    ("U-Net V2", "exposure_v2.json"),
]:
    print(f"Generating {version} map layers...")

    data = load_exposure(version)

    output = destination / filename

    output.write_text(
        json.dumps(data, separators=(",", ":")),
        encoding="utf-8",
    )

    print(
        f"Created {output.name}: "
        f"{output.stat().st_size / 1024:.1f} KB"
    )

print("Map export complete!")
