"""Download the pinned official external evaluation file, with checksum validation."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

REVISION = "3f22b66276b2d7ca5fe573c28c79cc0d077aafc5"
URL = f"https://raw.githubusercontent.com/seacowx/OpenToM/{REVISION}/data/opentom.json"
SHA256 = "11de58a9950e09d13f08df14dd595c101d4b5c183ed1ac5d0647d181c3676bf2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/external/opentom.json")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raw = output.read_bytes()
    else:
        with urlopen(URL, timeout=60) as response:
            raw = response.read()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("OpenToM checksum mismatch; original file is left unchanged")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    print(f"Verified OpenToM at {output}; external evaluation only")


if __name__ == "__main__":
    main()
