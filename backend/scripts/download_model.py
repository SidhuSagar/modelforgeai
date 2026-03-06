# scripts/download_model.py
"""
Simple CLI to download (copy) and optionally extract a packaged model zip.

Usage:
    python scripts/download_model.py --model MODEL_NAME --dest ./downloads --no-extract
    python scripts/download_model.py --package outputs/packages/my_model_20251105_000000.zip --dest ./downloads

If model_name is given, it attempts to find the latest package under outputs/packages/
If none exists but model_name is registered, it will attempt to call package_model_for_download()
to create the package and then download it.
"""
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.model_manager import download_model, find_package_for_model, package_model_for_download, load_model_metadata

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", "-m", type=str, help="Model name (registry name)")
    p.add_argument("--package", "-p", type=str, help="Direct package path (zip)")
    p.add_argument("--dest", "-d", type=str, default="./downloads", help="Destination folder")
    p.add_argument("--no-extract", action="store_true", help="Do not auto-extract the zip")
    args = p.parse_args()

    if not args.model and not args.package:
        print("Provide --model MODEL_NAME or --package /path/to/package.zip")
        return

    package_path = args.package
    if not package_path and args.model:
        # look for existing package
        package_path = find_package_for_model(args.model)
        if not package_path:
            # try to package from registry metadata
            meta = load_model_metadata(args.model)
            if meta and meta.get("model_path"):
                print("No package found, creating one from registry metadata...")
                package_path = package_model_for_download(meta, model_name=args.model)
            else:
                print("No package found and no registry entry available for this model.")
                return

    print(f"Using package: {package_path}")
    try:
        extracted = download_model(model_name=None, package_path=package_path, dest_dir=args.dest, extract=not args.no_extract)
        print("✅ Package copied/extracted to:", extracted)
    except Exception as e:
        print("❌ Failed:", e)

if __name__ == "__main__":
    main()
