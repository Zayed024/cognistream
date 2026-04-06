"""
CogniStream — Export Fine-tuned Model to Ollama

Converts the merged fine-tuned model to GGUF format and creates
an Ollama model for serving.

Prerequisites:
    pip install llama-cpp-python

Usage:
    # After training is complete
    python scripts/finetune/export_ollama.py

    # Custom model name
    python scripts/finetune/export_ollama.py --name cognistream-vlm

This creates an Ollama model you can use with:
    OLLAMA_MODEL=cognistream-vlm
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FINETUNE_DIR = DATA_DIR / "finetune"
MERGED_OUTPUT = FINETUNE_DIR / "cognistream-moondream-merged"
GGUF_PATH = FINETUNE_DIR / "cognistream-moondream.gguf"
MODELFILE_PATH = FINETUNE_DIR / "Modelfile"


def create_modelfile(gguf_path: Path, model_name: str) -> Path:
    """Create an Ollama Modelfile pointing to the GGUF weights."""
    content = f"""FROM {gguf_path}

TEMPLATE \"\"\"{{{{ .Prompt }}}}\"\"\"

PARAMETER temperature 0.2
PARAMETER num_predict 300
PARAMETER stop "<|endoftext|>"

SYSTEM You are a video frame analysis assistant. When given an image, analyze it and respond with structured fields: SCENE, OBJECTS, ACTIVITY, and ANOMALY.
"""
    MODELFILE_PATH.write_text(content)
    logger.info("Modelfile created at: %s", MODELFILE_PATH)
    return MODELFILE_PATH


def convert_to_gguf():
    """Convert the merged HuggingFace model to GGUF format."""
    if not MERGED_OUTPUT.exists():
        logger.error(
            "Merged model not found at %s\n"
            "Run training first with --merge: python scripts/finetune/train.py",
            MERGED_OUTPUT,
        )
        sys.exit(1)

    logger.info("Converting to GGUF format...")
    logger.info("This requires llama.cpp's convert script.")

    # Try using the Python conversion from llama-cpp-python
    try:
        cmd = [
            sys.executable, "-m", "llama_cpp.convert",
            str(MERGED_OUTPUT),
            "--outfile", str(GGUF_PATH),
            "--outtype", "q4_0",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            logger.info("GGUF conversion successful: %s", GGUF_PATH)
            return
        else:
            logger.warning("llama_cpp.convert failed: %s", result.stderr[:300])
    except Exception as exc:
        logger.warning("llama_cpp.convert not available: %s", exc)

    # Fallback: try HuggingFace's convert script
    try:
        convert_script = Path(sys.executable).parent / "Scripts" / "convert-hf-to-gguf.py"
        if not convert_script.exists():
            # Try pip-installed location
            import llama_cpp
            pkg_dir = Path(llama_cpp.__file__).parent
            convert_script = pkg_dir / "scripts" / "convert-hf-to-gguf.py"

        if convert_script.exists():
            cmd = [sys.executable, str(convert_script), str(MERGED_OUTPUT), "--outfile", str(GGUF_PATH)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                logger.info("GGUF conversion successful: %s", GGUF_PATH)
                return

    except Exception:
        pass

    logger.error(
        "GGUF conversion failed. Manual steps:\n"
        "1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp\n"
        "2. Run: python llama.cpp/convert_hf_to_gguf.py %s --outfile %s --outtype q4_0\n"
        "3. Then run this script again with --skip-convert",
        MERGED_OUTPUT,
        GGUF_PATH,
    )
    sys.exit(1)


def register_ollama(model_name: str):
    """Create the Ollama model from the GGUF file."""
    if not GGUF_PATH.exists():
        logger.error("GGUF file not found: %s", GGUF_PATH)
        sys.exit(1)

    modelfile = create_modelfile(GGUF_PATH, model_name)

    logger.info("Creating Ollama model: %s", model_name)
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile)],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode == 0:
        logger.info("Ollama model created successfully: %s", model_name)
        logger.info("")
        logger.info("To use it, set in .env:")
        logger.info("  OLLAMA_MODEL=%s", model_name)
        logger.info("")
        logger.info("Or test directly:")
        logger.info("  ollama run %s 'Describe this image' --image path/to/frame.jpg", model_name)
    else:
        logger.error("Ollama create failed: %s", result.stderr)


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned model to Ollama")
    parser.add_argument("--name", default="cognistream-vlm", help="Ollama model name")
    parser.add_argument("--skip-convert", action="store_true", help="Skip GGUF conversion (use existing)")
    args = parser.parse_args()

    if not args.skip_convert:
        convert_to_gguf()

    register_ollama(args.name)


if __name__ == "__main__":
    main()
