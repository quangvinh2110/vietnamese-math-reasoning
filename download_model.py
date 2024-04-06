import argparse
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="repo id on huggingface hub"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="The absolute path to output folder"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The absolute path to cache folder"
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    snapshot_download(
        repo_id=args.repo_id, 
        ignore_patterns=[".msgpack", ".h5", ".onnx","*.tflite"], 
        local_dir=args.local_dir,
        cache_dir=args.cache_dir,
        token="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"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.local_dir)
    model = AutoModelForCausalLM.from_pretrained(args.local_dir)

    model.save_pretrained(args.local_dir) 
    tokenizer.save_pretrained(args.local_dir)
    
    
if __name__ == "__main__":
    main()