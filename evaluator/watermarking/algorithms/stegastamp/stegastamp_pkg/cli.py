import argparse
import os

from .api import embed_file, decode_file


def main():
    parser = argparse.ArgumentParser(description="StegaStamp encode/decode utility")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_enc = sub.add_parser("encode", help="Embed a secret into an image")
    p_enc.add_argument("encoder", type=str, help="Path to encoder .pth")
    p_enc.add_argument("decoder", type=str, help="Path to decoder .pth")
    p_enc.add_argument("input", type=str, help="Input image path")
    p_enc.add_argument("output", type=str, help="Output encoded image path")
    p_enc.add_argument("--secret", type=str, default=None, help="Optional: secret string (<=7 chars) or 100-bit 0/1 string. If omitted, use random 100-bit.")
    p_enc.add_argument("--device", type=str, default=None, help="cuda or cpu")
    p_enc.add_argument("--residual", action="store_true", help="Also save residual image")

    p_dec = sub.add_parser("decode", help="Decode a secret from an image")
    p_dec.add_argument("encoder", type=str, help="Path to encoder .pth (unused but kept for symmetry)")
    p_dec.add_argument("decoder", type=str, help="Path to decoder .pth")
    p_dec.add_argument("input", type=str, help="Input image path (encoded)")
    p_dec.add_argument("--device", type=str, default=None, help="cuda or cpu")
    p_dec.add_argument("--bits", action="store_true", help="Return bits along with string")

    args = parser.parse_args()

    if args.cmd == "encode":
        secret = args.secret
        # allow 100-bit 0/1 string
        if isinstance(secret, str) and len(secret) == 100 and set(secret) <= {"0", "1"}:
            import numpy as np
            secret = np.array([int(c) for c in secret], dtype=np.uint8)
        if secret is None:
            print("[info] No secret provided; using random 100-bit vector.")
        embed_file(args.encoder, args.decoder, args.input, args.output, secret, device=args.device, return_residual=args.residual)
        print(f"Saved encoded image to {args.output}")
        if args.residual:
            base, _ = os.path.splitext(args.output)
            print(f"Saved residual image to {base}_residual.png")
    elif args.cmd == "decode":
        out = decode_file(args.encoder, args.decoder, args.input, device=args.device, return_bits=args.bits)
        if args.bits:
            s, b = out  # type: ignore
            print(f"Decoded: {s}")
            print("Bits:", ''.join(map(str, b.tolist())))
        else:
            print("Decoded:", out)


if __name__ == "__main__":
    main()
