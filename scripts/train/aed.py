from models.aed import AED
from utils.training import base_parser, train


def parser():
    p = base_parser()
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--nhead", type=int, default=6)
    p.add_argument("--d_ff", type=int, default=1536)
    p.add_argument("--dec_layers", type=int, default=4)
    p.add_argument("--dec_dropout", type=float, default=0.1)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--ctc_weight", type=float, default=0.3)
    p.add_argument("--ctc_weight_decode", type=float, default=0.3)
    p.add_argument("--length_bonus", type=float, default=0.0)
    p.add_argument("--max_len_ratio", type=float, default=0.5, help="max output chars per encoder frame")
    return p


def main(argv=None):
    train(AED, parser().parse_args(argv))


if __name__ == "__main__":
    main()
