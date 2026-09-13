import argparse


def main():
    p = argparse.ArgumentParser(prog="flappyfly", description="The male fly connectome plays Flappy Bird")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="download the connectome and build the cached brain")
    t = sub.add_parser("train", help="fit the readout by imitating the teacher bot")
    t.add_argument("--frames", type=int, default=5000)
    t.add_argument("--no-control", action="store_true", help="skip the random-reservoir control")
    sub.add_parser("play", help="open the live window")
    args = p.parse_args()

    if args.cmd == "build":
        from .data import build
        build()
    elif args.cmd == "train":
        from .brain import load
        from .train import train
        results = train(load(), n_frames=args.frames, control=not args.no_control)
        results["fly"][0].save()
    elif args.cmd == "play":
        from .play import play
        play()
