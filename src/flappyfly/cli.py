import argparse


def main():
    p = argparse.ArgumentParser(prog="flappyfly", description="The male fly connectome plays Flappy Bird")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="download the connectome and build the cached brain")
    t = sub.add_parser("train", help="fit the readout by imitating the teacher bot")
    t.add_argument("--frames", type=int, default=5000)
    t.add_argument("--no-control", action="store_true", help="skip the random-reservoir control")
    e = sub.add_parser("play", help="watch a population of fly brains evolve on the game")
    e.add_argument("--birds", type=int, default=None)
    sub.add_parser("showcase", help="watch the best fly trained so far, one bird, restarting on death")
    args = p.parse_args()

    if args.cmd == "build":
        from .data import build
        build()
    elif args.cmd == "train":
        from .train import train
        results = train(n_frames=args.frames, control=not args.no_control)
        results["fly"][0].save()
    elif args.cmd == "play":
        from .gui import play
        play(args.birds)
    elif args.cmd == "showcase":
        from .showcase import showcase
        showcase()
