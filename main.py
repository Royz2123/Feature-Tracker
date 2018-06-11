import tracker
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Tracking points')
    parser.add_argument('--mode', default=tracker.Tracker.SCANNING_MODE,
                       type=int, help='directory of all the training faces')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    point_tracker = tracker.Tracker(mode=args.mode)

    # scanning phase
    #point_tracker.scan_points()

    #point_tracker = tracker.Tracker()

    # mapping phase
    point_tracker.run()


if __name__ == "__main__":
    main()
