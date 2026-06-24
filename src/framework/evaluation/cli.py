from .evaluator import EventEvaluator
from .parser import build_parser

def main():
    """Entry point for the command-line interface to evaluate events against ground truth."""
    args = build_parser().parse_args()

    evaluator = EventEvaluator(
        gt_h5=args.gt,
        pred_h5=args.pred,
        output_dir=args.output_dir,
        width=args.resolution[0],
        height=args.resolution[1],
    )

    # MODE1: Evaluate a specific time window
    if args.timewindow is not None:
        start_us, end_us = args.timewindow
        metrics = evaluator.evaluate_window(start_us, end_us)
        if metrics:
            print("-" * 50)
            print(f"{'METRIC':<8} | {'VALUE':<12}")
            print("-" * 50)
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name.upper():<8} | {metric_value:<12.4f}")
        else:
            print("No valid metrics for this window.")
        return
    
    # MODE2: Evaluate using a windows file
    if args.windows:
        evaluator.evaluate_windows_file(args.windows)
        return
        
    # MODE3: Evaluate frame by frame
    frame_time = args.frame_time if args.frame_time is not None else 50000
    print(f"Running in frame-by-frame mode (Interval: {frame_time} us)")
    evaluator.evaluate_frame_by_frame(
        frame_time, 
        start_us=args.seconds[0]*1e6 if args.seconds else None, 
        end_us=args.seconds[1]*1e6 if args.seconds else None
    )

# uv run python -m framework.evaluation.cli -o timewindow --gt data/thun_00_a/thun_00_a_events_left/events.h5 --pred results/thun_00_a_events.h5 --timewindow 49599800165 49599900165
# uv run python -m framework.evaluation.cli -o window_queries --gt data/thun_00_a/thun_00_a_events_left/events.h5 --pred results/thun_00_a_events.h5 --windows data/thun_00_a/window_queries.txt
# uv run python -m framework.evaluation.cli -o test --gt data/thun_00_a/thun_00_a_events_left/events.h5 --pred results/thun_00_a_events.h5
# uv run python -m framework.evaluation.cli -o clip --gt data/thun_00_a/thun_00_a_events_left/events.h5 --pred results/thun_00_a_events.h5 --seconds 9 10
if __name__ == "__main__":
    main()