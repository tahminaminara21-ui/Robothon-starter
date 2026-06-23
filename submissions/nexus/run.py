"""Entry point for Robothon 2026 submission."""
import argparse


def main():
    parser = argparse.ArgumentParser(description='ShadowDex submission runner')
    parser.add_argument('--demo',  action='store_true', help='Record demo video -> results/demo.mp4')
    parser.add_argument('--audit', action='store_true', help='Run 63-check audit')
    parser.add_argument('--eval',  action='store_true', help='Full benchmark + ablation + dynamics + dataset')
    parser.add_argument('--quick', action='store_true', help='Quick 3-seed benchmark')
    args = parser.parse_args()

    if args.demo:
        from record_demo import record_demo
        record_demo()

    elif args.audit:
        from audit import run_audit
        run_audit()

    elif args.eval:
        from evaluate import main as eval_main
        eval_main()

    elif args.quick:
        from evaluate import run_benchmark
        run_benchmark(n_seeds=3, tag='quick')

    else:
        # Default: full suite
        from evaluate import run_benchmark, run_ablation, run_dynamics_analysis
        from env import export_dataset
        run_benchmark(n_seeds=20, tag='benchmark')
        run_benchmark(n_seeds=20, use_domain_rand=True, tag='benchmark_rand')
        run_ablation(n_seeds=10)
        run_dynamics_analysis()
        export_dataset(n=5)
        print("\nAll done. Run 'python run.py --audit' to verify.")


if __name__ == '__main__':
    main()
