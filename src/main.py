import sys
import argparse
from mboa import MBOA
from map_elites import MapElites


def main():
    parser = argparse.ArgumentParser(
        description="Run the Iterative trial and error algorithm"
    )
    parser.add_argument(
        "--generate_map",
        action="store_true",
        help="If set, generate the map for the environment",
    )
    parser.add_argument(
        "--no-generate_map",
        dest="generate_map",
        action="store_false",
        help="If set, skip the map generation step",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="QDAntBulletEnv-v0",
        help="Name of the environment to run the algorithm on",
    )
    parser.add_argument(
        "--max_trials",
        type=int,
        default=20,
        help="Maximum number of trials for the agent to adapt his gait after a damage",
    )
    parser.add_argument(
        "--damaged_leg",
        type=str,
        default="back_left",
        help="Name of the leg to damage in the ant robot",
    )
    parser.add_argument(
        "--training_render",
        action="store_true",
        help="If set render the environment during the generation of the map but n_jobs will be set to 1",
    )
    parser.add_argument(
        "--evaluation_render",
        action="store_true",
        help="Render the environment during the trials of the agent",
    )
    parser.add_argument(
        "--no-evaluation_render",
        dest="evaluation_render",
        action="store_false",
        help="If set, do not render the environment during the trials of the agent",
    )
    parser.add_argument(
        "--map_file",
        type=str,
        default="map_ant.npz",
        help="Path to the map file containing the precomputed data for the environment",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.5,
        help="Exploration parameter for the Gaussian Process",
    )
    parser.add_argument(
        "--alpha_gp",
        type=float,
        default=1e-4,
        help="Alpha parameter for the Gaussian Process",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=12,
        help="Number of parallel jobs for the Gaussian Process",
    )
    parser.add_argument(
        "--n_buckets",
        type=int,
        default=5,
        help="Number of buckets for the Map Elites algorithm",
    )
    parser.add_argument(
        "--theta_dim",
        type=int,
        default=16,
        help="Dimensionality of the theta vector for the Map Elites algorithm. For the ant robot, it is 16 (8 joints * 2 dimensions per joint)",
    )
    parser.add_argument(
        "--n_evals",
        type=int,
        default=5000,
        help="Number of evaluations for the Map Elites algorithm",
    )
    args = parser.parse_args()

    if args.generate_map:
        print("Generating map...")
        me = MapElites(
            env_name=args.env_name,
            n_buckets=args.n_buckets,
            theta_dim=args.theta_dim,
            sigma=0.2,
            n_evals=args.n_evals,
            batch_size=64,
            n_jobs=args.n_jobs,
            render=args.training_render,
        )
        me.run()
        print("Map generated and saved to 'map_ant.npz'.")

        print("Running MBOA with the generated map...")
        mboa = MBOA(
            env_name=args.env_name,
            map_file=args.map_file,
            damaged_leg=args.damaged_leg,
            max_trials=args.max_trials,
            kappa=args.kappa,
            alpha_gp=args.alpha_gp,
            render=args.evaluation_render,
        )
        mboa.run()

    else:
        print("Running MBOA using existing map...")
        mboa = MBOA(
            env_name=args.env_name,
            map_file=args.map_file,
            damaged_leg=args.damaged_leg,
            max_trials=args.max_trials,
            kappa=args.kappa,
            alpha_gp=args.alpha_gp,
            render=args.evaluation_render,
        )
        mboa.run()


if __name__ == "__main__":
    main()
    sys.exit(0)
