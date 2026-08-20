import subprocess
import os
import sys

def main():
    # Define the seed list to test
    # seeds = [1234, 2345, 1, 917]
    seeds = [1234, 1]

    # Define the order list to test and adjust as needed
    orders = [
        # adjust when needed
        # -------- GPU 0-3 ---------
        "Kvasir_camo_ISTD_ISIC_cod",
        "cod_ISIC_ISTD_camo_Kvasir",
        "Kvasir_ISIC_camo_cod_ISTD",
        # -------- GPU 4-7 ---------
        "camo_cod_Kvasir_ISIC_ISTD",
        "ISTD_Kvasir_ISIC_camo_cod",
        "ISTD_cod_camo_ISIC_Kvasir"
    ]

    module = "AugModule"
    meth = "sequ"

    # Iterate over all seed and order combinations
    for order in orders:
        for seed in seeds:
            print(f"Running with order={order} and seed={seed}")

            # Build the command
            cmd = [
                # adjust when needed
                "CUDA_VISIBLE_DEVICES=0,1,2,3",
                # "CUDA_VISIBLE_DEVICES=4,5,6,7",
                "torchrun",
                "--nnodes", "1",
                "--nproc_per_node", "4",
                "--master_port=2422",
                # "--master_port=2246",
                "train_regcl.py", # traning setups
                "--module", module,
                "--batch_size", "2",
                "--cuda", "-1",
                "--meth", meth,
                "--seed", str(seed),
                "--order", order
            ]

            # Check whether this experiment directory already exists
            base_dir = f"log/Comparison/{module}/regcl/{meth}/{order}"
            seed_prefix = f"{seed}_SEED_"

            # Check whether base_dir exists and contains a subdirectory with seed_prefix
            experiment_exists = False
            if os.path.exists(base_dir):
                for subdir in os.listdir(base_dir):
                    subdir_path = os.path.join(base_dir, subdir)
                    if os.path.isdir(subdir_path) and subdir.startswith(seed_prefix):
                        print(f"Experiment with order={order} and seed={seed} already exists:\n {subdir_path}")
                        print("Skipping this experiment...")
                        experiment_exists = True

            if not experiment_exists:
                # Run the command after joining the command list into a string
                full_cmd = " ".join(cmd)
                print(f"Executing: {full_cmd}")

                # Run the command and wait for completion
                try:
                    subprocess.run(full_cmd, shell=True, check=True)
                    print(f"------------------ Complete ------------------")
                except subprocess.CalledProcessError as e:
                    print(f"Error running experiment: {e}")
                    print("Stopping script due to error.")
                    sys.exit(1)
            else:
                # Skip waiting and continue to the next experiment
                print("Skipping wait time since experiment was skipped...")
                continue

            # Wait briefly to ensure the previous experiment has fully started
            print("Waiting for 10 seconds before next experiment...")
            subprocess.run(["sleep", "10"])

if __name__ == "__main__":
    main()
