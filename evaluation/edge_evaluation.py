import argparse
import os

seism_root = "seism/"


def nms_process(save_dir):
    """
    Do NMS process on edge prediction images
    :param str save_dir: Path of edge prediction directory
    """
    cwd = os.getcwd()
    os.chdir(seism_root)
    new_path = os.path.join("../", save_dir, "edge")
    os.system("matlab -nodisplay -nosplash -nodesktop -r \"nms_process('%s');exit\"" % new_path)
    os.chdir(cwd)


def eval_edge_predictions(dataset, exp_name, save_dir):
    """
    Evaluate edge predictions using seism in MatLab
    :param str dataset: Dataset name
    :param str exp_name: Name of experiment
    :param str save_dir: Path of edge prediction directory
    """
    print("Evaluate edge predictions using seism in Matlab.")

    # Generate MATLAB script
    script_base = os.path.join(seism_root, "pr_curves_base.m")
    with open(script_base) as f:
        seism_file = f.readlines()
    seism_file = [line.rstrip() for line in seism_file]
    output_file = seism_file[0:1]
    output_file += ["database = '%s';" % dataset]
    output_file += seism_file[1:12]

    # Add method
    output_file += ["methods(end+1).name = '%s';" % (exp_name)]
    output_file += ["methods(end).dir = '%s';" % os.path.join("../", save_dir, "edge", "nms")]
    output_file.extend(seism_file[13:49])

    # Add path to save output
    output_file += ["\t\t\tfilename = '%s';" % (os.path.join("../", save_dir, "edge_test.txt"))]
    output_file += seism_file[50:]

    # Save script file
    output_file_path = os.path.join(seism_root, exp_name + ".m")
    with open(output_file_path, "w") as f:
        for line in output_file:
            f.write(line + "\n")

    # Go to seism directory and perform evaluation
    print("Go to seism directory and run evaluation. Please wait...")
    cwd = os.getcwd()
    os.chdir(seism_root)
    os.system('matlab -nodisplay -nosplash -nodesktop -r "%s;exit"' % (exp_name))
    os.chdir(cwd)


def display_edge_eval_result(exp_name, save_dir):
    """
    Display edge evaluation result and clean up files
    :param str exp_name: Name of experiment
    :param str save_dir: Path of edge prediction directory and evaluation result
    """
    # Collect results from txt file
    with open(os.path.join(save_dir, "edge_test.txt"), "r") as f:
        seism_result = [line.strip() for line in f.readlines()]

    eval_dict = {}
    for line in seism_result:
        metric, score = line.split(":")
        eval_dict[metric] = float(score)

    # Print result
    print("Edge Detection odsF: %.4f" % (100 * eval_dict["odsF"]))

    # Cleanup - Important. Else Matlab will reuse the files.
    print("Cleanup result files in seism.")
    result_path = os.path.join(seism_root, "results/%s/" % exp_name)
    os.system("rm -rf %s" % result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+", required=True)
    parser.add_argument("--results_dir", type=str, default="../results", help="directory of results")
    parser.add_argument("--dataset", type=str, default="PASCALContext", help="PASCALContext or NYUD")
    parser.add_argument("--nms", action="store_true", help="Whether to do NMS.")
    parser.add_argument("--done", action="store_true", help="Whether evaluation has been done.")
    args = parser.parse_args()

    # get save directory
    results_dir = args.results_dir
    for exp_name in args.exp:
        save_dir = os.path.join(results_dir, exp_name, "predictions")

        if not args.done:
            # Step1: NMS process and Evaluate edge predictions using seism in Matlab
            if args.nms:
                nms_process(save_dir)
            eval_edge_predictions(args.dataset, exp_name, save_dir)
        else:
            # Step2: If evaluation is done, display result
            display_edge_eval_result(exp_name, save_dir)
