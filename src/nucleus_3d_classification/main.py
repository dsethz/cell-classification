import argparse
import pathlib
import subprocess
import sys

def get_call_py_path():
    # Get the absolute path of the current script (e.g., main.py)
    script_path = pathlib.Path(__file__).resolve()

    # Search upwards in the directory hierarchy for the 'src' folder
    for parent in script_path.parents:
        potential_path = parent / 'src' / 'nucleus_3d_classification' / 'call.py'
        if potential_path.exists():
            return potential_path

    # If we reach here, call.py was not found
    print("Error: call.py not found in the expected directory structure.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Flexible ML model training and prediction", add_help=False)

    # Model type argument
    parser.add_argument('--model_type', type=str, choices=['nn', 'logreg', 'rf'])
    # Command argument for training or prediction
    parser.add_argument('--command', type=str, choices=['train', 'predict'])
    # Base directory for output files
    parser.add_argument("--output_base_dir", type=str, required=True, help="Base directory for output files")

    # Capture additional arguments to pass directly to call.py
    args, unknown = parser.parse_known_args()

    # DEBUG: Print the parsed arguments
    print(f'Parsed arguments: {args}, unknown: {unknown}')

    # Get the absolute path to call.py
    call_py_path = get_call_py_path()

    # Construct the submit command
    submit_command = [
        'python',
        str(call_py_path),
        args.model_type if args.model_type else '',
        args.command if args.command else '',
        *unknown
    ]

    # Create a long string for the submit command
    submit_command = ' '.join([str(x) for x in submit_command])

    # Print the constructed submit command
    print(f'Submission:\n', submit_command)

    # Execute the command
    subprocess.run(submit_command, shell=True)

if __name__ == "__main__":
    main()
