# This is the entry point, which would call call.py.

import argparse
import pathlib
import subprocess

# We will construct the argument parser here, and send a command calling call.py using slightly modified args.

def main():

    parser = argparse.ArgumentParser(description="Flexible ML model training and prediction", add_help=False)

    # If input to argparse is --nn/--logreg/--rf it should be translated to 'nn', 'logreg', 'rf' for the call.py script
    parser.add_argument('--model_type', type=str, choices=['nn', 'logreg', 'rf'])
    # If input to argparse is --train/--predict it should be translated to 'train', 'predict' for the call.py script
    parser.add_argument('--command', type=str, choices=['train', 'predict'])
    
    # Additional model runner args:
    parser.add_argument("--output_base_dir", type=str, required=True, help="Base directory for output files")

    # Additional arguments should be passed as is, we will simply pass them as is to call.py.
    # We will not 'catch' them here, as we want to pass them as is to call.py.

    args, unknown = parser.parse_known_args()

    # DEBUG: Print the parsed arguments
    print(f'Parsed arguments: {args}, unknown: {unknown}')

    # Find call.py
        # Determine the base directory of your project
    try:
        base_dir = pathlib.Path(__file__).resolve().parent.parent.parent

        # Build the relative path to call.py
        call_py_path = base_dir / 'src' / 'nucleus_3d_classification' / 'call.py'
    
    except Exception as e:
        print(f"Could not find call.py: {e}")
        return



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

    # print the output command
    print(f'Submission:\n', submit_command)

    # Submit the job
    subprocess.run(submit_command, shell=True)

if __name__ == "__main__":
    main()

