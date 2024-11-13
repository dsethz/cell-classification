# NOT USED

import argparse
import os
import json
import itertools
import subprocess

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def assert_dir_exists(dir_path):
    # If list, assert all directories exist
    if isinstance(dir_path, list):
        for path in dir_path:
            assert_dir_exists(path)
    else:
        print('Checking directory:', dir_path)
        if not os.path.exists(dir_path):
            print('Creating directory:', dir_path)
            # TODO: Uncomment the line below to actually create the directory
            os.makedirs(dir_path)

def assert_numeric_int(value):
    if isinstance(value, list):
        for v in value:
            assert_numeric_int(v)
        return
    else:
        try:
            value = int(value)
        except ValueError:
            raise ValueError(f"Expected int, got {value} of type {type(value)}")
        
def assert_file_exists(file_path):
    # If list, assert all files exist
    if isinstance(file_path, list):
        for path in file_path:
            assert_file_exists(path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def create_combinations(param_grid):
    keys, values = zip(*param_grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

def submit_job(params, slurm_params, varied_params):
    # Generate a job name based only on parameters with more than one possibility.
    job_name = "_".join([f"{k}_{v}" for k, v in params.items() if k in varied_params])
    # Remove model_class_, replace learning_rate with lr, and such changes for better readability
    job_name = job_name.replace('model_class_', '').replace('learning_rate_', 'lr').replace('batch_size_', 'bs').replace('num_workers_', 'nw').replace('epochs_', 'e').replace('-', '').replace('.', '_')

    # print(f"Submitting job: \n{job_name}\n Params: \n{params}\n Slurm params: \n{slurm_params}")

    slurm_command = [
        'sbatch',
        '--job-name', job_name,
        '--output', f"outlogs/{job_name}.out",
        '--error', f"errlogs/{job_name}.err",
        '--tmp', str(slurm_params['tmp']),
        '--ntasks-per-node', str(slurm_params['ntasks_per_node']),
        '--time', slurm_params['time'],
        '--mem-per-cpu', slurm_params['mem_per_cpu'],
        '--cpus-per-task', str(slurm_params['cpus_per_task']),
        '--gpus', slurm_params['gpus'],
        '--wrap=', f"python train.py nn train"
    ]

    # Add additional parameters to the wrap command
    for k, v in params.items():
        # If key is either default_root_dir or dirpath, add job_name string
        if k in ['default_root_dir', 'dirpath']:
            slurm_command[-1] += f" --{k} {v}/{job_name}"
            assert_dir_exists(f"{v}/{job_name}")
            continue
        if isinstance(v, bool) and v:
            slurm_command[-1] += f" --{k}"
            continue
        elif not isinstance(v, bool):
            slurm_command[-1] += f" --{k} {v}"

    # # Create long string for slurm command, converting int to str
    # slurm_command = [str(x) for x in slurm_command]
    # # Add quotes around the wrap command
    # slurm_command[-1] = f'"{slurm_command[-1]}"'
    # slurm_command = ' '.join(slurm_command)

    # Join the wrap command with double quotes directly
    wrap_command = f'"{slurm_command[-1]}"'

    # Replace the last item in the slurm_command list with the properly quoted wrap command
    slurm_command[-1] = wrap_command

    # Convert the slurm_command list to a single string, but leave no gap for the wrap command
    slurm_command = ' '.join([str(x) for x in slurm_command[:-1]]) + slurm_command[-1]

    # DEBUG: Print the submission command
    print(f'Submission:\n', slurm_command)

    # TODO: Uncomment the line below to actually submit the job
    subprocess.run(slurm_command, shell=True)

def main():
    parser = argparse.ArgumentParser(description="Submit sbatch jobs from a JSON file")
    parser.add_argument('json_file', type=str, help='Path to the JSON file with parameters')
    args = parser.parse_args()

    config = load_json(args.json_file)
    param_grid = config['param_grid']
    slurm_params = config['slurm_params']

    # TODO: Assert setup_file exists
    assert_file_exists(param_grid['setup_file'])

    # Assert numeric values are integers, where necessary
    for key in ['tmp', 'ntasks_per_node', 'cpus_per_task', 'mem_per_cpu']:
        assert_numeric_int(slurm_params[key])
    for key in ['devices', 'save_top_k', 'max_epochs', 'log_every_n_steps', 'batch_size']:
        assert_numeric_int(param_grid[key])

    # Identify parameters with more than one possibility
    varied_params = [key for key, values in param_grid.items() if len(values) > 1]

    param_combinations = create_combinations(param_grid)

    for params in param_combinations:
        submit_job(params, slurm_params, varied_params)

if __name__ == "__main__":
    main()
