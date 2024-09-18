#!/bin/bash

# Function to extract the wandbid ID from SLURM logs
extract_wandbid_id() {
    array_job_id=$1
    # Check if SLURM_ARRAY_TASK_ID and SLURM_JOB_NAME are set
    if [ -z "$SLURM_ARRAY_TASK_ID" ] || [ -z "$SLURM_JOB_NAME" ]; then
        echo "SLURM_ARRAY_TASK_ID or SLURM_JOB_NAME is not set."
        return 1
    fi

    # Construct the output file path based on SLURM_JOB_NAME, SLURM_JOB_ID, and SLURM_ARRAY_TASK_ID
    local log_dir="/ihchomes/rezashkv/research/projects/bilevel_ZOFO/logs"
    local outp_file_path="${log_dir}/err-${SLURM_JOB_NAME}-${array_job_id}_${SLURM_ARRAY_TASK_ID}.log"

    # Check if the output file exists
    if [ ! -f "$outp_file_path" ]; then
        echo "Log file not found: $outp_file_path"
        return 1
    fi

    # Initialize the found flag
    found=0

    # Loop until the wandbid is found
    while [ $found -eq 0 ]; do
        if grep -q "Creating sweep with ID: " "$outp_file_path"; then
            # Extract the wandbid following "Creating sweep with ID: "
            local wandbid=$(grep "Creating sweep with ID: " "$outp_file_path" | head -n 1 | sed 's/.*Creating sweep with ID: //')
            echo $wandbid
            found=1
        else
            echo "Waiting for sweep ID to appear in the log..."
            sleep 1
        fi
    done
}
