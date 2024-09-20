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

    # Initialize the found flag
    found=0
    retries=0
    max_retries=30  # Set a maximum retry count to prevent an infinite loop

    # Loop until the wandbid is found or retries exceeded
    while [ $found -eq 0 ] && [ $retries -lt $max_retries ]; do
        if [ -f "$outp_file_path" ]; then
            if grep -q "Creating sweep with ID: " "$outp_file_path"; then
                # Extract the wandbid following "Creating sweep with ID: "
                local wandbid=$(grep "Creating sweep with ID: " "$outp_file_path" | head -n 1 | sed 's/.*Creating sweep with ID: //')
                echo $wandbid
                found=1
            else
                echo "Waiting for sweep ID to appear in the log..."
                sleep 1
            fi
        else
            # If the exact file doesn't exist, try to find one with wildcard
            matching_file=$(find "$log_dir" -name "err-${SLURM_JOB_NAME}-*_${SLURM_ARRAY_TASK_ID}.log" | head -n 1)
            if [ -n "$matching_file" ]; then
                outp_file_path="$matching_file"
            fi
        fi
        retries=$((retries + 1))
    done

    if [ $found -eq 0 ]; then
        echo "Failed to find sweep ID after $retries retries."
        return 1
    fi
}
