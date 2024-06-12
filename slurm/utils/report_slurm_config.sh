#!/bin/bash

echo "========================================================================"
echo "-------- Reporting SLURM configuration ---------------------------------"
date
echo ""
echo "SLURM_CLUSTER_NAME      = $SLURM_CLUSTER_NAME"        # Name of the cluster on which the job is executing.
echo "SLURM_JOB_QOS           = $SLURM_JOB_QOS"             # Quality Of Service (QOS) of the job allocation.
echo "SLURM_JOB_ID            = $SLURM_JOB_ID"              # The ID of the job allocation.
echo "SLURM_RESTART_COUNT     = $SLURM_RESTART_COUNT"       # The number of times the job has been restarted.
if [ "$SLURM_ARRAY_TASK_COUNT" != "" ]; then
    echo ""
    echo "SLURM_ARRAY_JOB_ID      = $SLURM_ARRAY_JOB_ID"        # Job array's master job ID number.
    echo "SLURM_ARRAY_TASK_COUNT  = $SLURM_ARRAY_TASK_COUNT"    # Total number of tasks in a job array.
    echo "SLURM_ARRAY_TASK_ID     = $SLURM_ARRAY_TASK_ID"       # Job array ID (index) number.
    echo "SLURM_ARRAY_TASK_MAX    = $SLURM_ARRAY_TASK_MAX"      # Job array's maximum ID (index) number.
    echo "SLURM_ARRAY_TASK_STEP   = $SLURM_ARRAY_TASK_STEP"     # Job array's index step size.
fi;
echo ""
echo "SLURM_JOB_NUM_NODES     = $SLURM_JOB_NUM_NODES"       # Total number of nodes in the job's resource allocation.
echo "SLURM_JOB_NODELIST      = $SLURM_JOB_NODELIST"        # List of nodes allocated to the job.
echo "SLURM_TASKS_PER_NODE    = $SLURM_TASKS_PER_NODE"      # Number of tasks to be initiated on each node.
echo "SLURM_NTASKS            = $SLURM_NTASKS"              # Number of tasks to spawn.
echo "SLURM_PROCID            = $SLURM_PROCID"              # The MPI rank (or relative process ID) of the current process
echo ""
echo "SLURM_GPUS_ON_NODE      = $SLURM_GPUS_ON_NODE"        # Number of allocated GPUs per node.
echo "SLURM_CPUS_ON_NODE      = $SLURM_CPUS_ON_NODE"        # Number of allocated CPUs per node.
echo "SLURM_CPUS_PER_GPU      = $SLURM_CPUS_PER_GPU"        # Number of CPUs requested per GPU. Only set if the --cpus-per-gpu option is specified.
echo "SLURM_MEM_PER_GPU       = $SLURM_MEM_PER_GPU"         # Memory per allocated GPU. Only set if the --mem-per-gpu option is specified.
echo ""
if [[ "$SLURM_TMPDIR" != "" ]];
then
    echo "------------------------------------"
    echo ""
    echo "SLURM_TMPDIR = $SLURM_TMPDIR"
    echo ""
    echo "Contents of $SLURM_TMPDIR"
    ls -lh "$SLURM_TMPDIR"
    echo ""
fi;
echo "========================================================================"
echo ""
