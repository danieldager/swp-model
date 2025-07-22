import os
import signal
import subprocess


class SlurmHandler:
    r"""A signal handler made to react to signal `SIGUSR1` on SLURM clusers, allowing
    to automatically requeue the job."""

    def __init__(self):
        self.stop_signal = False
        self.early_stop = False
        signal.signal(signal.SIGUSR1, self._handler)

    def _handler(self, signum, frame):
        print(f"Signal {signum} received.")
        self.stop_signal = True

    def ask_requeue(self):
        self.early_stop = True

    def land(self):
        if self.early_stop:
            job_id = os.environ.get("SLURM_JOB_ID")
            array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

            if job_id is not None:
                full_id = job_id
                if array_task_id is not None:
                    full_id = f"{full_id}_{array_task_id}"

                print(f"Requeuing job: {full_id}")
                try:
                    subprocess.run(["scontrol", "requeue", full_id], check=True)
                    print("Job successfully requeued.")
                except subprocess.CalledProcessError as e:
                    print(f"Error: Failed to requeue job {full_id}: {e}")
            else:
                print("SLURM_JOB_ID not found")
        else:
            print("Landed without requeue requests")
