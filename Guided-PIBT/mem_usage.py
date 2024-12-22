import psutil
import logging
import time
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(filename='memory_usage.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def log_memory_usage(pid):
    try:
        process = psutil.Process(pid)
        mem_info = process.memory_info()
        logging.info(f"Memory usage for PID {pid}: RSS = {mem_info.rss / 1024 ** 2:.2f} MB, VMS = {mem_info.vms / 1024 ** 2:.2f} MB")
    except psutil.NoSuchProcess:
        logging.error(f"Process with PID {pid} not found.")
        exit(0)
    except psutil.AccessDenied:
        logging.error(f"Access denied to process with PID {pid}.")

if __name__ == "__main__":
    pid = 2558092  # Replace with the actual PID you want to monitor
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=5)  # Set the end time to 5 minutes from the start

    while datetime.now() < end_time:
        log_memory_usage(pid)
        time.sleep(10)  # Sleep for 10 seconds

    # Optionally, log memory usage one last time at the end
    log_memory_usage(pid)
