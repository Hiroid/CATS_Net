import datetime

def start_time():
    return datetime.datetime.now()

def end_time():
    return datetime.datetime.now()

def run_time(time_start, time_end):
    print("Total running time is:")
    print((time_end - time_start).seconds)
