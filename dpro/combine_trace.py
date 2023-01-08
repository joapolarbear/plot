''' Combine trace files
    * Usage
    python3 combine_trace.py files/to/combine path/to/dump/rst pid/names bias
    or combine files under a directory
    python3 combine_trace.py path/to/directoy path/to/dump/rst
'''
import ujson as json
import os, sys
ALIGN_TIME = True
KEEP_PID = True
KEEP_TID = True

def combine_files(files, names, bias, output_path):
    final_traces = []
    for idx, file in enumerate(files):
        with open(file, 'r') as fp:
            traces = json.load(fp)
        if "traceEvents" in traces:
            traces = traces["traceEvents"]
        ts = None
        for trace in traces:
            if ALIGN_TIME and ts is None:
                ts = trace["ts"]
            if not KEEP_PID:
                trace["pid"] = names[idx]
            else:
                trace["pid"] = names[idx] + "." + trace["pid"]
            if ALIGN_TIME:
                trace["ts"] = trace["ts"] - ts
            trace["ts"] += bias[idx]
            if KEEP_TID:
                continue
            if "tid" in trace:
                if isinstance(trace["tid"], str) and trace["tid"].isdigit():
                    trace["tid"] = ">=5"
                elif isinstance(trace["tid"], int) and trace["tid"] >= 5:
                    trace["tid"] = ">=5"
        final_traces += traces

    with open(output_path, 'w') as fp:
        json.dump(final_traces, fp)

files = sys.argv[1]
output_path = sys.argv[2]
if len(sys.argv) >= 5:
    bias = [float(n)*1000 for n in sys.argv[4].split(",")]

files = files.split(",")

if len(files) == 1 and os.path.isdir(files[0]):
    names = sorted(os.listdir(files[0]))
    files = [os.path.join(files[0], n) for n in names]
    bias = [0 for _ in files]
else:
    names = sys.argv[3].split(",")

combine_files(files, names, bias, output_path)




