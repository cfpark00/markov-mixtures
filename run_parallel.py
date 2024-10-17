import subprocess
from concurrent.futures import ThreadPoolExecutor
import sys
import glob
import argparse
import os

def run(kwargs):
    """Function to execute a script using subprocess."""
    yaml_path=kwargs["yaml_path"]
    exname=kwargs["exname"]
    args=kwargs["args"]
    try:
        # Ensure your script has the appropriate executable permissions
        result = subprocess.run(['python3', exname+".py", yaml_path]+args , check=True, text=True, capture_output=True)
        print(f"{yaml_path} ran successfully:")
    except subprocess.CalledProcessError as e:
        error_message=e.stderr.decode()
        print(f"Error running {yaml_path}: {error_message}:")


def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple scripts in parallel")
    parser.add_argument("folfile", type=str, help="Folder/File containing the yamls")
    parser.add_argument("--n-par", type=int, default=0, help="Number of parallel processes")
    parser.add_argument("--exname", type=str,default="run", help="exname.py will be ran, default is \"run\" ")
    parser.add_argument("--args", nargs="*",default=[], help="Additional arguments to pass to the script")
    args=parser.parse_args()
    return args

if __name__ == "__main__":
    # Parse arguments
    args=parse_args()
    #check if folfile is a file
    if os.path.isfile(args.folfile):
        with open(args.folfile,"r") as f:
            yaml_paths = f.readlines()
        yaml_paths = [x.strip() for x in yaml_paths]
        yaml_paths = [x for x in yaml_paths if (x.endswith(".yaml") and x[0]!="#")] #remove comments and non
        yaml_paths = [x for x in yaml_paths if os.path.isfile(x)]
    else:
        fol=args.folfile
        yaml_paths = glob.glob(f"{fol}/*.yaml")
    exnames=[args.exname]*len(yaml_paths)
    if args.n_par>0:
        n_par=args.n_par
    else:
        n_par=len(yaml_paths)
    
    kwargss=[]
    for yaml_path in yaml_paths:
        kwargs={}
        kwargs["yaml_path"]=yaml_path
        kwargs["exname"]=args.exname
        kwargs["args"]=args.args
        kwargss.append(kwargs)

    print("Running:",yaml_paths)
    # Use ThreadPoolExecutor to run scripts in parallel
    with ThreadPoolExecutor(max_workers=n_par) as executor:
        # Map each script to the executor
        results = executor.map(run, kwargss)
    print("Done")