import sys
import os
import pysam
import numpy

import logging
import datetime

from multiprocessing import Pool

import subread
import counting_tools


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s: %(message)s')
logger.setLevel(logging.DEBUG)


def get_nice_date_string():
    today = datetime.date.today()
    return "%02d_%02d_%d" % (today.month, today.day, today.year)



def cleanargs(rawargs=[], verbose=True):
    """Returns a list and a dictionary with positional and keyword arguments.

    -This function assumes flags must start with a "-" and and cannot be a
        number (but can include them).

    -Flags should either be followed by the value they want to be associated
        with (i.e. -p 5) or will be assigned a value of True in the dictionary.

    -The dictionary will map flags to the name given minus either ONE or TWO "-" signs in
        front. (i.e. "--verbose" and "-verbose" both map to {"verbose":True}).


    Arguments:
        rawargs (list): List of positional/keyword arguments. As obtained from
                         sys.argv.

    Returns:
        list: List of positional arguments (i.e. arguments without flags),
                in order provided.
        dict: Dictionary mapping flag (key is flag minus the beginning "-") and
                their values.

    """

    if verbose:
        print("#Arguments: %s" % " ".join(sys.argv))

    if rawargs == []:
        rawargs = sys.argv[1:]
    args = []
    kwargs = {}
    count = 0
    # Loop through list of arguments
    while count < len(rawargs):
        # If the current argument starts with "-", then it's probably a flag
        if rawargs[count].startswith("-"):
            # Does it start with 1 or 2 "-"?
            # Useful for striping later
            dash_count = 1
            if rawargs[count].startswith("--"):
                dash_count = 2

            # Check if next argument is a number
            try:
                temp = float(rawargs[count + 1])
                nextIsNumber = True
            except:
                nextIsNumber = False

            # Check if there could be more arguments after
            stillNotFinished = count + 1 < len(rawargs)
            if stillNotFinished:
                # Check if the next value looks like an argument or not
                nextIsNotArgument = not rawargs[count + 1].startswith("-")
                # Check if the next argument looks like a list or not
                nextLooksLikeList = len(rawargs[count + 1].split(" ")) > 1
            else:
                nextIsNotArgument = True
                nextLooksLikeList = False

            # If still things in list, and they look like arguments to a flag, add them to dict
            if stillNotFinished and (
                    nextIsNotArgument or nextLooksLikeList or nextIsNumber):
                kwargs[rawargs[count][dash_count:]] = rawargs[count + 1]
                count += 1
            # Else it's a flag but without arguments/values so assign it True
            else:
                kwargs[rawargs[count][dash_count:]] = True
        # Else, it's probably a positional arguement without flags
        else:
            args.append(rawargs[count])
        count += 1
    return args, kwargs



if __name__ == "__main__":
    
    date_str = get_nice_date_string()

    # Do stuff here
    # Get argument
    logger.info("Starting...")
    args,kwargs = cleanargs()

    DEBUG = kwargs.get("debug", False)
    print(args)

    label = kwargs.get("label", get_nice_date_string())
    output_dir = kwargs.get("output_dir", "./BAM_and_Counts")

    force = kwargs.get("force", False)

    maxMismatches= int(kwargs.get("mm", 1))
    workers = int(kwargs.get("workers", 5))

    raw_fasta_path = kwargs.get("library", "RLC0013.rev.fasta")


    # Making output directories
    logger.info(f"Making output directory {output_dir}...") 
    os.makedirs(output_dir, exist_ok=True)
    
    upstream_seq = kwargs.get("upstream", "TGTAGCTTCTTTCGAGTACAAAAAC")
    downstream_seq = kwargs.get("downstream", "TCCCAGATTATATCTATCACTGATAGGGATCGCAAATCCCCAGGTCAGAGGGCTATTTTCCCTGGTCAGA")

    make_rev_lib = kwargs.get("make_rev_lib", False)
    if make_rev_lib:
        logger.info(f"Making reverse complement library...")
        logger.info(f"Upstream constant: {upstream_seq}")
        logger.info(f"Downstream constant: {downstream_seq}")
        
        counting_tools.make_rev_lib(raw_fasta_path, output_name=raw_fasta_path.replace(".fasta", ".rev.fasta"),
                upstream=upstream_seq,
                downstream=downstream_seq,
                safe=True, reverse=True, complement=True)
        fasta_path = raw_fasta_path.replace(".fasta", ".rev.fasta")
        index_path = os.path.basename(fasta_path).replace(".fasta", "")

    else:
        fasta_path = raw_fasta_path
        index_path = os.path.basename(fasta_path).replace(".fasta", "")


    logger.info("Building Index...")
    subread.build_index(fasta_path, fasta_path, force=force)
    
    count_file_list = []
    diagnostic_file_list = []

    logger.info(f"Processing reads with {workers} workers...")

    parallel_args = []
    for path in args:
        parallel_args.append(
            dict(sample_path=path, index_path=fasta_path, 
                    output_dir="./BAM_and_Counts", label=label, debug=DEBUG,
                    maxMismatches=maxMismatches, force=force)
        )

    p = Pool(workers)
    result_count_diag_list = p.map(counting_tools.parallel_count_reads, parallel_args)
    p.close()
    p.join()

   
    count_file_list = [x[0] for x in result_count_diag_list]
    diagnostic_file_list = [x[1] for x in result_count_diag_list]


    counting_tools.merge_counts(count_file_list, output_path="merged_%s_counts.txt" % label)
    counting_tools.merge_diagnostics(diagnostic_file_list, output_path="merged_%s_diagnostics.txt" % label)
     

    logger.info("Done.")



