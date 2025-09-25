import sys
import os
import pysam
import numpy
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import logging
import datetime

import re
import collections

from pathlib import Path

import subread

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s: %(message)s')
logger.setLevel(logging.DEBUG)


count_regex = "(?P<label>[\w_.-]+)_S(?P<lane>[\w]+)_(?P<run>[\w]+)_(?P<replicate>[\w]+)(?P<postfix>[a-z.]+)"
simplified_count_regex = "(?P<label>[\w_-]+)_(?P<run>[\w]+)(?P<postfix>[a-z.]+)"


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

#

def read_fasta(path, case=None):
    """
    Read in the input .fasta formatted file.
    Output is a list of tuples with header and sequence
    """
    seqlist = []
    name = ""
    seq = ""
    with open(path) as input_file:
        line = input_file.readline()
        while line:
            if line.startswith(">"):
                #name = line.strip().strip(">")
                #ID, idclass_name = _get_idobj_from_id(name)

                ID = line.strip().strip(">")


                seq = ""
                line = input_file.readline()
                while line and line[0] != ">":
                    seq += line.strip()
                    line = input_file.readline()
                if case == "upper":
                    seq = seq.upper()
                elif case == "lower":
                    seq = seq.lower()
                seqlist.append((ID, seq))
            else:
                line = input_file.readline()
            if not line:
                break
    return seqlist

rev_comp_map = {"A": "T", "a": "t",
                "T": "A", "t": "a",
                "G": "C", "g": "c",
                "C": "G", "c": "g",
                "-": "-", "_": "_",
                "N": "N", "n": "n"}

def reverse(X):
    """
    Reverse complement input DNA sequence
    """
    return X[::-1]

#

def complement(X):
    """
    Reverse complement input DNA sequence
    """
    return "".join([rev_comp_map.get(x, x) for x in X])

#

def reverse_complement(X):
    """
    Reverse complement input DNA sequence
    """
    return "".join([rev_comp_map.get(x, x) for x in X[::-1]])


def make_rev_lib(LIB_PATH, output_name="", upstream=None, downstream=None,
                 safe=True, reverse=True, complement=True):
    """
    Makes a REVISED .fasta reference file from input of sgRNA sequences.
    Takes arguments to optionally Reverse + complement sequences and/or append Downstream and Upstream constants.
    """


    logger.info("Making Revised LIB with options:\n\tUPSTREAM: %s\n\tDOWNSTREAM: %s\n\tReverse: %s\n\tComplement: %s" % (
        str(upstream), str(downstream), str(reverse), str(complement)))

    lib_name, lib_ext = LIB_PATH.split(".", 1)
    if output_name:
        output = open(output_name, "w")
    else:
        output = open("%s.%s.%s" % (lib_name, "rev", lib_ext), "w")
    lib_list = read_fasta(LIB_PATH)
    for name, seq in lib_list:
        if name[0] == ">":
            output.write("%s\n" % name)
        else:
            output.write(">%s\n" % name)

        if reverse and complement:
            final_seq = reverse_complement(seq)
        elif reverse and not complement:
            final_seq = reverse(seq)
        elif complement and not reverse:
            final_seq = complement(seq)
        else:
            final_seq = seq

        if upstream:
            if not safe or (safe and upstream not in final_seq):
                final_seq = upstream + final_seq
        if downstream:
            if not safe or (downstream not in final_seq):
                final_seq = final_seq + downstream
        output.write("%s\n" % final_seq)
    output.close()




def good_enough_cigar_trimmed(*args, **kwargs):
    """

    Args:
        *args:
        **kwargs:

    Returns:

    """
    x = args[0]
    matches = x.get_cigar_stats()[0][0]
    minNumMatches = kwargs.get("minNumMatches", 30)
    return matches > minNumMatches


def good_enough_cigar(*args, **kwargs):
    """

    Args:
        *args:
        **kwargs:

    Returns:

    """
    x = args[0]
    matches = x.get_cigar_stats()[0][0]
    minNumMatches = int(kwargs.get("minNumMatches", 30))
    return matches >= minNumMatches


def good_enough_read_qual(*args, **kwargs):
    """

    Args:
        *args:
        **kwargs:

    Returns:

    """
    x = args[0]
    return int(x.mapping_quality) >= 15


def good_enough_qual(*args, **kwargs):
    """

    Args:
        *args:
        **kwargs:

    Returns:

    """
    x = args[0]
    return int(x.mapping_quality) >= 8

def good_enough_sgrna(*args, **kwargs):
    """

    Args:
        *args:
        **kwargs:

    Returns:

    """
    x = args[0]
    return True



def count_bam_file(output_sorted_bam, **kwargs):


    trim_adaptors = False

    if trim_adaptors:
        default_cigar_func = good_enough_cigar_trimmed
    else:
        default_cigar_func = good_enough_cigar

    matchQCFunc = kwargs.get("matchQCFunc", default_cigar_func)
    qualQCFunc = kwargs.get("qualQCFunc", good_enough_qual)
    sgrnaQCFunc = kwargs.get("sgrnaQCFunc", good_enough_sgrna)
    DEBUG = kwargs.get("DEBUG", False)
    total_reads = kwargs.get("total_reads", -1)
    count_synth = kwargs.get("count_synth", False)
    minNumMatches =  kwargs.get("minNumMatches", 30)

    logger.debug("Requiring at least %s CIGAR matches" % minNumMatches)

    # Read in sorted bam file
    logger.info("Reading file bam: %s" % output_sorted_bam)
    samfile = pysam.AlignmentFile(output_sorted_bam, "rb")

    # Count matching reads
    logger.info("Counting file bam: %s" % output_sorted_bam)
    iter = samfile.fetch()
    sgRNA_counts = {}
    synth_error_counts = {}
    diagnostics = {"good": 0, "bad": 0, "bad_match": 0, "bad_qual": 0,
                   "bad_sgrna": 0,
                   "bad_all": 0, "synth": 0, "processed": 0, }

    if DEBUG:
        output_good = open(output_sorted_bam.replace(".bam", ".qc.good"), "w")
        output_bad = open(output_sorted_bam.replace(".bam", ".qc.bad"), "w")
        output_low_qual = open(
            output_sorted_bam.replace(".bam", ".qc.low_qual"), "w")
        output_low_match = open(
            output_sorted_bam.replace(".bam", ".qc.low_match"), "w")
        output_bad_sgrna = open(
            output_sorted_bam.replace(".bam", ".qc.bad_sgrna"), "w")
        output_really_bad = open(
            output_sorted_bam.replace(".bam", ".qc.really_bad"), "w")
        output_synth = open(output_sorted_bam.replace(".bam", ".qc.synth"), "w")


    for x in iter:
        name = x.reference_name.strip(">")
        if name not in sgRNA_counts: sgRNA_counts[name] = 0

        ## Get QC results
        if matchQCFunc is not None:
            good_match = matchQCFunc(x, **kwargs)
        else:
            good_match = True

        if qualQCFunc is not None:
            good_qual = qualQCFunc(x, **kwargs)
        else:
            good_qual = True

        if sgrnaQCFunc is not None:
            good_sgrna = sgrnaQCFunc(x, **kwargs)
        else:
            good_sgrna = True

        # Only check synth if not perfect match; saves time
        is_synth = 0
        synth_distance = -1
        if count_synth:
            if not good_sgrna and (good_qual and good_match):
                is_synth, synth_distance = is_synth_error(x)
            else:
                is_synth, synth_distance = False, -1

        else:
            is_synth = False
        ## if passes quality control, added +1 to sgRNA reads
        ## also, update diagnostics
        diagnostics["processed"] += 1
        if diagnostics["processed"] % 100000 == 0:
            logger.info("Processed %s reads (%5.2f%%)..." % (diagnostics["processed"],
                100.0*diagnostics["processed"]/total_reads))

        if good_match and good_qual and good_sgrna:
            sgRNA_counts[name] += 1
            diagnostics["good"] += 1
            if DEBUG: output_good.write("%s\n" % x)
        else:
            diagnostics["bad"] += 1
            if DEBUG: output_bad.write("%s\n" % x)
            if not good_match and (good_qual and good_sgrna):
                diagnostics["bad_match"] += 1
                if DEBUG: output_low_match.write("%s\n" % x)
            elif not good_qual and (good_match and good_sgrna):
                diagnostics["bad_qual"] += 1
                if DEBUG: output_low_qual.write("%s\n" % x)
            elif not good_sgrna and (good_match and good_qual):
                diagnostics["bad_sgrna"] += 1
                if DEBUG: output_bad_sgrna.write("%s\n" % x)
            else:
                diagnostics["bad_all"] += 1
                if DEBUG: output_really_bad.write("%s\n" % x)


    if DEBUG:
        output_good.close()
        output_bad.close()
        output_low_qual.close()
        output_low_match.close()
        output_bad_sgrna.close()
        output_really_bad.close()
        output_synth.close()

    return sgRNA_counts, synth_error_counts, diagnostics




def parallel_count_reads(kwargs):


    sample_path = kwargs.get("sample_path")
    index_path = kwargs.get("index_path")

    output_dir = kwargs.get("output_dir", "./")

    label  = kwargs.get("label", "")
    debug = kwargs.get("debug", False)
    force = kwargs.get("force", False)

    maxMismatches=int(kwargs.get("maxMismatches", 1))

    if label and label[0] !="_":
        label = "_"+label
    output_prefix = kwargs.get("output_prefix", os.path.join(output_dir, Path(sample_path).stem.replace(".fastq", "") +label))
   
 
    if sample_path.endswith(".gz"):
        trimmed_sample_path = f"{output_prefix}.trimmed.fastq.gz"
    else:
        trimmed_sample_path = f"{output_prefix}.trimmed.fastq"


    output_bam = f"{output_prefix}.bam"
    output_sorted_bam = f"{output_prefix}.sorted.bam"
    output_counts_file = f"{output_prefix}.counts"
    output_diag_file = f"{output_prefix}.diagnostics"

    

    if False:
        fastq_path = trimmed_sample_path
    else:
        fastq_path = sample_path

    #logger.info("Trimming...")
    #trim_reads.trim_fastq_file(sample_path, output_path=trimmed_sample_path)

    logger.info("Aligning...")
    subread.align(index_path, fastq_path, output_bam, maxMismatches = maxMismatches, type="dna", phredOffset=3, force=force)

    logger.info("Sorting...")
    pysam.sort("-o", output_sorted_bam, "-m", "2G", output_bam)
    logger.info("Indexing...")
    pysam.index(output_sorted_bam)

    # Calculate total reads for statistics
    total_reads = numpy.sum([eval('+'.join(l.rsplit('\t')[2:])) for l in
                         pysam.idxstats(output_sorted_bam).strip().split("\n")])

    logger.info("Counting reads ...")
    ## Count the reads from the bam file 
    BC_counts, synth_error_counts, diagnostics = count_bam_file(output_sorted_bam, total_reads=total_reads, DEBUG=debug)

    ## Save them
    write_counts(BC_counts, output_counts_file,
                 SGRNA_LIBRARY_REV=index_path)

    logger.info("Saving diagnostics...")

    ## Dignostics
    discarded_trim = 0
    diagnostics["total"] = total_reads
    diagnostics["discarded_trim"] = discarded_trim
    diagnostics["discarded_unmapped"] = total_reads - (discarded_trim + diagnostics.get("processed", 0))
    diagnostics["discarded"] = diagnostics["discarded_trim"] + diagnostics["discarded_unmapped"]

    ## Save them
    write_diagnostics(diagnostics, output_diag_file)
    #diagnostic_file_list.append(output_diag_file)
    return (output_counts_file, output_diag_file)


def write_counts(counts_dict, output_path=None, name_list=[],
                 SGRNA_LIBRARY_REV=""):
    """Writes the .counts fole for a given dictionary
    Also writes the .diagnostics file by default.
    """
    # Write to file or STDOUT
    if output_path:
        counts_out = open(output_path, "w")
    else:
        counts_out = sys.stdout

    # Write counts
    if not name_list:
        name_list = [rawname for rawname, seq in read_fasta(SGRNA_LIBRARY_REV)]

    for name in name_list:
        counts_out.write("%s\t%s\n" % (name, counts_dict.get(str(name), 0)))
    counts_out.close()
    return output_path


def write_diagnostics(diagnostics={}, output_path=None):
    """Save out diagnostic statistics
    """
    # Write to file or STDOUT
    if output_path:
        diag_out = open(output_path, "w")
    else:
        diag_out = sys.stdout

    diag_out.write("Total reads:              \t%9s\n" % (
        diagnostics.get("total", "Unknown")))
    diag_out.write("  Discarded reads:        \t%9s\n" % (
        diagnostics.get("discarded", "Unknown")))
    diag_out.write("    Too short:            \t%9s\n" % (
        diagnostics.get("too_short", "Unknown")))
    diag_out.write("    No adaptor:           \t%9s\n" % (
        diagnostics.get("no_adaptor", "Unknown")))
    diag_out.write("    Unmapped:             \t%9s\n" % (
        diagnostics.get("discarded_unmapped", "Unknown")))
    diag_out.write("  Processed reads:        \t%9s\n" % (
        diagnostics.get("processed", "Unknown")))
    diag_out.write("    Counted reads:        \t%9s\n" % (
        diagnostics.get("good", "Unknown")))
    diag_out.write("    Bad reads:            \t%9s\n" % (
        diagnostics.get("bad", "Unknown")))
    diag_out.write("      Low Quality reads:  \t%9s\n" % (
        diagnostics.get("bad_qual", "Unknown")))
    diag_out.write("      Low Match reads:    \t%9s\n" % (
        diagnostics.get("bad_match", "Unknown")))
    diag_out.write("      Bad sgRNA reads:    \t%9s\n" % (
        diagnostics.get("bad_sgrna", "Unknown")))
    diag_out.write("      Entirely bad reads: \t%9s\n" % (
        diagnostics.get("bad_all", "Unknown")))
    diag_out.write("    Synth Error reads:    \t%9s\n" % (
        diagnostics.get("synth", "Unknown")))
    if diag_out is not sys.stdout:
        diag_out.close()
    return output_path



def merge_diagnostics(X, output_path=None, extension="*.diagnostics"):
    """

    Args:
        X:
        output_path:
        extension:
    """
    if isinstance(X, (str, )):
        query_str = os.path.join(X, extension)
        filelist = glob.glob(query_str)
    elif isinstance(X, list):
        filelist = X
    else:
        raise TypeError("Function 'merge_counts' doesn't know how to handle %s" % type(X))

    CD = CountDiagnostics({})
    cd_list = []
    for path in filelist:
        cd_list.append(CountDiagnostics(path))
    newCD = CD.merge(cd_list)
    newCD.save(output_path)



def merge_counts(X, output_path=None, extension="*.counts", sample_name=None):
    """

    Args:
        X:
        output_path:
        extension:
    """
    if isinstance(X, (str, )):
        path = X
        query_str = os.path.join(path, extension)
        filelist = glob.glob(query_str)
    elif isinstance(X, list):
        filelist = X
    else:
        raise TypeError("Function 'merge_counts' doesn't know how to handle %s" % type(X))

    if output_path:
        output = open(output_path, "w")
    else:
        output = sys.stdout

    data_list = []
    cond_to_lane = {}
    for filepath in filelist:

        temp = filepath.rsplit("/", 1)[-1].split("_", 5)

        rawname = filepath.rsplit("/", 1)[-1]


        if sample_name is None:
            try:
                M = re.match(count_regex, rawname)
                if M is not None:
                    cond, lane, run, rep, ext = M.groups()
                else:
                    newM = re.match(simplified_count_regex, rawname)
                    if newM is not None:
                        cond, run, ext = newM.groups()
                        lane, rep = 1,1
                    else:    
                        logger.error("Exception: Sample label (%s) did not match expected format. You will need to manually merge lanes." % (rawname))
                        return 
            except TypeError as e:
                raise Exception("Exception: Sample label (%s) did not match expected format. Quitting.:\n\t%s\n" % (rawname, e))
            except Exception as e:
                raise Exception("Exception: Something went wrong with the counts regex:\n rawname: %s\nregex: %s\tError:%s\n" % (rawname, count_regex, e))
        else:
            cond = sample_name
        row2count = {}
        with open(filepath) as file_obj:
            for line in file_obj:
                tmp = line.strip().split("\t")
                if tmp[0][:2] == "X.":
                    header = tmp
                    continue
                name, count = line.split()
                row2count[name] = int(count)

        if cond not in cond_to_lane:
            cond_to_lane[cond] = row2count
        else:
            cond_to_lane[cond] = merge_count_dicts(row2count, cond_to_lane[cond])

    sorted_conditions = sorted_nicely(cond_to_lane.keys())
    output.write("Id\t%s\n" % ("\t".join([x for x in sorted_conditions])))
    with open(filelist[0]) as file_obj:
        for line in file_obj:
            tmp = line.strip().split("\t")
            if tmp[0][:2] == "X.":
                header = tmp
                continue
            name, count = line.split()

            count_list = ["%s" % cond_to_lane[C].get(name, 0) for C in
                          sorted_conditions]
            count_str = "\t".join(count_list)
            output.write("%s\t%s\n" % (name, count_str))

    if output is not sys.stdout:
        output.close()



    
class CountDiagnostics:
    """
    
    """ 
    def __init__(self, *args, **kwargs):
        X = args[0]
        if isinstance(X, (str,)):
            self.fromFile(*args, **kwargs)
        else:
            self.fromDict(X)
    
    def fromFile(self, *args, **kwargs):
        """

        Args:
            *args:
            **kwargs:
        """
        path = args[0]
        self.diagnostics = {}
        with open(path) as file_obj:
            for line in file_obj:
                if line.strip():
                    key, stat = line.split(":")
                    try:
                        self.diagnostics[key.strip()] = int(stat.strip())
                    except:
                        self.diagnostics[key.strip()] = 0
        
    def fromDict(self, *args, **kwargs):
        """
    
        Args:
            *args:
            **kwargs:
        """
        X = args[0]
        self.diagnostics = X


    def save(self, output_path=""):
        """

        Args:
            output_path:
        """
        # Write to file or STDOUT
        if output_path:
            diag_out = open(output_path, "w")
        else:
            diag_out = sys.stdout

        diag_out.write("Total reads:              \t%9s\n" % (self.total_reads))
        diag_out.write("  Discarded reads:        \t%9s\n" % (self.discarded_reads))
        diag_out.write("    Too short:            \t%9s\n" % (self.too_short))
        diag_out.write("    No adaptor:           \t%9s\n" % (self.no_adaptor))
        diag_out.write("    Unmapped:             \t%9s\n" % (self.unmapped))
        diag_out.write("  Processed reads:        \t%9s\n" % (self.processed_reads))
        diag_out.write("    Counted reads:        \t%9s\n" % (self.counted_reads))
        diag_out.write("    Bad reads:            \t%9s\n" % (self.bad_reads))
        diag_out.write("      Low Quality reads:  \t%9s\n" % (self.low_quality_reads))
        diag_out.write("      Low Match reads:    \t%9s\n" % (self.low_match_reads))
        diag_out.write("      Bad sgRNA reads:    \t%9s\n" % (self.bad_sgrna_reads))
        diag_out.write("      Entirely bad reads: \t%9s\n" % (self.entirely_bad_reads))
        diag_out.write("    Synth Error reads:    \t%9s\n" % (self.synth_error_reads))
        if output_path:
            diag_out.close()

    def merge(self, X):
        """

        Args:
            X:

        Returns:

        """
        if isinstance(X, list):
            return self.merge_object_list(X)
        else:
            return self.merge_object(X)

    def merge_object(self, Y):
        """

        Args:
            Y:

        Returns:

        """
        D = collections.defaultdict(lambda: 0)
        D.update(self.diagnostics)
        for key in Y.diagnostics:
            D[key] += Y.diagnostics[key]
        return CountDiagnostics(D)

    def merge_object_list(self, L):
        """

        Args:
            L:

        Returns:

        """
        D = collections.defaultdict(lambda: 0)
        D.update(self.diagnostics)
        for Y in L:
            for key in Y.diagnostics:
                D[key] += Y.diagnostics[key]
        return CountDiagnostics(D)

    @property
    def total_reads(self):
        """

        Returns:

        """
        return self.diagnostics.get("Total reads", 0)

    @property
    def discarded_reads(self):
        """

        Returns:

        """
        return self.diagnostics.get("Discarded reads", 0)

    @property
    def too_short(self):
        """

        Returns:

        """
        return self.diagnostics.get("Too short", 0)


    @property
    def no_adaptor(self):
        """

        Returns:

        """
        return self.diagnostics.get("No adaptor", 0)

    @property
    def unmapped(self):
        """

        Returns:

        """
        return self.diagnostics.get("Unmapped", 0)

    @property
    def processed_reads(self):
        """

        Returns:

        """
        return self.diagnostics.get("Processed reads", 0)

    @property
    def counted_reads(self):
        """

        Returns:

        """
        return self.diagnostics.get("Counted reads", 0)

    @property
    def bad_reads(self):
        """

        Returns:

        """
        return self.diagnostics.get("Bad reads", 0)


    @property
    def low_quality_reads(self):
        """

        Returns:

        """
        return self.diagnostics.get("Low Quality reads", 0)

    @property
    def low_match_reads(self):
        """

        Returns:

        """
        return self.diagnostics.get("Low Match reads", 0)

    @property
    def bad_sgrna_reads(self):
        """

        Returns:

        """
        return self.diagnostics.get("Bad sgRNA reads", 0)

    @property
    def entirely_bad_reads(self):
        """

        Returns:

        """
        return self.diagnostics.get("Entirely bad reads", 0)

    @property
    def synth_error_reads(self):
        """

        Returns:

        """
        return self.diagnostics.get("Synth Error reads", 0)


    def to_list(self):
        return [
                self.total_reads,
                self.discarded_reads,
                self.too_short,
                self.no_adaptor,
                self.unmapped,
                self.processed_reads,
                self.counted_reads,
                self.bad_reads,
                self.low_quality_reads,
                self.low_match_reads,
                self.bad_sgrna_reads,
                self.entirely_bad_reads,
                self.synth_error_reads
               ]

    def stats_list(self):
        return [
            self.total_reads,
            self.discarded_reads,
            self.too_short,
            self.no_adaptor,
            self.unmapped,
            self.processed_reads,
            self.counted_reads,
            self.bad_reads,
            self.low_quality_reads,
            self.low_match_reads,
            self.bad_sgrna_reads,
            self.entirely_bad_reads,
            self.synth_error_reads]



