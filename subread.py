"""Thin Python wrappers around the `subread` command-line tools.

Each external tool is invoked as a proper argv LIST (one token per flag/value —
never a single string with embedded spaces), and every call checks the child's
exit status: a nonzero exit or a missing executable raises instead of passing
silently. The command-building is factored into pure `*_command` helpers so it
can be unit-tested without subread installed.

Requires the `subread` package on PATH (e.g. `sudo apt install subread`).
"""

import os
import subprocess
import contextlib
import logging

logger = logging.getLogger(__name__)


def _flag(flag, present):
    """[flag] when present else [] — keeps each flag its own argv token."""
    return [flag] if present else []


def _run(command, label, stdout=None, stderr=None):
    """Run `command` (an argv list), sending stdout/stderr to the given file paths
    (or inheriting the parent's when None). Raises on a missing executable or a
    nonzero exit — external-program failures must never pass silently."""
    logger.debug("%s command: %s", label, " ".join(command))
    # ExitStack closes any opened handle on every exit path (including if the
    # second open() fails after the first succeeded).
    with contextlib.ExitStack() as stack:
        out_fh = stack.enter_context(open(stdout, "w")) if stdout else None
        err_fh = stack.enter_context(open(stderr, "w")) if stderr else None
        try:
            rc = subprocess.call(command, stdout=out_fh, stderr=err_fh)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not run '{command[0]}' for {label}. Is subread installed and on "
                f"PATH? (e.g. `sudo apt install subread`)") from e
    if rc != 0:
        raise RuntimeError(f"{label} failed (exit code {rc}): {' '.join(command)}")
    return rc


# ──────────────────────────────────────────────────────────────────
# subread-buildindex
# ──────────────────────────────────────────────────────────────────

def build_index_command(basename, reference, gappedIndex=False, indexSplit=False,
                        memory=8000, TH_subread=100, colorspace=False):
    """Pure builder for the subread-buildindex argv list."""
    cmd = ["subread-buildindex", "-M", str(memory)]
    cmd += _flag("-c", colorspace)          # colorspace index
    cmd += _flag("-F", not gappedIndex)     # full (non-gapped) index
    cmd += _flag("-B", not indexSplit)      # build a single block (no split)
    cmd += ["-f", str(TH_subread), "-o", basename, reference]
    return cmd


def build_index(basename, reference, gappedIndex=False, indexSplit=False, memory=8000,
                TH_subread=100, colorspace=False, stdout=None, stderr=None, force=False):
    """Build a subread index; skips if the index already exists unless `force`."""
    index_ext_list = ["log", "files", "00.b.tab", "00.b.array", "reads"]
    if all(os.path.exists("%s.%s" % (basename, ext)) for ext in index_ext_list) and not force:
        logger.debug("Index files already exist and not forcing. Skipping.")
        return
    command = build_index_command(basename, reference, gappedIndex, indexSplit,
                                  memory, TH_subread, colorspace)
    logger.debug("subread build-index started")
    _run(command, "subread-buildindex", stdout, stderr)
    logger.debug("subread build-index finished")


# ──────────────────────────────────────────────────────────────────
# subread-align
# ──────────────────────────────────────────────────────────────────

def align_command(index, readfile1, output_file, type="rna", output_format="BAM",
                  nsubreads=10, TH1=3, maxMismatches=3, nthreads=1, indels=5,
                  multiMapping=False, nBestLocations=10, nTrim5=0, nTrim3=0,
                  phredOffset=3):
    """Pure builder for the subread-align argv list."""
    type_str = "0" if type.lower() == "rna" else "1"
    cmd = ["subread-align",
           "-m", str(TH1), "-M", str(maxMismatches), "-B", str(nBestLocations),
           "-n", str(nsubreads), "-T", str(nthreads), "-I", str(indels)]
    cmd += _flag("--multiMapping", multiMapping)
    cmd += _flag("--SAMoutput", output_format.lower() == "sam")
    cmd += ["-i", index, "-r", readfile1, "-o", output_file, "-t", type_str,
            "--trim5", str(nTrim5), "--trim3", str(nTrim3), "-P", str(phredOffset)]
    return cmd


def align(index, readfile1, output_file, readfile2=None, type="rna", input_format="gzFASTQ",
          output_format="BAM", nsubreads=10, TH1=3, TH2=1, maxMismatches=3, nthreads=1,
          indels=5, complexIndels=False, phredOffset=3, multiMapping=False,
          nBestLocations=10, minFragLength=50, maxFragLength=600, PE_orientation="fr",
          nTrim5=0, nTrim3=0, readGroupID=None, readGroup=None, color2base=False,
          DP_GapOpenPenalty=-1, DP_GapExtPenalty=0, DP_MismatchPenalty=0, DP_MatchScore=2,
          detectSV=False, stdout=None, stderr=None, force=False):
    """Align reads with subread-align; skips if the output exists unless `force`."""
    if os.path.exists(output_file) and not force:
        logger.debug("Output file already exists and not forcing. Skipping.")
        return
    command = align_command(index, readfile1, output_file, type=type,
                            output_format=output_format, nsubreads=nsubreads, TH1=TH1,
                            maxMismatches=maxMismatches, nthreads=nthreads, indels=indels,
                            multiMapping=multiMapping, nBestLocations=nBestLocations,
                            nTrim5=nTrim5, nTrim3=nTrim3, phredOffset=phredOffset)
    logger.debug("subread align started")
    _run(command, "subread-align", stdout, stderr)
    logger.debug("subread align finished")


# ──────────────────────────────────────────────────────────────────
# featureCounts
# ──────────────────────────────────────────────────────────────────

def featurecounts_command(files, annotation, output_file, featureType="exon",
                          attrType="gene_id", minMQS=0, countMultiMappingReads=False,
                          nthreads=1):
    """Pure builder for the featureCounts argv list."""
    cmd = ["featureCounts", "-t", featureType, "-g", attrType, "-Q", str(minMQS),
           "-T", str(nthreads)]
    cmd += _flag("-M", countMultiMappingReads)
    cmd += ["-a", annotation, "-o", output_file] + list(files)
    return cmd


def featureCounts(files, annotation, output_file, featureType="exon", attrType="gene_id",
                  minOverlap=1, countMultiMappingReads=False, fraction=False, minMQS=0,
                  primaryOnly=False, nthreads=1, stdout=None, stderr=None, force=False):
    """Count reads per feature with featureCounts."""
    command = featurecounts_command(files, annotation, output_file,
                                    featureType=featureType, attrType=attrType,
                                    minMQS=minMQS,
                                    countMultiMappingReads=countMultiMappingReads,
                                    nthreads=nthreads)
    logger.debug("featureCounts started")
    _run(command, "featureCounts", stdout, stderr)
    logger.debug("featureCounts finished")
