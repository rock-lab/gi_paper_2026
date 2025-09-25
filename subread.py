
import os
import subprocess
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s: %(message)s')
logger.setLevel(logging.DEBUG)



def build_index(basename, reference, gappedIndex=False, indexSplit=False, memory=8000,
     TH_subread=100, colorspace=False, stdout=None, stderr=None, force=False):
    """

    Args:
        basename:
        reference:
        gappedIndex:
        indexSplit:
        memory:
        TH_subread:
        colorspace:
        stdout:
        stderr:
        force:

    Returns:

    """
    # system:
    #./subread-buildindex [options] -o <basename> {FASTA file1} [FASTA file2] 
    
    # Rsubread:
    #buildindex(basename,reference,gappedIndex=TRUE,indexSplit=TRUE,memory=8000,
    # TH_subread=100,colorspace=FALSE) 


    # Check if index already exists
    index_ext_list = ["log", "files", "00.b.tab", "00.b.array", "reads"]
    if sum([os.path.exists("%s.%s" % (basename, ext))  for ext in index_ext_list]) == len(index_ext_list):
        if not force:
            logger.debug('Index files already exist and not forcing. Skipping.')
            return


    # subprocess output
    if stdout:
        outfile = open(stdout, "w")
    if stderr:
        errfile = open(stderr, "w")

    indexsplit_str = " -B" if not indexSplit else ""
    gappedindex_str = " -F" if not gappedIndex else ""
    colorspace_str = " -c" if colorspace else ""
    command_list = ["subread-buildindex", "-M", 
            "{mem}{color}{gapped}{split}",
            "-f", "{th}", "-o", "{base}", "{ref}"]

    cmd_kwargs = dict(
        # Optional arguments
        mem=memory, th=TH_subread,
        # Optional Flags
        color=colorspace_str, gapped=gappedindex_str, split=indexsplit_str,
        # Required arguments
        base=basename, ref=reference)

    
    command = [c.format(**cmd_kwargs) for c in command_list]
    command_str = " ".join(command)    

    logger.debug("subread-align command: %s" % (command_str))
    logger.debug("subread build-index started")
    subprocess.call(command, stdout=stdout, stderr=stderr)
    logger.debug("subread build-index finished")

    if stdout:
        outfile.close()
    if stderr:
        errfile.close()

        


def align(index, readfile1, output_file, readfile2=None, type="rna", input_format="gzFASTQ", output_format="BAM",
        nsubreads=10,
        TH1=3, TH2=1, maxMismatches=3, nthreads=1, indels=5, complexIndels=False, phredOffset=3,
        multiMapping=False, nBestLocations=10, minFragLength=50, maxFragLength=600, PE_orientation="fr",
        nTrim5=0, nTrim3=0, readGroupID=None,readGroup=None, color2base=False,
        DP_GapOpenPenalty=-1, DP_GapExtPenalty=0, DP_MismatchPenalty=0, DP_MatchScore=2,
        detectSV=False, stdout=None, stderr=None, force=False):
    """

    Args:
        index:
        readfile1:
        output_file:
        readfile2:
        type:
        input_format:
        output_format:
        nsubreads:
        TH1:
        TH2:
        maxMismatches:
        nthreads:
        indels:
        complexIndels:
        phredOffset:
        multiMapping:
        nBestLocations:
        minFragLength:
        maxFragLength:
        PE_orientation:
        nTrim5:
        nTrim3:
        readGroupID:
        readGroup:
        color2base:
        DP_GapOpenPenalty:
        DP_GapExtPenalty:
        DP_MismatchPenalty:
        DP_MatchScore:
        detectSV:
        stdout:
        stderr:
        force:

    Returns:

    """
    # System
    #  ./subread-align [options] -i <index_name> -r <input> -o <output> -t <type>

    # rsubread:
    # align(index,readfile1,readfile2=NULL,type="rna",input_format="gzFASTQ",output_format="BAM",
    #   output_file=paste(readfile1,"subread",output_format,sep="."),nsubreads=10,
    #   TH1=3,TH2=1,maxMismatches=3,nthreads=1,indels=5,complexIndels=FALSE,phredOffset=33,
    #   unique=TRUE,nBestLocations=1,minFragLength=50,maxFragLength=600,PE_orientation="fr",
    #   nTrim5=0,nTrim3=0,readGroupID=NULL,readGroup=NULL,color2base=FALSE,
    #   DP_GapOpenPenalty=-1,DP_GapExtPenalty=0,DP_MismatchPenalty=0,DP_MatchScore=2,
    #   detectSV=FALSE)


    if os.path.exists(output_file):
        if not force:
            logger.debug('Output file already exist and not forcing. Skipping.')
            return

    if stdout:
        outfile = open(stdout, "w")
    if stderr:
        errfile = open(stderr, "w")


    type_str = "0" if type.lower()=="rna" else "1"
    multi_str = " --multiMapping" if multiMapping else ""
    format_str = " --SAMoutput" if output_format.lower() == "sam" else ""
    command_list = ["subread-align", "-m", "{th1}",
                "-M", "{mismatches}", "-B", "{nBest}",
                "-n", "{subreads}", "-T", "{threads}",
                "-I", "{indels}{multi}{format}", "-i", "{index}",
                "-r", "{input}", "-o", "{out}", "-t", "{type}",
                "--trim5", "{nTrim5}", "--trim3", "{nTrim3}",
                "-P", "{phredOffset}"]
    cmd_kwargs = dict(
        # Optional arguments
        th1=TH1, mismatches=maxMismatches, subreads=nsubreads, threads=nthreads, indels=indels,
        # Optional flags
        multi=multi_str, format=format_str, nBest=nBestLocations,
        nTrim5=nTrim5, nTrim3=nTrim3,    
        phredOffset=phredOffset,
        # Needed
        index=index, input=readfile1, out=output_file, type=type_str)

    command = [c.format(**cmd_kwargs) for c in command_list]
    command_str = " ".join(command)    
    logger.debug("subread-align command: %s" % command_str)
    logger.debug("subread align started")
    try:
        subprocess.call(command, stdout=stdout, stderr=stderr)
        logger.debug("subread align finished")
    except OSError:
        logger.error("Error running index. Did you make sure subread is installed?")
    

    if stdout:
        outfile.close()
    if stderr:
        errfile.close()



def featureCounts(files, annotation, output_file, featureType = "exon", attrType = "gene_id",
                    minOverlap = 1, countMultiMappingReads = False,
                    fraction = False, minMQS = 0, primaryOnly = False,
                    nthreads=1,
                    stdout=None, stderr=None, force=False):
    """
    (files, annot.inbuilt = "mm10", annot.ext = NULL, isGTFAnnotationFile = FALSE,
    GTF.featureType = "exon", GTF.attrType = "gene_id", chrAliases = NULL,
    useMetaFeatures = TRUE, allowMultiOverlap = FALSE, minOverlap = 1,
    largestOverlap = FALSE, readExtension5 = 0, readExtension3 = 0,
    read2pos = NULL, countMultiMappingReads = FALSE, fraction = FALSE,
    minMQS = 0, splitOnly = FALSE, nonSplitOnly = FALSE, primaryOnly = FALSE,
    ignoreDup = FALSE, strandSpecific = 0, juncCounts = FALSE,
    genome = NULL, isPairedEnd = FALSE, requireBothEndsMapped = FALSE,
    checkFragLength = FALSE, minFragLength = 50, maxFragLength = 600,
    countChimericFragments = TRUE, autosort = TRUE, nthreads = 1,
    maxMOp = 10, reportReads = FALSE)"""

    if stdout:
        outfile = open(stdout, "w")
    if stderr:
        errfile = open(stderr, "w")


    files_str = " ".join(files)
    multi_str = " -M" if countMultiMappingReads else ""
    command_str = "featureCounts -t {feature} -g {attribute} -Q {quality} -T {threads}{multi} -a {annotation} -o {out} {files}".format(
        # Optional arguments
        feature=featureType, attribute=attrType, quality=minMQS,
        threads=nthreads,

        # Optional flags
        multi=multi_str,

        # Needed
        annotation=annotation, out=output_file, files=files_str)

    command = command_str.split(" ")
    logger.debug("featureCounts command: %s" % command_str)
    logger.debug("featureCounts started")
    subprocess.call(command, stdout=stdout, stderr=stderr)
    logger.debug("featureCounts finished")


    if stdout:
        outfile.close()
    if stderr:
        errfile.close()



