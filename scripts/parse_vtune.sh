#!/bin/bash 
# 
# Parses a the output of a VTune version 2018 summary report for 
# uncore memory access counts using:
#
# amplxe-cl -report hw-events -r <result file> -group-by=package -format=csv -column=UNC_M_CAS_COUNT,UNC_E_RPQ_INSERTS,UNC_E_WPQ_INSERTS
#
# The above command outputs 2 lines of comma delimited column data, 
# the first line of a column is the name of the uncore counter field 
# and the second line of the column is the value

usage () {
  echo "usage: $0 [-v | --verbose] file1 file2 ..."
  exit 0
}

# Initial parameter values
verbose=0

# Parse command line parameters
while [ "$#" -ge 1 ] ; do
  case "$1" in
    "-h" | "--help")           usage;;
    "-v" | "--verbose")        verbose=1; shift 1;;
    *)                         break;;
  esac
done

# after parsing flags, make sure a file is specified
if [ $# -eq 0 ]; then
  usage
fi

# MCDRAM
# UNC_IMC_DRAM_DATA_READS
mcd_read_sum=0
# UNC_IMC_DRAM_DATA_WRITES
mcd_write_sum=0

file_list=$@
files_parsed=0
for file in $file_list; do
  if [ -e $file ]; then
    exec < $file
    if [ $verbose -eq 1 ]; then echo "parsing $file"; fi
    (( ++files_parsed ))
  else
    echo "***WARNING***: $file doesn't exist"
    continue
  fi

  # build arrays with event name and counter values
  while read aline; do
    if [ $verbose -eq 1 ]; then echo "Parsing: $aline"; fi

    field0=${aline:0:7}
    case "$field0" in
      "Package" ) 
         IFS=',' read -ra events <<< "$aline"
         for ((i=1; i<${#events[@]}; ++i)); do
           # strip off useless text from front of string
           events[$i]=${events[$i]#Uncore Event Count:}
           if [ $verbose -eq 1 ]; then echo "Event $i is ${events[$i]}"; fi
         done ;;

      "package" ) 
        num=${aline:7:2}
        case "$num" in 
          "_0" )	# package_0
            IFS=',' read -ra counts0 <<< "$aline"
            for ((i=1; i<${#counts0[@]}; ++i)); do
              if [ $verbose -eq 1 ]; then echo "package_0 event $i is ${counts0[$i]}"; fi
            done ;;
          "_1" )	# package_1
            IFS=',' read -ra counts1 <<< "$aline"
            for ((i=1; i<${#counts1[@]}; ++i)); do
              if [ $verbose -eq 1 ]; then echo "package_1 event $i is ${counts1[$i]}"; fi
            done ;;
        esac ;;

      * )
        echo "WARNING: Unkown line quailfier -> $field0" ;;
    esac
  done # reading lines from a file

  # loop over all events
  for ((i=1; i<${#events[@]}; ++i)); do
    # check to see if more than a single socket with counter data
    if [ ${#counts1[@]} -eq 0 ]; then 
      value=${counts0[$i]}
    else
      value=$(( ${counts0[$i]} + ${counts1[$i]} ))
    fi

    if [ $verbose -eq 1 ]; then
      echo "processing ${events[$i]}, count = $value"
    fi

    # MCDRAM uncore read counters
    if [ "${events[$i]:0:23}" == "UNC_IMC_DRAM_DATA_READS" ]; then
      mcdram=true;
      mcd_read_sum=$(( $mcd_read_sum + $value ));
    # MCDRAM uncore write counters
    elif [ "${events[$i]:0:24}" == "UNC_IMC_DRAM_DATA_WRITES" ]; then
      mcdram=true;
      mcd_write_sum=$(( $mcd_write_sum + $value ));
    fi
  done # all events
done # all files

# print MCDRAM out memory access summary
if [ $mcdram ]; then
  total_read=$mcd_read_sum
  total_write=$mcd_write_sum
  echo "--->MCDRAM Report"
  echo "--->Total read transactions = $total_read"
  echo "--->Total write transactions = $total_write"
  echo "--->Total transactions = $(( total_read + total_write ))"
fi