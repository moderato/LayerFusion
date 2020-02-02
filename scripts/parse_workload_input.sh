if [ "$#" != 1 ] ;
then
  echo "Exactly one argument is required"
fi

IFS=',' read -ra PARAM <<< "$1"
workload_name=${PARAM[0]}

echo ${PARAM[*]}

# Layer 1
mb_1=${PARAM[1]}
ih_1=${PARAM[2]}
iw_1=${PARAM[3]}
ic_1=${PARAM[4]}

kh_1=${PARAM[5]}
kw_1=${PARAM[5]}
sh_1=${PARAM[7]}
sw_1=${PARAM[7]}

is_depthwise_1=${PARAM[8]}
if [ $is_depthwise_1 ]
then
    g_1=${PARAM[4]}
    oc_1=$(( $ic_1 * ${PARAM[6]} ))
else # conv
    g_1=1
    oc_1=$(( ${PARAM[6]} ))
fi

if [ $sh_1 == 1 ]
then
    oh_1=$ih_1
    ow_1=$iw_1
else # Assume sh_1 = sw_1 = 2 for all non-1 cases
    oh_1=$(( $ih_1 / 2 ))
    ow_1=$(( $iw_1 / 2 ))
fi

if [ $kh_1 == 1 ]
then
    ph_1=0
    pw_1=0
else # Assume kh_1 = kw_1 = 3 for all non-1 cases
    ph_1=1
    pw_1=1
fi

# Layer 2
mb_2=$mb_1
ih_2=$oh_1
iw_2=$ow_1
ic_2=$oc_1

kh_2=${PARAM[10]}
kw_2=${PARAM[10]}
sh_2=${PARAM[12]}
sw_2=${PARAM[12]}
g_2=1
oc_2=$(( ${PARAM[11]} ))

if [ $sh_2 == 1 ]
then
    oh_2=$ih_2
    ow_2=$iw_2
else # Assume sh_2 = sw_2 = 2 for all non-1 cases
    oh_2=$(( $ih_2 / 2 ))
    ow_2=$(( $iw_2 / 2 ))
fi

if [ $kh_2 == 1 ]
then
    ph_2=0
    pw_2=0
else # Assume kh_2 = kw_2 = 3 for all non-1 cases
    ph_2=1
    pw_2=1
fi

layer_1_desc=$"g${g_1}mb${mb_1}ic${ic_1}ih${ih_1}iw${iw_1}oc${oc_1}oh${oh_1}ow${ow_1}kh${kh_1}kw${kw_1}sh${sh_1}sw${sw_1}ph${ph_1}pw${pw_1}n"
layer_2_desc=$"g${g_2}mb${mb_2}ic${ic_2}ih${ih_2}iw${iw_2}oc${oc_2}oh${oh_2}ow${ow_2}kh${kh_2}kw${kw_2}sh${sh_2}sw${sw_2}ph${ph_2}pw${pw_2}n"

echo "workload_name: ${workload_name}"
echo "layer_1_desc: ${layer_1_desc}"
echo "layer_2_desc: ${layer_2_desc}"