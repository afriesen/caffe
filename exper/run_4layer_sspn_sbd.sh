#!/bin/sh

PRINT_CMD_ONLY=0

## MODIFY PATH for YOUR SETTING
ROOT_DIR=

CAFFE_DIR=..
SSPN_DIR=../../../sspn
#CAFFE_DIR=../code
#CAFFE_BIN=${CAFFE_DIR}/.build_release/tools/caffe.bin
#CAFFE_BIN=${CAFFE_DIR}/build/tools/caffe
CAFFE_BIN=${SSPN_DIR}/build/caffe_sspn

EXP=sbd

if [ "${EXP}" = "sbd" ]; then
#    NUM_LABELS=8
    NUM_LABELS=15
#    DATA_ROOT=${ROOT_DIR}/rmt/data/pascal/VOCdevkit/VOC2012
#    DATA_ROOT="/home/afriesen/proj/data/VOCdevkit/VOC2012"
#    DATA_ROOT="/Users/afriesen/proj/sspn/data/iccv09Data/"
    DATA_ROOT="/home/afriesen/proj/sspn/data/iccv09Data/"
else
    NUM_LABELS=0
    echo "Wrong exp name"
fi
 

## Specify which model to train
########### voc12 ################
#NET_ID=deelab_largeFOV
#NET_ID=deeplab_resnet101
NET_ID=sspn_4layer_deeplab_resnet101

FOLD_SUFFIX=".1"
#FOLD_SUFFIX=".oneimg"

USE_GPU=1
DEV_ID=1

#####

## Create dirs

CONFIG_DIR=${EXP}/config/${NET_ID}
MODEL_DIR=${EXP}/model/${NET_ID}
#MODEL_DIR=${EXP}/model/${NET_ID}/deeplab_trained
#MODEL_DIR=sbd/model/deeplab_resnet101/fold1/sqr_lr1e-4/
mkdir -p ${MODEL_DIR}
LOG_DIR=${EXP}/log/${NET_ID}
mkdir -p ${LOG_DIR}
export GLOG_log_dir=${LOG_DIR}

## Run

RUN_TRAIN=0
RUN_TEST=1
RUN_TRAIN2=0
RUN_TEST2=0

## Training #1 (on train_aug)

if [ ${RUN_TRAIN} -eq 1 ]; then
    #
    LIST_DIR=${EXP}/list
    TRAIN_SET=train${TRAIN_SET_SUFFIX}${FOLD_SUFFIX}
    #
    #MODEL=${MODEL_DIR}/init.caffemodel
#    MODEL=${EXP}/model/init.caffemodel
#    MODEL=${MODEL_DIR}/train.1_iter_10000_orig.caffemodel
    MODEL=${MODEL_DIR}/train.1_2layers_iter_10000.caffemodel
    #
    echo Training net ${EXP}/${NET_ID} with fold suffix ${FOLD_SUFFIX} using weights from ${MODEL}
    for pname in train solver; do
        sed "$(eval echo $(cat sub.sed))" \
            ${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
    done
        CMD="${CAFFE_BIN} train \
           --solver=${CONFIG_DIR}/solver_${TRAIN_SET}.prototxt" 
        if [ ${USE_GPU} -ne 0 ]; then
            CMD="${CMD} --gpu=${DEV_ID}"
        fi
        if [ -f ${MODEL} ]; then
            CMD="${CMD} --weights=${MODEL}"
	fi
#       CMD="${CMD} --snapshot=${MODEL_DIR}/${TRAIN_SET}_iter_1196.solverstate"
    if [ ${PRINT_CMD_ONLY} -ne 0 ]; then
        echo Running ${CMD}
    else
        echo Running ${CMD} && ${CMD}
    fi
fi

## Test #1 specification (on val or test)

if [ ${RUN_TEST} -eq 1 ]; then
    #
    for TEST_SET in val; do
#    for TEST_SET in test; do
        TEST_SET=${TEST_SET}${FOLD_SUFFIX}
        TEST_ITER=`cat ${EXP}/list/${TEST_SET}.txt | wc -l | sed 's/^ *//'`
        echo TEST ITER = "${TEST_ITER}"

        MODEL=${MODEL_DIR}/test${FOLD_SUFFIX}.caffemodel
        if [ ! -f ${MODEL} ]; then
#            MODEL=`ls -t ${MODEL_DIR}/train${FOLD_SUFFIX}_iter_*.caffemodel | head -n 1`
            MODEL=`ls -t ${MODEL_DIR}/train${FOLD_SUFFIX}_*layers_iter_*.caffemodel | head -n 1`
        fi
	#
        echo Testing net ${EXP}/${NET_ID} with fold suffix ${FOLD_SUFFIX} with weights from ${MODEL} \(test set = ${TEST_SET}\)

        FEATURE_DIR=${EXP}/features/${NET_ID}
        mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc1

        sed "$(eval echo $(cat sub.sed))" \
            ${CONFIG_DIR}/test.prototxt > ${CONFIG_DIR}/test_${TEST_SET}.prototxt
        CMD="${CAFFE_BIN} test \
             --model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
             --weights=${MODEL} \
             --iterations=${TEST_ITER}"
#             --gpu=${DEV_ID} \
        if [ ${USE_GPU} -ne 0 ]; then
            CMD="${CMD} --gpu=${DEV_ID}"
        fi
#        echo Running ${CMD} && ${CMD}
        if [ ${PRINT_CMD_ONLY} -ne 0 ]; then
            echo Running ${CMD}
        else
            echo Running ${CMD} && ${CMD}
        fi
    done
fi

## Training #2 (finetune on trainval_aug)

if [ ${RUN_TRAIN2} -eq 1 ]; then
    #
    LIST_DIR=${EXP}/list
    TRAIN_SET=trainval${TRAIN_SET_SUFFIX}${FOLD_SUFFIX}
#    if [ -z ${TRAIN_SET_WEAK_LEN} ]; then
#				TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}
#				comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
#    else
#				TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}_head${TRAIN_SET_WEAK_LEN}
#				comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt | head -n ${TRAIN_SET_WEAK_LEN} > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
#    fi
    #
    MODEL=${MODEL_DIR}/init2${FOLD_SUFFIX}.caffemodel
    if [ ! -f ${MODEL} ]; then
				MODEL=`ls -t ${MODEL_DIR}/train${FOLD_SUFFIX}_iter_*.caffemodel | head -n 1`
    fi
    #
    echo Training2 net ${EXP}/${NET_ID}
    for pname in train solver2; do
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
    done
    CMD="${CAFFE_BIN} train \
         --solver=${CONFIG_DIR}/solver2_${TRAIN_SET}.prototxt \
         --weights=${MODEL}"
#         --gpu=${DEV_ID}"
    if [ ${USE_GPU} -ne 0 ]; then
      	CMD="${CMD} --gpu=${DEV_ID}"
	fi
	echo Running ${CMD} && ${CMD}
fi

## Test #2 on official test set

if [ ${RUN_TEST2} -eq 1 ]; then
    #
    for TEST_SET in val test; do
#    for TEST_SET in val; do
		TEST_SET = ${TEST_SET}${FOLD_SUFFIX}
		TEST_ITER=`cat ${EXP}/list/${TEST_SET}.txt | wc -l | sed 's/^ *//'`
		MODEL=${MODEL_DIR}/test2${FOLD_SUFFIX}.caffemodel
		if [ ! -f ${MODEL} ]; then
			MODEL=`ls -t ${MODEL_DIR}/train2${FOLD_SUFFIX}_iter_*.caffemodel | head -n 1`
		fi
		#
		echo Testing2 net ${EXP}/${NET_ID}
		FEATURE_DIR=${EXP}/features2/${NET_ID}
		mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
		mkdir -p ${FEATURE_DIR}/${TEST_SET}/crf
		mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc1
		sed "$(eval echo $(cat sub.sed))" \
				${CONFIG_DIR}/test.prototxt > ${CONFIG_DIR}/test_${TEST_SET}.prototxt
		CMD="${CAFFE_BIN} test \
             --model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
             --weights=${MODEL} \
             --iterations=${TEST_ITER}"
#             --gpu=${DEV_ID} \
    	if [ ${USE_GPU} -ne 0 ]; then
      		CMD="${CMD} --gpu=${DEV_ID}"
		fi
		echo Running ${CMD} && ${CMD}
    done
fi
