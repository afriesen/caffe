#!/bin/sh

## MODIFY PATH for YOUR SETTING
ROOT_DIR=

CAFFE_DIR=..
#CAFFE_DIR=../code
#CAFFE_BIN=${CAFFE_DIR}/.build_release/tools/caffe.bin
CAFFE_BIN=${CAFFE_DIR}/build/tools/caffe

EXP=sbd

if [ "${EXP}" = "sbd" ]; then
    NUM_LABELS=8
#    DATA_ROOT=${ROOT_DIR}/rmt/data/pascal/VOCdevkit/VOC2012
#    DATA_ROOT="/home/afriesen/proj/data/VOCdevkit/VOC2012"
    DATA_ROOT="/Users/afriesen/proj/sspn/data/iccv09Data/"
else
    NUM_LABELS=0
    echo "Wrong exp name"
fi
 

## Specify which model to train
########### voc12 ################
#NET_ID=deelab_largeFOV
NET_ID=deeplab_resnet101
#NET_ID=deeplab_vgg

FOLD_SUFFIX=".1"
#FOLD_SUFFIX=".oneimg"

## Variables used for weakly or semi-supervisedly training
#TRAIN_SET_SUFFIX=
#TRAIN_SET_SUFFIX=_aug

#TRAIN_SET_STRONG=train
#TRAIN_SET_STRONG=train200
#TRAIN_SET_STRONG=train500
#TRAIN_SET_STRONG=train1000
#TRAIN_SET_STRONG=train750

#TRAIN_SET_WEAK_LEN=5000

USE_GPU=1
DEV_ID=2

#####

## Create dirs

CONFIG_DIR=${EXP}/config/${NET_ID}
MODEL_DIR=${EXP}/model/${NET_ID}
mkdir -p ${MODEL_DIR}
LOG_DIR=${EXP}/log/${NET_ID}
mkdir -p ${LOG_DIR}
export GLOG_log_dir=${LOG_DIR}

## Run

RUN_TRAIN=1
RUN_TEST=1
RUN_TRAIN2=0
RUN_TEST2=0

## Training #1 (on train_aug)

if [ ${RUN_TRAIN} -eq 1 ]; then
    #
    LIST_DIR=${EXP}/list
    TRAIN_SET=train${TRAIN_SET_SUFFIX}${FOLD_SUFFIX}
#    if [ -z ${TRAIN_SET_WEAK_LEN} ]; then
#				TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}
#				comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
#    else
#				TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}_head${TRAIN_SET_WEAK_LEN}
#				comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt | head -n ${TRAIN_SET_WEAK_LEN} > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
#    fi
    #
    MODEL=${EXP}/model/${NET_ID}/init.caffemodel
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
#                CMD="${CMD} --snapshot=${EXP}/model/${NET_ID}/${TRAIN_SET}_iter_2026.solverstate"

		echo Running ${CMD} && ${CMD}
fi

## Test #1 specification (on val or test)

if [ ${RUN_TEST} -eq 1 ]; then
    #
    for TEST_SET in val; do
		TEST_SET=${TEST_SET}${FOLD_SUFFIX}
		TEST_ITER=`cat ${EXP}/list/${TEST_SET}.txt | wc -l | sed 's/^ *//'`
		echo TEST ITER = "${TEST_ITER}"
		MODEL=${EXP}/model/${NET_ID}/test${FOLD_SUFFIX}.caffemodel
		if [ ! -f ${MODEL} ]; then
				MODEL=`ls -t ${EXP}/model/${NET_ID}/train${FOLD_SUFFIX}_iter_*.caffemodel | head -n 1`
		fi
		#
		echo Testing net ${EXP}/${NET_ID} with fold suffix ${FOLD_SUFFIX} with weights from ${MODEL} \(test set = ${TEST_SET}\)
		FEATURE_DIR=${EXP}/features/${NET_ID}
		mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
#		mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc9
#		mkdir -p ${FEATURE_DIR}/${TEST_SET}/seg_score
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
    MODEL=${EXP}/model/${NET_ID}/init2${FOLD_SUFFIX}.caffemodel
    if [ ! -f ${MODEL} ]; then
				MODEL=`ls -t ${EXP}/model/${NET_ID}/train${FOLD_SUFFIX}_iter_*.caffemodel | head -n 1`
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
		MODEL=${EXP}/model/${NET_ID}/test2${FOLD_SUFFIX}.caffemodel
		if [ ! -f ${MODEL} ]; then
			MODEL=`ls -t ${EXP}/model/${NET_ID}/train2${FOLD_SUFFIX}_iter_*.caffemodel | head -n 1`
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
