/*
 * Copyright (c) 2022 HiSilicon (Shanghai) Technologies CO., LIMITED.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#include "sample_comm_nnie.h"
#include "ai_infer_process.h"
#include "sample_media_ai.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

#define MODEL_FILE_HAND    "/userdata/models/hand_classify/hand_detect.wk" // darknet framework wk model
// #define MODEL_FILE_HAND    "/userdata/models/hand_classify/human_detect_inst.wk" // darknet framework wk model
#define PIRIOD_NUM_MAX     49 // Logs are printed when the number of targets is detected
#define DETECT_OBJ_MAX     32 // detect max obj
static uintptr_t g_handModel = 0;

// 
#define MODEL_FILE_POSE    "/userdata/models/pose_detect/yolox_pose.wk" // darknet framework wk model
static uintptr_t g_poseModel = 0;



static HI_S32 Yolo2FdLoad(uintptr_t* model)
{
    SAMPLE_SVP_NNIE_CFG_S *self = NULL;
    HI_S32 ret;

    ret = Yolo2Create(&self, MODEL_FILE_HAND);
    *model = ret < 0 ? 0 : (uintptr_t)self;
    SAMPLE_PRT("Yolo2FdLoad ret:%d\n", ret);

    return ret;
}

static HI_S32 YoloXFdLoad(uintptr_t* model)
{
    SAMPLE_SVP_NNIE_CFG_S *self = NULL;
    HI_S32 ret;

    ret = YoloXCreate(&self, MODEL_FILE_POSE);
    *model = ret < 0 ? 0 : (uintptr_t)self;
    SAMPLE_PRT("YoloXFdLoad ret:%d\n", ret);

    return ret;
}




HI_S32 HandDetectInit()
{
    return Yolo2FdLoad(&g_handModel);
}

HI_S32 PoseDetectInit()
{
    return YoloXFdLoad(&g_poseModel);
}





static HI_S32 Yolo2FdUnload(uintptr_t model)
{
    Yolo2Destory((SAMPLE_SVP_NNIE_CFG_S*)model);
    return 0;
}

static HI_S32 YoloXFdUnload(uintptr_t model)
{
    YoloXDestory((SAMPLE_SVP_NNIE_CFG_S*)model);
    return 0;
}





HI_S32 PoseDetectExit()
{
    return YoloXFdUnload(g_poseModel);
}

HI_S32 HandDetectExit()
{
    return Yolo2FdUnload(g_handModel);
}





static HI_S32 PoseDetect(uintptr_t model, IVE_IMAGE_S *src, PoseObjInfo kpts[])
{
    // printf("[POSE_DETECT] ============= \n");
    SAMPLE_SVP_NNIE_CFG_S *self = (SAMPLE_SVP_NNIE_CFG_S*)model;
    int objNum;
    // TODO: 具体调用地方
    // 网络，原始图像，输出kpts，最大检测数量，实际检测数量
    // SAMPLE_PRT("                                                    src_width: %d\n", src->u32Width);
    // SAMPLE_PRT("                                                    src_height: %d\n", src->u32Height);
    // YUV图像，存放结果的kpts，最大检测数量，以及真是检测到的数量
    int ret = YoloXCalImg(self, src, kpts, DETECT_OBJ_MAX, &objNum);
    if (ret < 0) {
        SAMPLE_PRT("Hand detect YoloXCalImg FAIL, for cal FAIL, ret:%d\n", ret);
        return ret;
    }

    return objNum;
}






static HI_S32 HandDetect(uintptr_t model, IVE_IMAGE_S *srcYuv, DetectObjInfo boxs[])
{
    SAMPLE_SVP_NNIE_CFG_S *self = (SAMPLE_SVP_NNIE_CFG_S*)model;
    int objNum;
    int ret = Yolo2CalImg(self, srcYuv, boxs, DETECT_OBJ_MAX, &objNum);
    if (ret < 0) {
        SAMPLE_PRT("Hand detect Yolo2CalImg FAIL, for cal FAIL, ret:%d\n", ret);
        return ret;
    }

    return objNum;
}





// 检测Yuv图像， 结果存放在resArr中
HI_S32 PoseDetectCal(IVE_IMAGE_S *srcYuv, PoseObjInfo resArr[])
{
    // 调用posedetect，模型，原始图像
    // 传入的是 g_poseModel 没问题 
    int ret = PoseDetect(g_poseModel, srcYuv, resArr);
    return ret;
}

HI_S32 HandDetectCal(IVE_IMAGE_S *srcYuv, DetectObjInfo resArr[])
{
    int ret = HandDetect(g_handModel, srcYuv, resArr);
    return ret;
}




#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */
