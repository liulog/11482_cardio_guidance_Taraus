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

/*
 * 本文件将垃圾分类wk模型部署到板端，通过NNIE硬件加速进行推理。该文件提供了垃圾分类场景的API接口，
 * 包括模型的加载、模型的卸载、模型的推理、AI flag业务处理接口。支持语音实时播放功能。
 *
 * This file deploys the trash classification wk model to the board,
 * and performs inference through NNIE hardware acceleration.
 * This file provides API interfaces for trash classification scenarios,
 * including model loading, model unloading, model reasoning,
 * and AI flag business processing interfaces. Support audio real-time playback function.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <sys/prctl.h>

#include "sample_comm_nnie.h"
#include "sample_media_ai.h"
#include "ai_infer_process.h"
#include "vgs_img.h"
#include "ive_img.h"
#include "posix_help.h"
#include "audio_aac_adp.h"
#include "base_interface.h"
#include "osd_img.h"
#include "cnn_trash_classify.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

#define MODEL_FILE_TRASH    "/userdata/models/cnn_trash_classify/resnet_inst.wk" // Open source model conversion
#define SCORE_MAX           4096    // The score corresponding to the maximum probability
#define DETECT_OBJ_MAX      32
#define RET_NUM_MAX         4
#define THRESH_MIN          30      // Acceptable probability threshold (over this value will be returned to the app)

#define FRM_WIDTH           256
#define FRM_HEIGHT          256
#define TXT_BEGX            20
#define TXT_BEGY            20

static int g_num = 108;
static int g_count = 0;
#define AUDIO_CASE_TWO     2
#define AUDIO_SCORE        40       // Confidence can be configured by yourself
#define AUDIO_FRAME        14       // Recognize once every 15 frames, can be configured by yourself

#define MULTIPLE_OF_EXPANSION 100   // Multiple of expansion
#define UNKOWN_WASTE          20    // Unkown Waste
#define BUFFER_SIZE           16    // buffer size
#define MIN_OF_BOX            16    // min of box
#define MAX_OF_BOX            240   // max of box

static HI_BOOL g_bAudioProcessStopSignal = HI_FALSE;
static pthread_t g_audioProcessThread = 0;
static OsdSet* g_osdsTrash = NULL;
static HI_S32 g_osd0Trash = -1;

static SkPair g_stmChn = {
    .in = -1,
    .out = -1
};

/*
 * 将识别的结果进行音频播放
 * Audio playback of the recognition results
 */
static HI_VOID PlayAudio(const RecogNumInfo items)
{   

    // items就是识别到的结果，连续调用15帧之后才会播报语音
    if  (g_count < AUDIO_FRAME) {
        g_count++;
        return;
    }

    const RecogNumInfo *item = &items;
    uint32_t score = item->score * MULTIPLE_OF_EXPANSION / SCORE_MAX;

    // 得分大于播报阈值，并且类别不是上一次识别到的类别就更新，然后播报语音
    if ((score > AUDIO_SCORE) && (g_num != item->num)) {
        g_num = item->num;
        if (g_num != UNKOWN_WASTE) {

            // 主要就在这里
            AudioTest(g_num, -1);
        }
    }
    g_count = 0;
}

static HI_VOID* GetAudioFileName(HI_VOID* arg)
{
    RecogNumInfo resBuf = {0};
    int ret;

    // 通过这个信号唤醒这个函数
    while (g_bAudioProcessStopSignal == false) {    // 收到信号
        ret = FdReadMsg(g_stmChn.in, &resBuf, sizeof(RecogNumInfo));
        if (ret == sizeof(RecogNumInfo)) {
            PlayAudio(resBuf);
        }
    }

    return NULL;
}

/*
 * 加载垃圾分类wk模型
 * Load the trash classification wk model
 */
HI_S32 CnnTrashClassifyLoadModel(uintptr_t* model, OsdSet* osds)
{
    SAMPLE_SVP_NNIE_CFG_S *self = NULL;
    HI_S32 ret;
    HI_CHAR audioThreadName[BUFFER_SIZE] = {0};

    ret = OsdLibInit();
    HI_ASSERT(ret == HI_SUCCESS);

    g_osdsTrash = osds;
    HI_ASSERT(g_osdsTrash);
    g_osd0Trash = OsdsCreateRgn(g_osdsTrash);
    HI_ASSERT(g_osd0Trash >= 0);

    ret = CnnCreate(&self, MODEL_FILE_TRASH);
    *model = ret < 0 ? 0 : (uintptr_t)self;
    SAMPLE_PRT("load cnn trash classify model, ret:%d\n", ret);


    // 检查是否支持音频
    if (GetCfgBool("audio_player:support_audio", true)) {
        ret = SkPairCreate(&g_stmChn);
        HI_ASSERT(ret == 0);
        if (snprintf_s(audioThreadName, BUFFER_SIZE, BUFFER_SIZE - 1, "AudioProcess") < 0) {
            HI_ASSERT(0);
        }
        prctl(PR_SET_NAME, (unsigned long)audioThreadName, 0, 0, 0);
        ret = pthread_create(&g_audioProcessThread, NULL, GetAudioFileName, NULL);  // 创建了 语音处理 的线程
        if (ret != 0) {
            SAMPLE_PRT("audio proccess thread creat fail:%s\n", strerror(ret));
            return ret;
        }
    }

    return ret;
}

/*
 * 卸载垃圾分类wk模型
 * Unload the trash classification wk model
 */
HI_S32 CnnTrashClassifyUnloadModel(uintptr_t model)
{
    CnnDestroy((SAMPLE_SVP_NNIE_CFG_S*)model);
    SAMPLE_PRT("unload trash classify model success\n");
    OsdsClear(g_osdsTrash);

    if (GetCfgBool("audio_player:support_audio", true)) {
        SkPairDestroy(&g_stmChn);
        SAMPLE_PRT("SkPairDestroy success\n");
        g_bAudioProcessStopSignal = HI_TRUE;
        pthread_join(g_audioProcessThread, NULL);
        g_audioProcessThread = 0;
    }

    return HI_SUCCESS;
}

/*
 * 根据推理结果进行业务处理
 * Perform business processing based on inference results
 */
static HI_S32 CnnTrashClassifyFlag(const RecogNumInfo items[], HI_S32 itemNum, HI_CHAR* buf, HI_S32 size)
{
    HI_S32 offset = 0;
    HI_CHAR *trashName = NULL;

    offset += snprintf_s(buf + offset, size - offset, size - offset - 1, "trash classify: {");
    for (HI_U32 i = 0; i < itemNum; i++) {
        const RecogNumInfo *item = &items[i];
        uint32_t score = item->score * HI_PER_BASE / SCORE_MAX;
        if (score < THRESH_MIN) {
            break;
        }
        SAMPLE_PRT("----trash item flag----num:%d, score:%d\n", item->num, score);
        switch (item->num) {
            case 0u:
            case 1u:
            case 2u:
            case 3u:
            case 4u:
            case 5u:
                trashName = "Kitchen Waste";
                break;
            case 6u:
            case 7u:
            case 8u:
            case 9u:
                trashName = "Harmful Waste";
                break;
            case 10u:
            case 11u:
            case 12u:
            case 13u:
            case 14u:
            case 15u:
                trashName = "Recyle Waste";
                break;
            case 16u:
            case 17u:
            case 18u:
            case 19u:
                trashName = "Other Waste";
                break;
            default:
                trashName = "Unkown Waste";
                break;
        }
        offset += snprintf_s(buf + offset, size - offset, size - offset - 1,
            "%s%s %u:%u%%", (i == 0 ? " " : ", "), trashName, (int)item->num, (int)score);
        HI_ASSERT(offset < size);
    }
    offset += snprintf_s(buf + offset, size - offset, size - offset - 1, " }");
    HI_ASSERT(offset < size);
    return HI_SUCCESS;
}

/*
 * 先进行预处理，再使用NNIE进行硬件加速推理，不支持层通过AI CPU进行计算
 *
 * Perform preprocessing first, and then use NNIE for hardware accelerated inference,
 * and do not support layers to be calculated by AI CPU
 */
HI_S32 CnnTrashClassifyCal(uintptr_t model, VIDEO_FRAME_INFO_S *srcFrm, VIDEO_FRAME_INFO_S *resFrm)
{
    SAMPLE_PRT("begin CnnTrashClassifyCal\n");
    SAMPLE_SVP_NNIE_CFG_S *self = (SAMPLE_SVP_NNIE_CFG_S*)model; // reference to SDK sample_comm_nnie.h Line 99
    IVE_IMAGE_S img; // referece to SDK hi_comm_ive.h Line 143
    RectBox cnnBoxs[DETECT_OBJ_MAX] = {0};
    VIDEO_FRAME_INFO_S resizeFrm;  // Meet the input frame of the plug
    static HI_CHAR prevOsd[NORM_BUF_SIZE] = "";
    HI_CHAR osdBuf[NORM_BUF_SIZE] = "";
    /*
        01-Kitchen_Watermelon_rind    02_Kitchen_Egg_shell
        03_Kitchen_Fishbone           04_Kitchen_Eggplant
        05_Kitchen_Scallion           06_Kitchen_Mushromm
        07_Hazardous_Waste_battery    08_Hazardous_Expired_cosmetrics
        09_Hazardous_Woundplast       10_Hazardous_Medical_gauze
        11_Recyclabel_Old_dolls       12_Recyclabel_Old_clip
        13_Recyclabel_Toothbrush      14_Recyclabel_Milk_box
        15_Recyclabel_Old_handbag     16_Recyclabel_Zip_top_can
        17_other_Ciggrate_end         18_Other_Bad_closestool
        19_other_Brick                20_Other_Dish
        21_unkown_waste_or_background
    */
    RecogNumInfo resBuf[RET_NUM_MAX] = {0};
    HI_S32 resLen = 0;
    HI_S32 ret;
    IVE_IMAGE_S imgIn;

    cnnBoxs[0].xmin = MIN_OF_BOX;
    cnnBoxs[0].xmax = MAX_OF_BOX;
    cnnBoxs[0].ymin = MIN_OF_BOX;
    cnnBoxs[0].ymax = MAX_OF_BOX;

    ret = MppFrmResize(srcFrm, &resizeFrm, FRM_WIDTH, FRM_HEIGHT);  // resize 256*256
    SAMPLE_CHECK_EXPR_RET(ret != HI_SUCCESS, ret, "for resize FAIL, ret=%x\n", ret);

    ret = FrmToOrigImg(&resizeFrm, &img);
    SAMPLE_CHECK_EXPR_RET(ret != HI_SUCCESS, ret, "for Frm2Img FAIL, ret=%x\n", ret);

    ret = ImgYuvCrop(&img, &imgIn, &cnnBoxs[0]); // Crop the image to classfication network
    SAMPLE_CHECK_EXPR_RET(ret < 0, ret, "ImgYuvCrop FAIL, ret=%x\n", ret);

    ret = CnnCalImg(self, &imgIn, resBuf, sizeof(resBuf) / sizeof((resBuf)[0]), &resLen);
    SAMPLE_CHECK_EXPR_RET(ret < 0, ret, "cnn cal FAIL, ret=%x\n", ret);

    HI_ASSERT(resLen <= sizeof(resBuf) / sizeof(resBuf[0]));
    ret = CnnTrashClassifyFlag(resBuf, resLen, osdBuf, sizeof(osdBuf));
    SAMPLE_CHECK_EXPR_RET(ret < 0, ret, "CnnTrashClassifyFlag cal FAIL, ret=%x\n", ret);

    // 
    if (GetCfgBool("audio_player:support_audio", true)) {
        // 写 8个字符
        if (FdWriteMsg(g_stmChn.out, &resBuf[0], sizeof(RecogNumInfo)) != sizeof(RecogNumInfo)) {
            SAMPLE_PRT("FdWriteMsg FAIL\n");
        }
    }

    /*
     * 仅当计算结果与之前计算发生变化时，才重新打OSD输出文字
     * Only when the calculation result changes from the previous calculation, re-print the OSD output text
     */
    if (strcmp(osdBuf, prevOsd) != 0) {
        HiStrxfrm(prevOsd, osdBuf, sizeof(prevOsd));
        HI_OSD_ATTR_S rgn;
        TxtRgnInit(&rgn, osdBuf, TXT_BEGX, TXT_BEGY, ARGB1555_YELLOW2); // font width and heigt use default 40
        OsdsSetRgn(g_osdsTrash, g_osd0Trash, &rgn);
        /*
         * 用户向VPSS发送数据
         * User sends data to VPSS
         */
        ret = HI_MPI_VPSS_SendFrame(0, 0, srcFrm, 0);
        if (ret != HI_SUCCESS) {
            SAMPLE_PRT("Error(%#x), HI_MPI_VPSS_SendFrame failed!\n", ret);
        }
    }

    IveImgDestroy(&imgIn);
    MppFrmDestroy(&resizeFrm);

    return ret;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */
