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
 * 该文件提供了基于yolov2的手部检测以及基于resnet18的手势识别，属于两个wk串行推理。
 * 该文件提供了手部检测和手势识别的模型加载、模型卸载、模型推理以及AI flag业务处理的API接口。
 * 若一帧图像中出现多个手，我们通过算法将最大手作为目标手送分类网进行推理，
 * 并将目标手标记为绿色，其他手标记为红色。
 *
 * This file provides hand detection based on yolov2 and gesture recognition based on resnet18,
 * which belongs to two wk serial inferences. This file provides API interfaces for model loading,
 * model unloading, model reasoning, and AI flag business processing for hand detection
 * and gesture recognition. If there are multiple hands in one frame of image,
 * we use the algorithm to use the largest hand as the target hand for inference,
 * and mark the target hand as green and the other hands as red.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <sys/time.h>

#include "sample_comm_nnie.h"
#include "sample_media_ai.h"
#include "ai_infer_process.h"
#include "yolov2_hand_detect.h"
#include "vgs_img.h"
#include "ive_img.h"
#include "misc_util.h"
#include "hisignalling.h"
#include "hand_classify.h"

// 音频播放需要的，暂时不太需要 用进程通信
#include "audio_aac_adp.h"
#include "posix_help.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

// YOLOV2输入
#define HAND_FRM_WIDTH     640
#define HAND_FRM_HEIGHT    384

// 新增加的内容
#define POSE_FRM_WIDTH     416
#define POSE_FRM_HEIGHT    416

#define DETECT_OBJ_MAX     32
#define RET_NUM_MAX        4
#define DRAW_RETC_THICK    2    // Draw the width of the line
#define WIDTH_LIMIT        32
#define HEIGHT_LIMIT       32

#define IMAGE_WIDTH        224  // The resolution of the model IMAGE sent to the classification is 224*224
#define IMAGE_HEIGHT       224

// 后续手势识别需要的分类内容
#define MODEL_FILE_GESTURE    "/userdata/models/hand_classify/hand_gesture.wk" // darknet framework wk model

static int biggestBoxIndex;
static IVE_IMAGE_S img;
static DetectObjInfo objs[DETECT_OBJ_MAX] = {0};

// 仿照objs增加的内容, 存放最终的结果
static PoseObjInfo poseobjs[DETECT_OBJ_MAX] = {0};

static RectBox boxs[DETECT_OBJ_MAX] = {0};
static RectBox objBoxs[DETECT_OBJ_MAX] = {0};
static RectBox remainingBoxs[DETECT_OBJ_MAX] = {0};
static RectBox cnnBoxs[DETECT_OBJ_MAX] = {0}; // Store the results of the classification network

int uartFd = 0;

int uartFd_pose = 0;

// 1920, 1080
// 420,0        1500,1080
static RectBox POSE_ROI = {0};

static RecogNumInfo numInfo[RET_NUM_MAX] = {0};
static IVE_IMAGE_S imgIn;
static IVE_IMAGE_S imgDst;
static VIDEO_FRAME_INFO_S frmIn;
static VIDEO_FRAME_INFO_S frmDst;


static HI_BOOL g_bAudioProcessStopSignal = HI_FALSE;
static pthread_t g_audioProcessThread = 0;
static OsdSet* g_osdsTrash = NULL;
static HI_S32 g_osd0Trash = -1;

static SkPair g_stmChn = {
    .in = -1,
    .out = -1
};


static int g_num = 108;     // g_num 用于表示当前识别到的哪一个有问题，选择对应的文件进行语音播报
static int audio_count = 0;     // 帧数, 计数器, 不会对所有的都进行同样的播报

#define AUDIO_FRAME        25   // 在这里做了限制， 不会频繁的播放，多次调用之后才会播报

// 信号量，用于通知播放线程立即停止
// sem_t stop_signal;
pthread_t audio_thread;

// 线程函数，用于调用 AudioTest 并设置为detach状态， 直接播放
void* thread_function(void* args) {
    int arg1 = *((int*)args);
    int arg2 = *((int*)args + 1);

    // 调用 AudioTest 函数
    AudioTest(arg1, arg2);

    // 线程结束时，自动释放资源
    pthread_detach(pthread_self());
    return NULL;
}

int args[2] = {0, 0};

// 线程函数，用于调用 AudioTest 并设置为detach状态， 进行定时播放
// void* thread_function2(void* args) {
    // int arg1 = *((int*)args);
    // int arg2 = *((int*)args + 1);

    // 调用 AudioTest 函数
    // PlayAudio(arg1, arg2);

    // 线程结束时，自动释放资源
//     pthread_detach(pthread_self());
//     return NULL;
// }


/*
 * 将识别的结果进行音频播放
 * Audio playback of the recognition results
 */
static HI_VOID PlayAudio(const int num, int time)       // 积累到一定次数之后才会播报
{   

    // items就是识别到的结果，连续调用15帧之后才会播报语音
    if  (audio_count < AUDIO_FRAME) {
        audio_count++;
        return;
    }

    // const RecogNumInfo *item = &items;
    // uint32_t score = item->score * MULTIPLE_OF_EXPANSION / SCORE_MAX;

    // 得分大于播报阈值，并且类别不是上一次识别到的类别就更新，然后播报语音
    // if ( (g_num != num)) {
        // g_num = num;                        // 两次播报不一致才会继续提醒，不会一直提醒
        // 主要就在这里
        // 配置，然后播放g_num 语音
        // AudioTest(g_num, time);
        // 
    args[0] = num;
    args[1] = time;
    pthread_create(&audio_thread, NULL, thread_function, (void*)args);  // 播放对应语音
    // }
    audio_count = 0;
}


/*
 * 获得最大的手
 * Get the maximum hand
 */
static HI_S32 GetBiggestHandIndex(PoseObjInfo poses[], int detectNum)
{
    HI_S32 handIndex = 0;
    HI_S32 biggestBoxIndex = handIndex;
    HI_S32 biggestBoxWidth = poses[handIndex].box.xmax - poses[handIndex].box.xmin + 1;
    HI_S32 biggestBoxHeight = poses[handIndex].box.ymax - poses[handIndex].box.ymin + 1;
    HI_S32 biggestBoxArea = biggestBoxWidth * biggestBoxHeight;

    for (handIndex = 1; handIndex < detectNum; handIndex++) {
        HI_S32 boxWidth = poses[handIndex].box.xmax - poses[handIndex].box.xmin + 1;
        HI_S32 boxHeight = poses[handIndex].box.ymax - poses[handIndex].box.ymin + 1;
        HI_S32 boxArea = boxWidth * boxHeight;
        if (biggestBoxArea < boxArea) {
            biggestBoxArea = boxArea;
            biggestBoxIndex = handIndex;
        }
        biggestBoxWidth = poses[biggestBoxIndex].box.xmax - poses[biggestBoxIndex].box.xmin + 1;
        biggestBoxHeight = poses[biggestBoxIndex].box.ymax - poses[biggestBoxIndex].box.ymin + 1;
    }

    if ((biggestBoxWidth == 1) || (biggestBoxHeight == 1) || (detectNum == 0)) {
        biggestBoxIndex = -1;
    }

    return biggestBoxIndex;
}


/*
 * 手势识别信息
 * Hand gesture recognition info
 */
static void HandDetectFlag(const RecogNumInfo resBuf)
{
    HI_CHAR *gestureName = NULL;
    switch (resBuf.num) {
        case 0u:
            gestureName = "gesture fist";
            UartSendRead(uartFd, FistGesture); // 拳头手势
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        case 1u:
            gestureName = "gesture indexUp";
            UartSendRead(uartFd, ForefingerGesture); // 食指手势
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        case 2u:
            gestureName = "gesture OK";
            UartSendRead(uartFd, OkGesture); // OK手势
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        case 3u:
            gestureName = "gesture palm";
            UartSendRead(uartFd, PalmGesture); // 手掌手势
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        case 4u:
            gestureName = "gesture yes";
            UartSendRead(uartFd, YesGesture); // yes手势
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        case 5u:
            gestureName = "gesture pinchOpen";
            UartSendRead(uartFd, ForefingerAndThumbGesture); // 食指 + 大拇指
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        case 6u:
            gestureName = "gesture phoneCall";
            UartSendRead(uartFd, LittleFingerAndThumbGesture); // 大拇指 + 小拇指
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        default:
            gestureName = "gesture others";
            UartSendRead(uartFd, InvalidGesture); // 无效值
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
    }
    SAMPLE_PRT("hand gesture success\n");
}











/*
 * 加载手部检测和手势分类模型
 * Load hand detect and classify model
 */
HI_S32 Yolo2HandDetectResnetClassifyLoad(uintptr_t* model)
{
    SAMPLE_SVP_NNIE_CFG_S *self = NULL;
    HI_S32 ret;

    ret = CnnCreate(&self, MODEL_FILE_GESTURE);
    *model = ret < 0 ? 0 : (uintptr_t)self;
    HandDetectInit(); // Initialize the hand detection model
    SAMPLE_PRT("Load hand detect claasify model success\n");
    /*
     * Uart串口初始化
     * Uart open init
     */
    uartFd = UartOpenInit();        
    if (uartFd < 0) {
        printf("uart1 open failed\r\n");
    } else {
        printf("uart1 open successed\r\n");
    }
    return ret;
}



HI_S32 YoloxPoseDetectLoad(uintptr_t* model)
{
    // printf("[YOLOX_POSE_DETECT_LOAD] ============ \n");

    SAMPLE_SVP_NNIE_CFG_S *self = NULL;
    HI_S32 ret;

    // 不改这部分是为了避免其他影响，不改这部分就可以不需要改之前的手势识别后面的resnet
    ret = CnnCreate(&self, MODEL_FILE_GESTURE);
    *model = ret < 0 ? 0 : (uintptr_t)self;

    // 初始化这个pose模型
    PoseDetectInit(); // Initialize the hand detection model
    SAMPLE_PRT("Load pose detect model success\n");
    
    /*
     * Uart串口初始化
     * Uart open init
     */

    // 这里对应的串口是 ttyAMA1 , 注意串口
    // 对应打开一个串口
    
    // receiveData = (ReceiveData*)malloc(sizeof(ReceiveData));
    // uartReadData = (ReceiveData*)malloc(sizeof(ReceiveData));
    // uartSendData = (SendData*)malloc(sizeof(SendData));
    
    // uartFd_pose = UartOpenInit();
    // // 初始化了一个串口

    // if (uartFd_pose < 0) {
    //     printf("uart1 open failed\r\n");
    // } else {
    //     printf("uart1 open successed\r\n");
    // }

    // *model = 0;
    return ret;
}










/*
 * 卸载手部检测和手势分类模型
 * Unload hand detect and classify model
 */
HI_S32 Yolo2HandDetectResnetClassifyUnload(uintptr_t model)
{
    CnnDestroy((SAMPLE_SVP_NNIE_CFG_S*)model);
    HandDetectExit(); // Uninitialize the hand detection model
    close(uartFd);
    SAMPLE_PRT("Unload hand detect claasify model success\n");
    return 0;
}


HI_S32 YoloxPoseDetectUnload(uintptr_t model)
{
    CnnDestroy((SAMPLE_SVP_NNIE_CFG_S*)model);
    // HandDetectExit();
    PoseDetectExit();   // Uninitialize the hand detection model
    
    // free(receiveData);  // 释放
    // free(uartReadData);
    // free(uartSendData);
    
    // close(uartFd_pose); // 关闭对应的串口
    SAMPLE_PRT("Unload pose detect model success\n");
    return 0;
}







/*
 * 手部检测和手势分类推理
 * Hand detect and classify calculation
 */
HI_S32 Yolo2HandDetectResnetClassifyCal(uintptr_t model, VIDEO_FRAME_INFO_S *srcFrm, VIDEO_FRAME_INFO_S *dstFrm)
{
    SAMPLE_SVP_NNIE_CFG_S *self = (SAMPLE_SVP_NNIE_CFG_S*)model;
    HI_S32 resLen = 0;
    int objNum;
    int ret;
    int num = 0;

    // YUV转成RGB
    // ret = FrmToRgbImg((VIDEO_FRAME_INFO_S*)srcFrm, &img);
    // ret = ImgRgbToBgr(&img);

    ret = FrmToOrigImg((VIDEO_FRAME_INFO_S*)srcFrm, &img);

    SAMPLE_CHECK_EXPR_RET(ret != HI_SUCCESS, ret, "hand detect for YUV Frm to Img FAIL, ret=%#x\n", ret);

    objNum = HandDetectCal(&img, objs); // Send IMG to the detection net for reasoning
    for (int i = 0; i < objNum; i++) {
        cnnBoxs[i] = objs[i].box;
        RectBox *box = &objs[i].box;
        RectBoxTran(box, HAND_FRM_WIDTH, HAND_FRM_HEIGHT,
            dstFrm->stVFrame.u32Width, dstFrm->stVFrame.u32Height);
        SAMPLE_PRT("yolo2_out: {%d, %d, %d, %d}\n", box->xmin, box->ymin, box->xmax, box->ymax);
        boxs[i] = *box;
    }
    biggestBoxIndex = 0;
    biggestBoxIndex = GetBiggestHandIndex(boxs, objNum);
    SAMPLE_PRT("biggestBoxIndex:%d, objNum:%d\n", biggestBoxIndex, objNum);

    /*
     * 当检测到对象时，在DSTFRM中绘制一个矩形
     * When an object is detected, a rectangle is drawn in the DSTFRM
     */
    if (biggestBoxIndex >= 0) {
        objBoxs[0] = boxs[biggestBoxIndex];
        MppFrmDrawRects(dstFrm, objBoxs, 1, RGB888_GREEN, DRAW_RETC_THICK); // Target hand objnum is equal to 1

        for (int j = 0; (j < objNum) && (objNum > 1); j++) {
            if (j != biggestBoxIndex) {
                remainingBoxs[num++] = boxs[j];
                /*
                 * 其他手objnum等于objnum -1
                 * Others hand objnum is equal to objnum -1
                 */
                MppFrmDrawRects(dstFrm, remainingBoxs, objNum - 1, RGB888_RED, DRAW_RETC_THICK);
            }
        }

        /*
         * 裁剪出来的图像通过预处理送分类网进行推理
         * The cropped image is preprocessed and sent to the classification network for inference
         */
        ret = ImgYuvCrop(&img, &imgIn, &cnnBoxs[biggestBoxIndex]);
        SAMPLE_CHECK_EXPR_RET(ret < 0, ret, "ImgYuvCrop FAIL, ret=%#x\n", ret);

        if ((imgIn.u32Width >= WIDTH_LIMIT) && (imgIn.u32Height >= HEIGHT_LIMIT)) {
            COMPRESS_MODE_E enCompressMode = srcFrm->stVFrame.enCompressMode;
            ret = OrigImgToFrm(&imgIn, &frmIn);
            frmIn.stVFrame.enCompressMode = enCompressMode;
            SAMPLE_PRT("crop u32Width = %d, img.u32Height = %d\n", imgIn.u32Width, imgIn.u32Height);
            ret = MppFrmResize(&frmIn, &frmDst, IMAGE_WIDTH, IMAGE_HEIGHT);
            ret = FrmToOrigImg(&frmDst, &imgDst);
            ret = CnnCalImg(self,  &imgDst, numInfo, sizeof(numInfo) / sizeof((numInfo)[0]), &resLen);
            SAMPLE_CHECK_EXPR_RET(ret < 0, ret, "CnnCalImg FAIL, ret=%#x\n", ret);
            HI_ASSERT(resLen <= sizeof(numInfo) / sizeof(numInfo[0]));
            HandDetectFlag(numInfo[0]);
            MppFrmDestroy(&frmDst);
        }
        IveImgDestroy(&imgIn);
    }
    IveImgDestroy(&img);

    return ret;
}


// 讲一组坐标归一化, 备用
// void normalizeKeypoints(Keypoint keypoints[], int numKeypoints) {
//     // 寻找x和y的最小值和最大值
//     float minX = keypoints[0].x;
//     float maxX = keypoints[0].x;
//     float minY = keypoints[0].y;
//     float maxY = keypoints[0].y;

//     for (int i = 1; i < numKeypoints; i++) {
//         if (keypoints[i].x < minX) {
//             minX = keypoints[i].x;
//         }
//         if (keypoints[i].x > maxX) {
//             maxX = keypoints[i].x;
//         }
//         if (keypoints[i].y < minY) {
//             minY = keypoints[i].y;
//         }
//         if (keypoints[i].y > maxY) {
//             maxY = keypoints[i].y;
//         }
//     }

//     // 计算x和y的范围
//     float rangeX = maxX - minX;
//     float rangeY = maxY - minY;

//     // 归一化坐标
//     for (int i = 0; i < numKeypoints; i++) {
//         keypoints[i].x = (keypoints[i].x - minX) / rangeX;
//         keypoints[i].y = (keypoints[i].y - minY) / rangeY;
//     }
// }


// 计算 关键点中某三个点之间的角度
// 计算三个关键点之间的角度，也不需要归一化了, 也不需要矫正了
// float calculate_angle( PoseObjInfo* pose, int part_indexes[] ) {
//     float x1 =  pose->keypoints.keypoints[ 2 * part_indexes[0]];
//     float y1 = pose->keypoints.keypoints[ 2 * part_indexes[0] + 1];
//     // int v1 = keypoints[part_indexes[0]].visibility;
    
//     float x2 = pose->keypoints.keypoints[2 * part_indexes[1]];
//     float y2 = pose->keypoints.keypoints[2 * part_indexes[1] + 1];
//     // int v2 = keypoints[part_indexes[1]].visibility;

//     float x3 = pose->keypoints.keypoints[2 * part_indexes[2]];
//     float y3 = pose->keypoints.keypoints[2 * part_indexes[2] + 1];
//     // int v3 = keypoints[part_indexes[2]].visibility;
    
//     if( y3==y2 && x3==x2)
//         return 0;
//     if( y1==y2 && x1==x2)
//         return 0;
//     // if (v1 > 0 && v2 > 0 && v3 > 0) { // 确保关键点可见
    
//     float angle = atan2(y3 - y2, x3 - x2) - atan2(y1 - y2, x1 - x2);
    
//     angle = fmodf(angle * (180 / M_PI), 360);   // 将弧度转换为角度并限制在 0 到 360 度之间
//     if (angle < 0.0f) {                         // 将所有角限定在0-360度
//         angle += 360.0f;
//     }
    
//     return angle;
//     // } else {
//         // return NAN; // 若关键点不可见，则返回 NAN
//     // }
// }

float calculate_angle_with_cos( PoseObjInfo* pose, int part_indexes[] ) {
    float x1 =  pose->keypoints.keypoints[ 2 * part_indexes[0]];
    float y1 = pose->keypoints.keypoints[ 2 * part_indexes[0] + 1];
    // int v1 = keypoints[part_indexes[0]].visibility;
    
    float x2 = pose->keypoints.keypoints[2 * part_indexes[1]];
    float y2 = pose->keypoints.keypoints[2 * part_indexes[1] + 1];
    // int v2 = keypoints[part_indexes[1]].visibility;

    float x3 = pose->keypoints.keypoints[2 * part_indexes[2]];
    float y3 = pose->keypoints.keypoints[2 * part_indexes[2] + 1];
    // int v3 = keypoints[part_indexes[2]].visibility;

    if( y3==y2 && x3==x2)
        return 0;
    if( y1==y2 && x1==x2)
        return 0;
    // if (v1 > 0 && v2 > 0 && v3 > 0) { // 确保关键点可见
    
    float distance12 = sqrtf(powf(x2 - x1, 2) + powf(y2 - y1, 2));
    float distance23 = sqrtf(powf(x3 - x2, 2) + powf(y3 - y2, 2));
    float distance13 = sqrtf(powf(x3 - x1, 2) + powf(y3 - y1, 2));

    // float angle = atan2(y3 - y2, x3 - x2) - atan2(y1 - y2, x1 - x2);
    float cos_angle = (powf(distance12, 2) + powf(distance23, 2) - powf(distance13, 2)) / (2 * distance12 * distance23);
    float angle = acosf(cos_angle);

    angle = fmodf(angle * (180 / M_PI), 360);   // 将弧度转换为角度并限制在 0 到 360 度之间
    if (angle < 0.0f) {                         // 将所有角限定在0-360度
        angle += 360.0f;
    }
    
    return angle;
    // } else {
        // return NAN; // 若关键点不可见，则返回 NAN
    // }
}


// 左小臂
static int left_arm_small_index[] = {5, 7, 9};
// 左大臂
static int left_arm_big_index[] = {7, 5, 11};
// 右小臂
static int right_arm_small_index[] = {10, 8, 6};
// 右大臂
static int right_arm_big_index[] = {12, 6, 8};
// 左小腿
static int left_leg_small_index[] = {11, 13, 15};
// 左大腿
static int left_leg_big_index[] = {13, 11, 5};
// 右小腿
static int right_leg_small_index[] = {16, 14, 12};
// 右大腿
static int right_leg_big_index[] = {14, 12, 6};

////////////////////////////////////////
//      暂时停止使用标准动作
//
//   2D点不适合用来做高精度的锻炼
////////////////////////////////////////


// 存放标准的动作
// 其中减脂不需要和标准动作比对，也就不需要比较
// static float standard_action[][34]={
//     {   367.9154,  80.5427, 374.3532,  73.5137, 361.3801,  74.6169, 387.6558, 76.7188, 356.0481,  80.5031, 407.8863, 115.1285, 360.4904, 118.4781,
//         440.2998, 154.9889, 345.2270, 158.1436, 443.4957, 161.3785, 308.2241, 177.5476, 426.5242, 208.8545, 392.9198, 209.7236, 423.1604, 280.6751,
//         367.1153, 277.3203, 475.1979, 353.3342, 398.0457, 346.0401   },             // 测试动作: 谷爱凌 经典滑雪动作
//     {   1505.7450,  517.6636, 1569.0137,  464.0472, 1428.5017,  460.1899, 1647.2605,  562.4453, 1309.0588,  556.3184, 1789.2338, 1012.8788,
//         1150.8367,  983.7099, 1985.3356, 1427.0769,  845.3273, 1341.0758, 2268.6506, 1824.6118,  684.4734, 1683.1501, 1680.3955, 1965.3783,
//         1266.0643, 1963.2247, 1779.6865, 2643.8608, 1277.6105, 2643.7112, 1853.6089, 3307.8833, 1296.8743, 3319.6301},  // 测试动作
//     {},
//     {},
//     {}
// };

// 对应的8个角度, 跟上面的顺序一样  标准动作图片来源 https://github.com/open-mmlab/mmpose/tree/main/projects/yolox_pose
// static int standard_action_angle[][8]={
//     { 197.882431, 28.621984, 137.867889, 40.220417, 141.917542, 165.814636, 135.396835, 140.252243 },  // 测试动作: 谷爱凌 经典滑雪动作
//     { 169.797821, 31.805746, 195.220322, 47.180054, 178.051865, 165.167633, 179.315857, 174.176208 },
//     {},
//     {},
//     {}
// };



void read_uart_data(int uartFd, ReceiveData* data) {
    unsigned char buffer[sizeof(ReceiveData)];
    int readLen;

    // 从串口读取字节流
    readLen = UartRead(uartFd, buffer, sizeof(ReceiveData), 50);

    if (readLen == sizeof(ReceiveData)) {
        // 将字节流转换为数据结构
        memcpy(data, buffer, sizeof(ReceiveData));
    }
}


int compare_recive_data(const ReceiveData* data1, const ReceiveData* data2) {
    return memcmp(data1, data2, sizeof(ReceiveData));
}

// 添加一个语音播放的线程
void* play_audio_thread(void* arg) {
    printf("Audio playback thread started.\n");
    
    // 模拟音频播放循环
    // while (is_playing) {
        // printf("Playing audio...\n");
        // 在这里添加播放音频的代码
        // sleep(1); // 假设每次播放需要1秒钟
    // }
    AudioTest(70, 243);     // 将这段4min的语音设置为70号

    printf("Audio playback thread finished.\n");
    return NULL;
}



// 增加一些误识别缓冲, 2 帧, 缓冲为两帧

int g_count = 0;    // 蹲起, 开合跳的计数器
float gf_time = 0;  // global float time 用来计时 ms 数, 最后串口传回去的时间 单位是s
bool action_flag;   // 用来辅助计数, 默认当前处于站立状态, true时表示处于特定判断姿势

int judge_count = 0;
int judge_count_thresh = 5;

HI_VOID my_test(){
    uartReadData->startflag = '!';
    receiveData->startflag = '!';
    uartReadData->mode = 2;
    receiveData->mode = 2;
    uartReadData->action1 = 0x2;    // 这一个动作是什么
    receiveData->action1 = 0x2;
}

bool first_audio;   // 第一次播报

struct timeval start_time, end_time;
long long elapsed_time;

bool first_boot = true; // 第一次启动

// 将我放置于1m左右的高度, 距离3m左右, 效果最佳

bool mode4_first = true;
bool last_mode = 0;
bool calc_flag = false;
int mode4_time_cnt = -1; // 为了保证计算每次的图像只会处理一次
int mode4_score = 0;
bool first = true;



int diff;

bool begin_play = true;


/*
 * 姿态检测推理
 */
HI_S32 YoloxPoseDetectCal(uintptr_t model, VIDEO_FRAME_INFO_S *srcFrm, VIDEO_FRAME_INFO_S *dstFrm, int action)
{
    // 网络模型，检测结果部分相关变量
    SAMPLE_SVP_NNIE_CFG_S *self = (SAMPLE_SVP_NNIE_CFG_S*)model;
    HI_S32 resLen = 0;
    int objNum;
    int ret;
    int num = 0;

    // SAMPLE_PRT("------------------------- SEND %d !  \n", uartFd_pose);
    // 测试效果使用, 与串口通信时注释掉
    // my_test();
    // 测试串口没问题

    // if(begin_play){
        // getchar();
        // int a;
        // scanf("%d", &a);
        // begin_play = false;
    // }

    // 测试 娱乐模式, 看一下计时
    // if( first ){
    //     first = false;
    //     gettimeofday(&start_time, NULL);
    // }
    // sleep(1);   // 休眠1s

    // gettimeofday(&end_time,NULL);
    
    // int diff = (end_time.tv_sec * 1000 + end_time.tv_usec/1000) - (start_time.tv_sec * 1000 + start_time.tv_usec/1000);

    // SAMPLE_PRT("                %d \n", diff);
    // return HI_SUCCESS;

    // 从串口读取数据, 增加一个线程读写
    // read_uart_data(uartFd_pose, uartReadData);
    
    // 播放语音, 播放使用时的注意事项， 每次上电时进行语音提示，只有第一次
    if(first_boot){
        AudioTest(29, 12);    // 提示用户如何放置使用该设备
        SAMPLE_PRT("        [first boot]     \n");
        first_boot = false;
    }

    // if(uartReadData->mode==0){
        // SAMPLE_PRT("------------------------- SEND !  \n");
        // uartSendData->score = 254;
        // uartSendData->startflag = '!';
        // SAMPLE_PRT("------------------------- SEND %d !  \n", uartFd_pose);
        // UartSend(30, (char *)uartSendData, sizeof(SendData));

    // }

    // 为了测试和3861的通信，目前没问题
    // SAMPLE_PRT("uartFd                                            mode  %d, action %d \n", uartReadData->mode, uartReadData->action1);

    // 测试串口通信的数据
    // if(uartReadData->startflag!='!'){       // 起始位不对
    //     return HI_SUCCESS;
    // }

    // 比较串口收到的数据和保留的数据是否一致    
    int cmp = compare_recive_data(uartReadData, receiveData); // 比较存下来的数据是否和串口收到的数据完全一样
    
    // 两次比对数据不一致: 表明用户更换了模式 
    if(cmp != 0){
        // 这部分只是为 mode = 4 准备
        if((last_mode != uartReadData->mode) && (uartReadData->mode == 4)){ //第一次进入模式4，进行计时
            mode4_first = true;
            // 开始计时
            // 创建一个线程
            gettimeofday(&start_time, NULL);    // 获取当前的时间，如果时模式4的话
            // 创建线程，并detach
            
            // 语音 不同步放弃了板端播放背景音乐, 板端用来进行语音提示
            // int thread_create_result = pthread_create(&audio_thread, NULL, play_audio_thread, NULL);
            // if (thread_create_result) {
            //     printf("Failed to create audio playback thread, error code: %d\n", thread_create_result);
            //     return HI_SUCCESS;
            // }
        }
        // 这种情况下，需要杀死这个线程
        // if((last_mode != receiveData->mode)&&(last_mode == 4)){  // 这时候关闭这条语音
        //     pthread_cancel(audio_thread);
        // }

        // 记录上一次的mode, 为娱乐模式 计时 准备
        last_mode = receiveData->mode;
        mode4_time_cnt = -1;    // 对应的时间进行清楚
        mode4_score = 0;        // 得分清 0
        
        // 下面是其他部分都需要的， 拷贝mode 和 action
        // memcpy(uartReadData, receiveData, sizeof(ReceiveData));
        receiveData->action1 = uartReadData->action1;
        receiveData->mode = uartReadData->mode;
        receiveData->startflag = uartReadData->startflag;
        
        // 目前修改了多线程, 大约1秒1帧的频率获取
        g_count = 0;                    // 计数器清零
        action_flag = false;            // false 默认处于初始状态，辅助计数使用 
        first_audio = true;             // 对于模式1 坚持模式 中的动作开始前先播放动作标准提示
        return HI_SUCCESS;
    }

    // 读取的模式不符合要求的模式
    if(receiveData->mode < 1 || receiveData->mode > 4)
        return HI_SUCCESS;  

    diff = 0;                                           // diff用于表示当前和开始时间的ms数
    // 模式4要记录时间
    // SAMPLE_PRT(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>        mode: %d \n", receiveData->mode);
    if(receiveData->mode == 4){         // 娱乐模式
        gettimeofday(&end_time, NULL);                  // 获取当前时刻
        // 距离开始时间的ms数
        diff = (end_time.tv_sec * 1000 + end_time.tv_usec/1000) - (start_time.tv_sec * 1000 + start_time.tv_usec/1000);
        // SAMPLE_PRT(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>        time: %d \n", diff);
        if(diff / 1000  > 243){         // 对于模式4来说 超过预定的时间 4min 2s
            return HI_SUCCESS;          // 返回
        }
        // SAMPLE_PRT(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>         %d        \n", diff);
    }

    // 模式1 动作矫正  模式2 体验模式  模式3 无限模式
    if((receiveData->mode == 2) && g_count >=10 ){
        return HI_SUCCESS;
    }

    // 图像转换成bgr
    ret = FrmToRgbImg((VIDEO_FRAME_INFO_S*)srcFrm, &img);
    ret = ImgRgbToBgr(&img);    
    
    // resize后宽 768 高 432, 为了提高准确率, 提取中间416x416大小的图像送入网络
    POSE_ROI.xmin = 384 - 208;  // 176
    POSE_ROI.ymin = 216 - 208;  // 8 
    POSE_ROI.xmax = 384 + 208;  // 592
    POSE_ROI.ymax = 216 + 208;  // 424

    ret = ImgU3Crop(&img, &imgIn, &POSE_ROI);

    // 进行姿态检测, 识别关键点
    objNum = PoseDetectCal(&imgIn, poseobjs);

    if(objNum > 0){ // 目标数大于0, 识别到人
        int i = 0;
        // 将识别到的关键点重新映射到原始图像尺寸上, 便于绘制
        RectBoxTran(&poseobjs[i].box, POSE_FRM_WIDTH, POSE_FRM_HEIGHT, dstFrm->stVFrame.u32Width, dstFrm->stVFrame.u32Height);
        KeypointsTran(&poseobjs[i].keypoints, POSE_FRM_WIDTH, POSE_FRM_HEIGHT, dstFrm->stVFrame.u32Width, dstFrm->stVFrame.u32Height);
        MppFrmDrawRects(dstFrm, &poseobjs[i].box, 1, RGB888_GREEN, 6);
        MppFrmDrawKeypoints(dstFrm, &poseobjs[i].keypoints, RGB888_YELLOW, 6); 

        // 左小臂
        float left_arm_small = calculate_angle_with_cos(&poseobjs[i], left_arm_small_index) ;
        // 左大臂
        float left_arm_big = calculate_angle_with_cos(&poseobjs[i], left_arm_big_index);
        // 右小臂
        float right_arm_small = calculate_angle_with_cos(&poseobjs[i], right_arm_small_index);
        // 右大臂
        float right_arm_big = calculate_angle_with_cos(&poseobjs[i], right_arm_big_index);
        // 左小腿
        float left_leg_small = calculate_angle_with_cos(&poseobjs[i], left_leg_small_index);
        // 左大腿
        float left_leg_big = calculate_angle_with_cos(&poseobjs[i], left_leg_big_index);
        // 右小腿
        float right_leg_small = calculate_angle_with_cos(&poseobjs[i], right_leg_small_index);
        // 右大腿
        float right_leg_big = calculate_angle_with_cos(&poseobjs[i], right_leg_big_index);

        // 调试, 打印角度信息
        // SAMPLE_PRT("angle {%f, %f, %f, %f, %f, %f, %f, %f}\n", left_arm_small, left_arm_big, right_arm_small, right_arm_big,
        //                                                        left_leg_small, left_leg_big, right_leg_small, right_leg_big);

        // 坚持模式
        if(receiveData->mode == 1){
            int k = receiveData->action1;   // 用户选择的动作
            
            // k = 1 平板支撑
            // k = 2 俯卧撑
            // k = 3 四点支撑
            // k = 4 反向卷腹
            // k = 5 臀桥

            // 这些语音参考了kepp等专业语音提示
            // 判断标准咨询了体育教师（todo）
            if( k == 1 ){   // 平板支撑
                if(first_audio){
                    // pthread_create(&audio_thread, NULL, thread_function, (void*)args);
                    // 阻塞播放语音11s
                    AudioTest(33, 11);                      // 播放 平板支撑 提示语音 11s
                    first_audio = false;
                }
                if( right_leg_big < 160 || left_leg_big < 160 ){
                    PlayAudio(46, 3);           // playAudio 检测到一定次数之后才会播报 
                    // args[0]=46;
                    // args[1]=3;
                    // pthread_create(&audio_thread, NULL, thread_function2, (void*)args);}
                    // SAMPLE_PRT(" please waist 180 \n");        // 播放 请不要塌腰或提臀，保持身体一条直线
                }
                else if (left_leg_small < 160 || right_leg_small < 160){
                    PlayAudio(47, 2);
                    // args[0]=47;
                    // args[1]=2;
                    // pthread_create(&audio_thread, NULL, thread_function2, (void*)args);}
                    // SAMPLE_PRT(" please leg 180 \n");     // 播放 双腿肌肉收紧
                }
                else if( left_arm_big < 60 || left_arm_big > 100 || right_arm_big < 60 || right_arm_big > 100 ) {
                    // args[0]=48;
                    // args[1]=3;
                    // pthread_create(&audio_thread, NULL, thread_function2, (void*)args);}
                    PlayAudio(48, 3);
                    // SAMPLE_PRT("left big arm  \n");          // 播放 注意小臂互相平行 保持大臂垂直地面
                }
                else
                {
                    // SAMPLE_PRT("begin time \n");            // 播放 鼓励语音1,2 随机
                    // args[0]=31;
                    // args[1]=6;
                    // SAMPLE_PRT("----------");
                    // pthread_create(&audio_thread, NULL, thread_function, (void*)args);
                    PlayAudio(31, 6);                           // 随机播放鼓励语音 6s钟
                }
            }
            // if( k == 2 ){   // 俯卧撑
            //     if(first_audio){
            //         AudioTest(34, 10);                      // 播放 俯卧撑 提示语音 10s
            //         first_audio = false;
            //     }
            //     if( right_leg_big < 165 || left_leg_big < 165 )
            //         PlayAudio(49, 3);
            //         // SAMPLE_PRT("waist straight \n");        // 播放 注意不要塌腰，身体成一条直线
            //     else if (left_leg_small < 165 || right_leg_small < 165)
            //         PlayAudio(50, 2);
            //         // SAMPLE_PRT("left leg straight \n");     // 播放 注意腿部不要弯曲
            //     else{
            //         PlayAudio(30, 4);
            //         // SAMPLE_PRT("begin time \n");            // 播放 鼓励语音1,2 随机
            //     }
            // }

            // 四点支撑 改成 V字支撑
            if( k == 3 ){
                if(first_audio){
                    AudioTest(57, 12);                       // 阻塞播放 四点支撑 提示语音 8s
                    first_audio = false;
                }
                if((left_leg_small < 50) && (right_leg_small < 50)){
                    // 进行对应的语音提示
                    PlayAudio(58, 1);
                }
                else if((left_leg_big > 90) && (right_leg_big > 90)){
                    // 进行对应的语音提示
                    PlayAudio(58, 1);
                }
                else{
                    // args[0]=31;
                    // args[1]=6;
                    // pthread_create(&audio_thread, NULL, thread_function2, (void*)args);
                    PlayAudio(31, 6);                           // 鼓励
                    // SAMPLE_PRT("begin time \n");
                }
            }

            if( k == 4 ){   // 侧平板支撑
                if(first_audio){
                    // args[0]=36;
                    // args[1]=10;
                    // pthread_create(&audio_thread, NULL, thread_function2, (void*)args);
                    AudioTest(60, 7);                      // 阻塞 播放 反向卷腹 提示语音 10s
                    first_audio = false;
                }
                // if( right_leg_small < 165 || left_leg_small < 165 ){
                    // args[0] = 54;
                    // args[1] = 2;
                    // pthread_create(&audio_thread, NULL, thread_function2, (void*)args);
                    // PlayAudio(54, 2);
                    // SAMPLE_PRT("waist straight \n");        // 播放 请保持小腿向上伸直
                // }
                // else if (left_leg_small < 75 || right_leg_small < 75 || left_leg_small > 100 || right_leg_small > 100){
                    // args[0]=55;
                    // args[1]=3;
                    // pthread_create(&audio_thread, NULL, thread_function2, (void*)args);
                    // PlayAudio(55, 3);
                    // SAMPLE_PRT("left leg straight \n");     // 播放 请保持大腿与身体成90度
                // }
                if( (right_leg_big < 155) && (left_leg_big < 155)){ // 请保持身体一条直线
                    PlayAudio(61, 2);
                }
                // else if((right_arm_small < 150) && (left_arm_small < 150)){
                    // 提示 手臂伸直
                // }
                else{
                    // args[0]=30;
                    // args[1]=4;
                    // pthread_create(&audio_thread, NULL, thread_function2, (void*)args);
                    PlayAudio(30, 4);
                    // SAMPLE_PRT("begin time \n");
                }
            }
            if( k == 5 ){   // 反向支撑
                if(first_audio){                            
                    AudioTest(62, 13);                      // 播放 臀桥 提示语音 12s
                    first_audio = false;                    // 阻塞播放语音提示
                }
                // if( right_leg_big < 165 || left_leg_big < 165 ){
                    // args[0]=56;
                    // args[1]=2;
                    // pthread_create(&audio_thread, NULL, thread_function2, (void*)args);
                    // PlayAudio(56, 2);
                    // SAMPLE_PRT("waist straight \n");        // 播放: 请保持身体成一条直线
                // }
                if( (right_leg_big < 155) && (left_leg_big < 155)){     // 提示身体更直一点
                    PlayAudio(64, 2);
                }
                else if((right_leg_small < 155) && (left_leg_small < 155)){
                    // 语音提示 手臂伸直
                    PlayAudio(63, 2);
                }
                else{
                    PlayAudio(31, 6);
                    // args[0]=31;
                    // args[1]=6;
                    // pthread_create(&audio_thread, NULL, thread_function2, (void*)args);
                    // SAMPLE_PRT("begin time \n");
                }
            }
        }


        // 体验模式，从1 报数到10
        else if(receiveData->mode == 2){
            // 蹲起一直测的都没啥问题

            if(receiveData->action1 == 1){                             // 蹲起动作
                if((left_leg_small < 95) && (right_leg_small < 95)){   // 蹲下
                    action_flag = true;
                }
                if((left_leg_small > 145) && (right_leg_small > 145)){  // 站立
                    if(action_flag){
                        g_count++;                                      
                        if(g_count < 10){
                            args[0] = 16 + g_count;
                            args[1] = 1;
                            // 语音播报线程
                            pthread_create(&audio_thread, NULL, thread_function, (void*)args);  // 独立线程播放
                            // AudioTest(16 + g_count, 1);                 // 播报 1 到 9
                        }else if(g_count == 10){
                            AudioTest(26, 1);                           // 播报 10  结束阻塞播报
                            AudioTest(32, 5);                           // 同时播报鼓励语句
                        }                       
                    }                
                    action_flag = false;
                }
            }
            
            // 开合跳现在频率也能比较高
            else if(receiveData->action1 == 2){                          // 开合跳
                if((left_arm_big < 55) && (right_arm_big < 55))        // 抬起双臂
                    action_flag = true;
                if((left_arm_big > 100) && (right_arm_big > 100)){         // 落下双臂
                    if(action_flag){
                        g_count++;                                      
                        if(g_count < 10){
                            args[0] = 16 + g_count;
                            args[1] = 1;
                            // 语音播报线程
                            pthread_create(&audio_thread, NULL, thread_function, (void*)args);  // 独立线程
                            // AudioTest(16 + g_count, 1);                 // 播报 1 到 9
                        }else if(g_count == 10){
                            AudioTest(26, 1);                           // 播报 10
                            AudioTest(32, 5);                           // 同时播报鼓励语句
                        }                       
                    }                
                    action_flag = false;
                }
            }
            else if(receiveData->action1 == 3){                         // 仰卧起坐
                if((left_leg_big > 95 ) && (right_leg_big > 95))         // 起来
                    action_flag = true;
                if((left_leg_big < 60 ) && (right_leg_big < 60)){        // 躺下
                    if(action_flag){
                        g_count++;                                      
                        if(g_count < 10){
                            args[0] = 16 + g_count;
                            args[1] = 1;
                            pthread_create(&audio_thread, NULL, thread_function, (void*)args);
                            // AudioTest(16 + g_count, 1);                 // 播报 1 到 9
                        }else if(g_count == 10){
                            AudioTest(26, 1);                           // 播报 10
                            AudioTest(32, 5);                           // 同时播报鼓励语句
                        }                       
                    }                
                    action_flag = false;
                }
            }else if(receiveData->action1 == 4){                             // 俯卧撑,  改成波比跳
                // if((left_arm_small < 125) && (right_arm_small < 125))        // 下去
                HI_FLOAT height = poseobjs[i].box.ymax - poseobjs[i].box.ymin;
                HI_FLOAT width = poseobjs[i].box.xmax - poseobjs[i].box.xmin;

                SAMPLE_PRT(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>       ratio: %.2f\n", height / width);
                if(first_audio){                    // 过滤初始站立状态          
                    if(width > 1.7 * height)
                        first_audio = false;
                }
                else{
                    if(width > 1.7 * height)
                        action_flag = true;
                    if(height > 2.5 * width){       // 起来
                        if(action_flag){
                            g_count++;                                      
                            if(g_count < 10){
                                args[0] = 16 + g_count;
                                args[1] = 1;
                                pthread_create(&audio_thread, NULL, thread_function, (void*)args);
                                // AudioTest(16 + g_count, 1);                 // 播报 1 到 9
                            }else if(g_count == 10){
                                AudioTest(26, 1);                               // 播报 10
                                AudioTest(32, 5);                               // 同时播报鼓励语句
                            }                     
                        }                
                        action_flag = false;
                    }   
                
                }
            }
            // 调试是否正常
            // 串口号绑死的 30, 就先写死吧
            uartSendData->score = g_count;
            uartSendData->startflag = '!';
            UartSend(uartFd_pose, uartSendData, sizeof(SendData));
            // SAMPLE_PRT(" >>>>>>>>>>>>>>  uart fd  send  %d", uartFd_pose);
            SAMPLE_PRT(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>        num: %d\n", uartSendData->score);
        } 
        // 无尽模式
        else if(receiveData->mode == 3){
            if(receiveData->action1 == 1){                             // 蹲起动作
                // 蹲起, 主要根据腿部数据判断
                if((left_leg_small < 95) && (right_leg_small < 95)){   // 蹲下
                    action_flag = true;
                }
                if((left_leg_small > 145) && (right_leg_small > 145)){  // 站立
                    if(action_flag){
                        g_count++;
                        if(g_count == 20){
                            args[0] = 38;
                            args[1] = 4;
                            pthread_create(&audio_thread, NULL, thread_function, (void*)args);
                            // AudioTest(38, 4);                           // 提醒用户 完成了20次动作
                        }else if(g_count == 50){
                            args[0] = 39;
                            args[1] = 4;
                            pthread_create(&audio_thread, NULL, thread_function, (void*)args);
                            // AudioTest(39, 4);                           // 提醒用户 完成了50次动作
                        }                       
                        else if(g_count == 100){
                            args[0] = 40;
                            args[1] = 4;
                            pthread_create(&audio_thread, NULL, thread_function, (void*)args);
                            // AudioTest(40, 4);                           // 提醒用户 完成了100次动作
                        }
                    }                
                    action_flag = false;
                }
            }
            else if(receiveData->action1 == 2){                          // 开合跳
                if((left_arm_big < 55) && (right_arm_big < 55))        // 张开双臂
                    action_flag = true;
                if((left_arm_big > 100) && (right_arm_big > 100)){         // 落下双臂
                    if(action_flag){
                        g_count++;                                      
                        if(g_count == 20){
                            args[0] = 38;
                            args[1] = 4;
                            pthread_create(&audio_thread, NULL, thread_function, (void*)args);
                            // AudioTest(38, 4);                           // 提醒用户 完成了20次动作
                        }else if(g_count == 50){
                            args[0] = 39;
                            args[1] = 4;
                            pthread_create(&audio_thread, NULL, thread_function, (void*)args);
                            // AudioTest(39, 4);                           // 提醒用户 完成了50次动作
                        }                       
                        else if(g_count == 100){
                            args[0] = 40;
                            args[1] = 4;
                            pthread_create(&audio_thread, NULL, thread_function, (void*)args);
                            // AudioTest(40, 4);                           // 提醒用户 完成了100次动作
                        }                       
                    }                
                    action_flag = false;
                }
            }
            else if(receiveData->action1 == 3){                         // 仰卧起坐
                if((left_leg_big > 95) && (right_leg_big > 95))         // 坐起
                    action_flag = true;
                if((left_leg_big < 60) && (right_leg_big < 60)){      // 躺下
                    if(action_flag){
                        g_count++;                                      
                        if(g_count == 20){
                            args[0] = 38;
                            args[1] = 4;
                            pthread_create(&audio_thread, NULL, thread_function, (void*)args);
                            // AudioTest(38, 4);                           // 提醒用户 完成了20次动作
                        }else if(g_count == 50){
                            args[0] = 39;
                            args[1] = 4;
                            pthread_create(&audio_thread, NULL, thread_function, (void*)args);
                            // AudioTest(39, 4);                           // 提醒用户 完成了50次动作
                        }                       
                        else if(g_count == 100){
                            args[0] = 40;
                            args[1] = 4;
                            pthread_create(&audio_thread, NULL, thread_function, (void*)args);
                            // AudioTest(40, 4);                           // 提醒用户 完成了100次动作
                        }                       
                    }                
                    action_flag = false;
                }
            }else if(receiveData->action1 == 4){                        //俯卧撑
                // if((left_arm_small < 125) && (right_arm_small < 125))        // 下去
                HI_FLOAT height = poseobjs[i].box.ymax - poseobjs[i].box.ymin;
                HI_FLOAT width = poseobjs[i].box.xmax - poseobjs[i].box.xmin;

                // SAMPLE_PRT(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>       ratio: %.2f\n", height / width);
                if(first_audio){                    // 过滤初始站立状态          
                    if(width > 1.7 * height)
                        first_audio = false;
                }
                else{
                    if(width > 1.7 * height)
                        action_flag = true;
                    if(height > 2.5 * width){       // 起来
                        if(action_flag){
                            g_count++;                                      
                            if(g_count == 20){
                                args[0] = 38;
                                args[1] = 4;
                                pthread_create(&audio_thread, NULL, thread_function, (void*)args);
                                // AudioTest(38, 4);                           // 提醒用户 完成了20次动作
                            }else if(g_count == 50){
                                args[0] = 39;
                                args[1] = 4;
                                pthread_create(&audio_thread, NULL, thread_function, (void*)args);
                                // AudioTest(39, 4);                           // 提醒用户 完成了50次动作
                            }                       
                            else if(g_count == 100){
                                args[0] = 40;
                                args[1] = 4;
                                pthread_create(&audio_thread, NULL, thread_function, (void*)args);
                                // AudioTest(40, 4);                           // 提醒用户 完成了100次动作
                            }                       
                        }                
                        action_flag = false;
                    }
                }
            }
            // 返回的 Score 为完成的动作的计数
            uartSendData->score = g_count;
            uartSendData->startflag = '!';
            UartSend(uartFd_pose, uartSendData, sizeof(SendData));
            // SAMPLE_PRT(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>        num: %d\n", uartSendData->score);
        }else if(receiveData->mode == 4){    // 娱乐模式
        //  这个模式比较丑陋
            // gettimeofday(&end_time,NULL);
            // 距离开始时间的ms数
            // int diff = (end_time.tv_sec * 1000 + end_time.tv_usec/1000) - (start_time.tv_sec * 1000 + start_time.tv_usec/1000);
            // if(diff / 1000  > 243){         // 超过预定的时间
                // return HI_SUCCESS;          // 返回
            // }

            // 动作对应着   0-242, 共 243 帧图像

            // 测得的帧率大概在4左右，一次大概250ms左右，按照300算
            // 实际对应的时间左右

            // SAMPLE_PRT("       >>>>>>>>>>>>>>>>>>>                    \n ");

            // SAMPLE_PRT("       >>>>>>>>>>>>>>>>>>>           %d, %d  \n", diff % 1000, diff % 1000);

            // if( (diff % 1000) > 850 && (diff % 1000) < 150 ) {  // 为了尽量保证时间和秒数的同步
            // 获取对应的 秒数 
            // 幂次图像推理的时间不一致    
                ///////////////////////////////// 保证同一秒只会用到一次
                // if( mode4_time_cnt == diff / 1000 ) {   // 避免内存泄漏, 很重要, 不然会在运行过程中程序崩掉, 获取当前的秒数
                    // IveImgDestroy(&imgIn);
                    // IveImgDestroy(&img);
                    // return HI_SUCCESS;              // 简化，直接返回
                // }
                mode4_time_cnt = diff / 1000;       // 获取对应的秒数
                ///////////////////////////////////////////////////////

                // 时间超过4min，然后就停止了
                if( mode4_time_cnt > 242){            // 直接返回，避免访问数组越界
                    // uartSendData->startflag = '!';
                    // uartSendData->score = cur_score;  // 得分除以5， 得分在100以内
                    // ret = UartSend(uartFd, (unsigned char*)gestureName, strlen(gestureName));
                    UartSend(uartFd_pose, uartSendData, sizeof(SendData));       // 超过4min，就一直发送原来的数据
                    IveImgDestroy(&imgIn);
                    IveImgDestroy(&img);
                    return HI_SUCCESS;
                }
                
                // SAMPLE_PRT("          >>>     %d,    %d                 \n ", diff % 1000, mode4_time_cnt);

                // 为了提高一下体验感，将镜像左右
                // 该模式为娱乐模式
                // 受限于板子端的推理速度精度以及计时的精度等原因, 在判定标准上进行了适当的放宽

                bool condition[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

                // SAMPLE_PRT("real %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f \n", right_arm_small, right_arm_big, left_arm_small, left_arm_big, right_leg_small, right_leg_big, left_leg_small, left_leg_big);
                // SAMPLE_PRT("std  %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f \n", std_angle[mode4_time_cnt*3][0],std_angle[mode4_time_cnt*3][1],std_angle[mode4_time_cnt*3][2],std_angle[mode4_time_cnt*3][3],std_angle[mode4_time_cnt*3][4],std_angle[mode4_time_cnt*3][5],std_angle[mode4_time_cnt*3][6],std_angle[mode4_time_cnt*3][7]);
                
                // 3s内的9帧
                for(int error_cnt = - 4 ; error_cnt < 5 ; error_cnt ++){
                    if(abs(right_arm_small - std_angle[(mode4_time_cnt*3+error_cnt)%727][0]) > 30){
                    // SAMPLE_PRT("left arm small \n");
                        condition[error_cnt+4] = false;
                    }
                    else if(abs(right_arm_big - std_angle[(mode4_time_cnt*3+error_cnt)%727][1]) > 30){
                    // SAMPLE_PRT("left arm big \n");
                        condition[error_cnt+4] = false;
                    }
                    else if(abs(left_arm_small - std_angle[(mode4_time_cnt*3+error_cnt)%727][2]) > 30){
                    // SAMPLE_PRT("right arm small \n");
                        condition[error_cnt+4] = false;
                    }
                    else if(abs(left_arm_big - std_angle[(mode4_time_cnt*3+error_cnt)%727][3]) > 30){
                    // SAMPLE_PRT("right arm big \n");
                        condition[error_cnt+4] = false;
                    }
                    else if(abs(right_leg_small - std_angle[(mode4_time_cnt*3+error_cnt)%727][4]) > 30){
                    // SAMPLE_PRT("left leg small \n");
                        condition[error_cnt+4] = false;
                    }
                    else if(abs(right_leg_big - std_angle[(mode4_time_cnt*3+error_cnt)%727][5]) > 30){
                    // SAMPLE_PRT("left leg big \n");
                        condition[error_cnt+4] = false;
                    }
                    else if(abs(left_leg_small - std_angle[(mode4_time_cnt*3+error_cnt)%727][6]) > 30){
                    // SAMPLE_PRT("right leg small \n");
                        condition[error_cnt+4] = false;
                    }
                    else if(abs(left_leg_big - std_angle[(mode4_time_cnt*3+error_cnt)%727][7]) > 30){
                    // SAMPLE_PRT("right leg big \n");
                        condition[error_cnt+4] = false;
                    }
                }

                // 和三帧9张里面的有类似的就认为正确，得分加1
                if(condition[0] | condition[1] | condition[2] | condition[3] | condition[4] | condition[5] | condition[6] | condition[7] | condition[8] ){
                    mode4_score += 1;   // 符合条件, 那么得分加1
                }
                
                SAMPLE_PRT("                                %d ms \n ", diff);
                // 返回的 Score 为娱乐模式的得分
                // 4fps, 720f左右 240 x 4 

                int cur_score = mode4_score / 3;

                // 压分
                // if(cur_score < 20){
                //     cur_score = cur_score / 2 ;
                // }   // 
                // else if(cur_score < 60){
                //     cur_score = (cur_score - 20) * 0.9 + 20;
                // }
                // else if(cur_score < 85){
                //     cur_score = (cur_score - 60) * 0.8 + 60;
                // }
                // else{
                //     cur_score = min(max(cur_score * 0.7, 85), 99);
                // }
                // if(cur_score >= 30)  
                    // cur_score = min(max(cur_score * 0.5, 30), 99);

                cur_score *= 0.7;
                if(cur_score > 95)
                    cur_score = 96 ;


                // 返回最终的得分
                uartSendData->startflag = '!';
                uartSendData->score = cur_score;  // 得分除以5， 得分在100以内
                // ret = UartSend(uartFd, (unsigned char*)gestureName, strlen(gestureName));
                UartSend(uartFd_pose, uartSendData, sizeof(SendData));
                // SAMPLE_PRT(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>        fd   : %d\n", );
                SAMPLE_PRT("                                    score: %d\n", uartSendData->score);
            // }
        }
    }
    else    // 当前没有检测到目标, 也就是目标的
        SAMPLE_PRT("                                           no person   \n");

    // 测试通信
    // UartSend(uartFd_pose, uartSendData, sizeof(SendData));
    
    // WARNING: 释放图片申请的内存, 否则会内存溢出
    IveImgDestroy(&imgIn);
    IveImgDestroy(&img);

    return ret;
}




#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */
