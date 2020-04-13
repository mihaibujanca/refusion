/*
 *
 *
 * This benchmark file can run any SLAM algorithm compatible with the SLAMBench API
 * We recommend you to generate a library which is compatible with this application.
 * 
 * The interface works that way :
 *   - First benchmark.cpp will call an initialisation function of the SLAM algorithm. (void sb_init())
 *     This function provides the compatible interface with the SLAM algorithm.
 *   - Second benchmark.cpp load an interface (the textual interface in our case)
 *   - Then for every frame, the benchmark.cpp will call sb_process()
 *
 *
 */

#include <Eigen/Core>
#include <sstream>
#include <opencv2/imgproc/imgproc.hpp>
#include "tracker/tracker.h"
#include <SLAMBenchAPI.h>

#include <io/sensor/CameraSensor.h>
#include <io/sensor/CameraSensorFinder.h>
#include <io/sensor/DepthSensor.h>
#include <io/SLAMFrame.h>




// ===========================================================
// Default Parameters
// ===========================================================

// Options for the TSDF representation
static const float
        default_voxel_size = 0.01,
        default_truncation_distance = 0.1,
        default_max_sensor_depth = 5,
        default_min_sensor_depth = 0.1;

static const int
        default_num_buckets = 50000,
        default_bucket_size = 10,
        default_num_blocks = 500000,
        default_block_size = 8,
        default_max_sdf_weight = 64,


// Options for the tracker
        default_max_iterations_per_level_0 = 6,
        default_max_iterations_per_level_1 = 3,
        default_max_iterations_per_level_2 = 2,
        default_downsample_0 = 4,
        default_downsample_1 = 2,
        default_downsample_2 = 1;

static const float
        default_min_increment = 0.0001,
        default_regularization = 0.002,
        default_huber_constant = 0.02,
        default_depth_factor = 5000;

// ===========================================================
// Algorithm parameters
// ===========================================================

float voxel_size,
        truncation_distance,
        max_sensor_depth,
        min_sensor_depth;

int num_buckets,
        bucket_size,
        num_blocks,
        block_size,
        max_sdf_weight,
        max_iterations_per_level_0,
        max_iterations_per_level_1,
        max_iterations_per_level_2,
        downsample_0,
        downsample_1,
        downsample_2;

float min_increment,
      regularization,
      huber_constant,
      depth_factor;

static refusion::Tracker *tracker;
static refusion::RgbdSensor *sensor;
static cv::Mat *imRGB, *imD, *virtual_rgb;

static sb_uint2 inputSize;


// ===========================================================
// SLAMBench Sensors
// ===========================================================

static slambench::io::DepthSensor *depth_sensor;
static slambench::io::CameraSensor *rgb_sensor;


// ===========================================================
// SLAMBench Outputs
// ===========================================================

slambench::outputs::Output *pose_output;
slambench::outputs::Output *pointcloud_output;
slambench::outputs::Output *depth_frame_output;
slambench::outputs::Output *rgb_frame_output;
slambench::outputs::Output *render_frame_output;



bool sb_new_slam_configuration(SLAMBenchLibraryHelper * slam_settings)  {

    slam_settings->addParameter(TypedParameter<float>("", "voxelSize",     "Voxel size in meters",      &voxel_size, &default_voxel_size));
    slam_settings->addParameter(TypedParameter<float>("td", "truncationDistance",          "Voxel Truncation Distance",           &truncation_distance,      &default_truncation_distance));
    slam_settings->addParameter(TypedParameter<float>("maxd", "maxDepth",            "Maximum sensor depth that is considered",             &max_sensor_depth, &default_max_sensor_depth));
    slam_settings->addParameter(TypedParameter<float>("mind", "minDepth",   " Minimum sensor depth that is considered",    &min_sensor_depth, &default_min_sensor_depth));
    slam_settings->addParameter(TypedParameter<int>("nbuck", "numBuckets",      "Total number of buckets in the table",       &num_buckets, &default_num_buckets));
    slam_settings->addParameter(TypedParameter<int>("bucks", "bucketSize",    "Maximum number of entries in a bucket",     &bucket_size, &default_bucket_size));
    slam_settings->addParameter(TypedParameter<int>("nbl", "numBlocks",     "Maximum number of blocks that can be allocated ",      &num_blocks, &default_num_blocks));
    slam_settings->addParameter(TypedParameter<int>  ("bsize", "blockSize", "Size in voxels of the side of a voxel block",  &block_size, &default_block_size));
    slam_settings->addParameter(TypedParameter<int>  ("sdfw", "maxSDFWeight",      "Maximum weight that a voxel can have",       &max_sdf_weight,      &default_max_sdf_weight));

    slam_settings->addParameter(TypedParameter<int>  ("it0", "maxIt0",      "Maximum number of iteration per subsampling level 0",       &max_iterations_per_level_0,      &default_max_iterations_per_level_0));
    slam_settings->addParameter(TypedParameter<int>  ("it1", "maxIt1",      "Maximum number of iteration per subsampling level 1",       &max_iterations_per_level_1,      &default_max_iterations_per_level_1));
    slam_settings->addParameter(TypedParameter<int>("it2", "maxIt2",        "Maximum number of iteration per subsampling level 2",        &max_iterations_per_level_2       , &default_max_iterations_per_level_2       ));

    slam_settings->addParameter(TypedParameter<int>("ds0", "",           "Downsampling level 0",           &downsample_0          , &default_downsample_0          ));
    slam_settings->addParameter(TypedParameter<int>("ds1", "",        "Downsampling level 1",        &downsample_1       , &default_downsample_1       ));
    slam_settings->addParameter(TypedParameter<int>("ds2", "",             "Downsampling level 2",             &downsample_2            , &default_downsample_2            ));

    slam_settings->addParameter(TypedParameter<float>("minincr", "minIncrement", "Minimum norm of the increment to terminate the least-squares algorithm", &min_increment, &default_min_increment));
    slam_settings->addParameter(TypedParameter<float>("reg", "regularisation", "Initial regularization term of Levenberg-Marquardt", &regularization, &default_regularization));
    slam_settings->addParameter(TypedParameter<float>("hub", "huber", "Constant used by the Huber estimator", &huber_constant, &default_huber_constant));
    slam_settings->addParameter(TypedParameter<float>("depth", "depthFactor", "Scale factor to convert the depth in millimeters", &depth_factor, &default_depth_factor));

    return true;
}

bool sb_init_slam_system(SLAMBenchLibraryHelper * slam_settings) {


    slambench::io::CameraSensorFinder sensor_finder;
    rgb_sensor = sensor_finder.FindOne(slam_settings->get_sensors(), {{"camera_type", "rgb"}});
    depth_sensor = (slambench::io::DepthSensor*)sensor_finder.FindOne(slam_settings->get_sensors(), {{"camera_type", "depth"}});

    if ((rgb_sensor == nullptr) || (depth_sensor == nullptr)) {
        std::cerr << "Invalid sensors found, RGB or Depth not found." << std::endl;
        return false;
    }

    if(rgb_sensor->FrameFormat != slambench::io::frameformat::Raster) {
        std::cerr << "RGB data is in wrong format" << std::endl;
        return false;
    }
    if(depth_sensor->FrameFormat != slambench::io::frameformat::Raster) {
        std::cerr << "Depth data is in wrong format" << std::endl;
        return false;
    }
    if(rgb_sensor->PixelFormat != slambench::io::pixelformat::RGB_III_888) {
        std::cerr << "RGB data is in wrong format pixel" << std::endl;
        return false;
    }
    if(depth_sensor->PixelFormat != slambench::io::pixelformat::D_I_16) {
        std::cerr << "Depth data is in wrong pixel format" << std::endl;
        return false;
    }

    assert(depth_sensor->Width == rgb_sensor->Width);
    assert(depth_sensor->Height == rgb_sensor->Height);

    inputSize = make_sb_uint2(rgb_sensor->Width, rgb_sensor->Height);

    refusion::tsdfvh::TsdfVolumeOptions tsdf_options;
    tsdf_options.voxel_size = voxel_size;
    tsdf_options.num_buckets = num_buckets;
    tsdf_options.bucket_size = bucket_size;
    tsdf_options.num_blocks = num_blocks;
    tsdf_options.block_size = block_size;
    tsdf_options.max_sdf_weight = max_sdf_weight;
    tsdf_options.truncation_distance = truncation_distance;
    tsdf_options.max_sensor_depth = max_sensor_depth;
    tsdf_options.min_sensor_depth = min_sensor_depth;

    refusion::TrackerOptions tracker_options;
    tracker_options.max_iterations_per_level[0] = max_iterations_per_level_0;
    tracker_options.max_iterations_per_level[1] = max_iterations_per_level_1;
    tracker_options.max_iterations_per_level[2] = max_iterations_per_level_2;
    tracker_options.downsample[0] = downsample_0;
    tracker_options.downsample[1] = downsample_1;
    tracker_options.downsample[2] = downsample_2;
    tracker_options.min_increment = min_increment;
    tracker_options.regularization = regularization;
    tracker_options.huber_constant = huber_constant;

    sensor = new refusion::RgbdSensor();
    sensor->fx = rgb_sensor->Intrinsics[0] * rgb_sensor->Width;
    sensor->fy = rgb_sensor->Intrinsics[1] * rgb_sensor->Height;
    sensor->cx = rgb_sensor->Intrinsics[2] * rgb_sensor->Width;
    sensor->cy = rgb_sensor->Intrinsics[2] * rgb_sensor->Height;
    sensor->rows = rgb_sensor->Height;
    sensor->cols = rgb_sensor->Width;
    sensor->depth_factor = depth_factor;

    // fx, fy, cx, cy
    std::cout << "Intrisics are fx:" << sensor->fx
                           << " fy:" << sensor->fy
                           << " cx:" << sensor->cx
                           << " cy:" << sensor->cy << std::endl;

    tracker = new refusion::Tracker(tsdf_options, tracker_options, *sensor);

    imRGB = new cv::Mat (rgb_sensor->Height, rgb_sensor->Width, CV_8UC3);
    virtual_rgb = new cv::Mat (rgb_sensor->Height, rgb_sensor->Width, CV_8UC3);
    imD   = new cv::Mat (depth_sensor->Height, depth_sensor->Width, CV_16UC1);

    pose_output = new slambench::outputs::Output("Pose ReFusion", slambench::values::VT_POSE, true);
    slam_settings->GetOutputManager().RegisterOutput(pose_output);

    pointcloud_output = new slambench::outputs::Output("PointCloud ReFusion", slambench::values::VT_COLOUREDPOINTCLOUD, true);
    pointcloud_output->SetKeepOnlyMostRecent(true);
    slam_settings->GetOutputManager().RegisterOutput(pointcloud_output);

    depth_frame_output = new slambench::outputs::Output("Depth Frame ReFusion", slambench::values::VT_FRAME);
    depth_frame_output->SetKeepOnlyMostRecent(true);
    slam_settings->GetOutputManager().RegisterOutput(depth_frame_output);

    render_frame_output = new slambench::outputs::Output("Rendered frame ReFusion", slambench::values::VT_FRAME);
    render_frame_output->SetKeepOnlyMostRecent(true);
    slam_settings->GetOutputManager().RegisterOutput(render_frame_output);

    rgb_frame_output = new slambench::outputs::Output("RGB Frame ReFusion", slambench::values::VT_FRAME);
    rgb_frame_output->SetKeepOnlyMostRecent(true);
    slam_settings->GetOutputManager().RegisterOutput(rgb_frame_output);

    return true;
}


bool depth_ready = false;
bool rgb_ready   = false;

bool sb_update_frame (SLAMBenchLibraryHelper * slam_settings, slambench::io::SLAMFrame* s) {

    if (depth_ready and rgb_ready) {
        depth_ready = false;
        rgb_ready   = false;
    }
    assert(s != nullptr);
    if(s->FrameSensor == depth_sensor and imD) {
        memcpy(imD->data, s->GetData(), s->GetSize());
        depth_ready = true;
        s->FreeData();
    } else if(s->FrameSensor == rgb_sensor and imRGB) {
        memcpy(imRGB->data, s->GetData(), s->GetSize());
        rgb_ready = true;
        s->FreeData();
    }

    return depth_ready and rgb_ready;
}

bool sb_process_once (SLAMBenchLibraryHelper * slam_settings)  {
    static int frame = 0;
    cv::Mat convertedD( rgb_sensor->Height ,  rgb_sensor->Width, CV_32FC1);
    imD->convertTo(convertedD, CV_32FC1, 1.0f / depth_factor);
    tracker->AddScan(*imRGB, convertedD);
    frame++;
    return true;
}


bool sb_clean_slam_system() {
    delete tracker;
    delete imD;
    delete imRGB;
    delete virtual_rgb;
    delete sensor;
    delete rgb_sensor;
    delete depth_sensor;

    delete pose_output;
    delete render_frame_output;
    delete rgb_frame_output;
    delete depth_frame_output;
    delete pointcloud_output;
    return true;
}



bool sb_update_outputs(SLAMBenchLibraryHelper *lib, const slambench::TimeStamp *ts_p) {
    slambench::TimeStamp ts = *ts_p;

    if(pose_output->IsActive()) {
        // Get the current pose as an eigen matrix
        Eigen::Matrix4f mat = tracker->GetCurrentPose().cast<float>();

        std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
        pose_output->AddPoint(ts, new slambench::values::PoseValue(mat));
    }

    if(pointcloud_output->IsActive() && false) {
//        TODO: make these default values and add to algorithm parameters
        float3 low_limits = make_float3(-3, -3, 0);
        float3 high_limits = make_float3(3, 3, 4);
        auto mesh = tracker->ExtractMesh(low_limits, high_limits);
        mesh.SaveToFile("");

    }
    if(render_frame_output->IsActive()) {
        *virtual_rgb = tracker->GenerateRgb(inputSize.x, inputSize.y);
        std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
        render_frame_output->AddPoint(ts, new slambench::values::FrameValue(inputSize.x, inputSize.y, slambench::io::pixelformat::RGB_III_888, virtual_rgb->data));
    }

    if(rgb_frame_output->IsActive()) {
        std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
        rgb_frame_output->AddPoint(ts, new slambench::values::FrameValue(inputSize.x, inputSize.y, slambench::io::pixelformat::RGB_III_888, imRGB->data));
    }
    if(depth_frame_output->IsActive()) {
        std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
        depth_frame_output->AddPoint(ts, new slambench::values::FrameValue(inputSize.x, inputSize.y, slambench::io::pixelformat::D_I_16, imD->data));
    }

    return true;
}

