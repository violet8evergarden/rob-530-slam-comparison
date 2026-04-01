#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <sstream>
#include <unistd.h>

#include <opencv2/core/core.hpp>

#include <System.h>
#include "ImuTypes.h"

using namespace std;

// Function Signatures
void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro, double time_offset);

int main(int argc, char **argv)
{
    if(argc != 6)
    {
        cerr << endl << "Usage: ./stereo_inertial_didlm path_to_vocabulary path_to_settings path_to_sequence path_to_association path_to_imu_txt" << endl;
        return 1;
    }

    // 1. Read IMU Time Offset from YAML
    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return 1;
    }
    
    double imu_time_offset = 0.0;
    cv::FileNode nodeOffset = fsSettings["IMU.TimeOffset"];
    if(!nodeOffset.empty() && nodeOffset.isReal())
    {
        imu_time_offset = nodeOffset.real();
    }
    cout << "Loaded IMU.TimeOffset from config: " << imu_time_offset << " seconds." << endl;

    // 2. Retrieve paths to images and IMU
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestampsCam;
    
    vector<double> vTimestampsImu;
    vector<cv::Point3f> vAcc, vGyro;

    string strSequencePath = string(argv[3]);
    string strAssociationFilename = string(argv[4]);
    string strImuFilename = string(argv[5]);

    cout << endl << "Loading images from association file: " << strAssociationFilename << endl;
    LoadImages(strAssociationFilename, vstrImageLeft, vstrImageRight, vTimestampsCam);

    cout << "Loading IMU from file: " << strImuFilename << endl;
    LoadIMU(strImuFilename, vTimestampsImu, vAcc, vGyro, imu_time_offset);

    // Check consistency
    int nImages = vstrImageLeft.size();
    int nImu = vTimestampsImu.size();
    
    if(nImages == 0 || nImu == 0)
    {
        cerr << endl << "ERROR: Failed to load images or IMU." << endl;
        return 1;
    }

    // 3. Find first IMU measurement to be considered
    int first_imu = 0;
    while(first_imu < nImu && vTimestampsImu[first_imu] <= vTimestampsCam[0])
        first_imu++;
    first_imu--; 
    if(first_imu < 0) first_imu = 0;

    // 4. Create SLAM system in IMU_STEREO mode
    cout << "Initializing SLAM system in IMU_STEREO mode..." << endl;
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_STEREO, true);

    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl;
    cout << "IMU measurements: " << nImu << endl << endl;

    // Main loop
    cv::Mat imLeft, imRight;
    vector<ORB_SLAM3::IMU::Point> vImuMeas;

    for(int ni=0; ni<nImages; ni++)
    {
        // Read left and right images
        imLeft = cv::imread(strSequencePath + "/" + vstrImageLeft[ni], cv::IMREAD_UNCHANGED);
        imRight = cv::imread(strSequencePath + "/" + vstrImageRight[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestampsCam[ni];

        if(imLeft.empty() || imRight.empty())
        {
            cerr << endl << "Failed to load image at frame " << ni << endl;
            return 1;
        }

        // 5. Load IMU measurements between the previous frame and this current frame
        vImuMeas.clear();
        if(ni > 0)
        {
            while(first_imu < nImu && vTimestampsImu[first_imu] <= tframe)
            {
                vImuMeas.push_back(ORB_SLAM3::IMU::Point(
                    vAcc[first_imu].x, vAcc[first_imu].y, vAcc[first_imu].z,
                    vGyro[first_imu].x, vGyro[first_imu].y, vGyro[first_imu].z,
                    vTimestampsImu[first_imu]));
                first_imu++;
            }
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // 6. Pass the images AND the IMU vector to the SLAM system
        SLAM.TrackStereo(imLeft, imRight, tframe, vImuMeas);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        vTimesTrack[ni] = ttrack;
        double track_ms = ttrack * 1000.0;
        SLAM.InsertTrackTime(track_ms);

        // Wait to load the next frame to simulate real-time
        double T=0;
        if(ni < nImages-1)
            T = vTimestampsCam[ni+1] - tframe;
        else if(ni > 0)
            T = tframe - vTimestampsCam[ni-1];

        if(ttrack < T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Save trajectories
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");   

    return 0;
}

// -----------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());

    if(!fAssociation.is_open()) {
        cerr << endl << "CRITICAL ERROR: Could not open association file at: " << strAssociationFilename << endl;
        return; 
    }

    int lineCount = 0;
    string s;

    while(getline(fAssociation, s))
    {
        if(!s.empty() && s[0] != '#')
        {
            stringstream ss;
            ss << s;
            double t;
            string sLeft, sRight;
            
            if(!(ss >> t >> sLeft >> t >> sRight)) {
                continue;
            }

            vTimestamps.push_back(t);
            vstrImageLeft.push_back(sLeft);
            vstrImageRight.push_back(sRight);
            lineCount++;
        }
    }
    fAssociation.close();
    cout << "Finished! Total stereo pairs loaded: " << vstrImageLeft.size() << endl;
}

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro, double time_offset)
{
    ifstream fImu;
    fImu.open(strImuPath.c_str());
    if(!fImu.is_open()){
        cerr << "Could not open IMU file!" << endl;
        return;
    }

    string line;
    double current_t = 0;
    cv::Point3f current_acc, current_gyr;

    while(getline(fImu, line))
    {
        if (line.find("IMU timestamp:") != string::npos) {
            size_t pos = line.find(":");
            current_t = stod(line.substr(pos + 1)) / 1e9; 
            current_t = current_t + time_offset; 
        }
        
        if (line.find("Angular Velocity:") != string::npos) {
            sscanf(line.c_str(), "  Angular Velocity: x=%f, y=%f, z=%f", &current_gyr.x, &current_gyr.y, &current_gyr.z);
        }

        if (line.find("Linear Acceleration:") != string::npos) {
            sscanf(line.c_str(), "  Linear Acceleration: x=%f, y=%f, z=%f", &current_acc.x, &current_acc.y, &current_acc.z);
            
            vTimeStamps.push_back(current_t);
            vAcc.push_back(current_acc);
            vGyro.push_back(current_gyr);
        }
    }
    fImu.close();
}
