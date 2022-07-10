/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/core/ocl.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

//#include<stdint-gcc.h>
#define TO_SIZE_T(x) (x > 0 ? x : 0)

using namespace std;

namespace ORB_SLAM3
{

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th, const bool bFarPoints, const float thFarPoints)
{
    int nmatches=0, left = 0, right = 0;

    const bool bFactor = th!=1.0;

    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
    {
        MapPoint* pMP = vpMapPoints[iMP];
        if(!pMP->mbTrackInView && !pMP->mbTrackInViewR)
            continue;

        if(bFarPoints && pMP->mTrackDepth>thFarPoints)
            continue;

        if(pMP->isBad())
            continue;

        if(pMP->mbTrackInView)
        {
            const int &nPredictedLevel = pMP->mnTrackScaleLevel;

            // The size of the window will depend on the viewing direction
            float r = RadiusByViewingCos(pMP->mTrackViewCos);

            if(bFactor)
                r*=th;

            const vector<size_t> vIndices =
                    F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

            if(!vIndices.empty()){
                const cv::Mat MPdescriptor = pMP->GetDescriptor();

                int bestDist=256;
                int bestLevel= -1;
                int bestDist2=256;
                int bestLevel2 = -1;
                int bestIdx =-1 ;

                // Get best and second matches with near keypoints
                for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                {
                    const size_t idx = *vit;

                    if(F.mvpMapPoints[idx])
                        if(F.mvpMapPoints[idx]->Observations()>0)
                            continue;

                    if(F.Nleft == -1 && F.mvuRight[idx]>0)
                    {
                        const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                        if(er>r*F.mvScaleFactors[nPredictedLevel])
                            continue;
                    }

                    const cv::Mat &d = F.mDescriptors.row(idx);

                    const int dist = DescriptorDistance(MPdescriptor,d);

                    if(dist<bestDist)
                    {
                        bestDist2=bestDist;
                        bestDist=dist;
                        bestLevel2 = bestLevel;
                        bestLevel = (F.Nleft == -1) ? F.mvKeysUn[idx].octave
                                                    : (idx < TO_SIZE_T(F.Nleft)) ? F.mvKeys[idx].octave
                                                                        : F.mvKeysRight[idx - F.Nleft].octave;
                        bestIdx=idx;
                    }
                    else if(dist<bestDist2)
                    {
                        bestLevel2 = (F.Nleft == -1) ? F.mvKeysUn[idx].octave
                                                        : (idx < TO_SIZE_T(F.Nleft)) ? F.mvKeys[idx].octave
                                                                        : F.mvKeysRight[idx - F.Nleft].octave;
                        bestDist2=dist;
                    }
                }

                // Apply ratio to second match (only if best and second are in the same scale level)
                if(bestDist<=TH_HIGH)
                {
                    if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                        continue;

                    if(bestLevel!=bestLevel2 || bestDist<=mfNNratio*bestDist2){
                        F.mvpMapPoints[bestIdx]=pMP;

                        if(F.Nleft != -1 && F.mvLeftToRightMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                            F.mvpMapPoints[F.mvLeftToRightMatch[bestIdx] + F.Nleft] = pMP;
                            nmatches++;
                            right++;
                        }

                        nmatches++;
                        left++;
                    }
                }
            }
        }

        if(F.Nleft != -1 && pMP->mbTrackInViewR){
            const int &nPredictedLevel = pMP->mnTrackScaleLevelR;
            if(nPredictedLevel != -1){
                float r = RadiusByViewingCos(pMP->mTrackViewCosR);

                const vector<size_t> vIndices =
                        F.GetFeaturesInArea(pMP->mTrackProjXR,pMP->mTrackProjYR,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel,true);

                if(vIndices.empty())
                    continue;

                const cv::Mat MPdescriptor = pMP->GetDescriptor();

                int bestDist=256;
                int bestLevel= -1;
                int bestDist2=256;
                int bestLevel2 = -1;
                int bestIdx =-1 ;

                // Get best and second matches with near keypoints
                for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                {
                    const size_t idx = *vit;

                    if(F.mvpMapPoints[idx + F.Nleft])
                        if(F.mvpMapPoints[idx + F.Nleft]->Observations()>0)
                            continue;


                    const cv::Mat &d = F.mDescriptors.row(idx + F.Nleft);

                    const int dist = DescriptorDistance(MPdescriptor,d);

                    if(dist<bestDist)
                    {
                        bestDist2=bestDist;
                        bestDist=dist;
                        bestLevel2 = bestLevel;
                        bestLevel = F.mvKeysRight[idx].octave;
                        bestIdx=idx;
                    }
                    else if(dist<bestDist2)
                    {
                        bestLevel2 = F.mvKeysRight[idx].octave;
                        bestDist2=dist;
                    }
                }

                // Apply ratio to second match (only if best and second are in the same scale level)
                if(bestDist<=TH_HIGH)
                {
                    if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                        continue;

                    if(F.Nleft != -1 && F.mvRightToLeftMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                        F.mvpMapPoints[F.mvRightToLeftMatch[bestIdx]] = pMP;
                        nmatches++;
                        left++;
                    }


                    F.mvpMapPoints[bestIdx + F.Nleft]=pMP;
                    nmatches++;
                    right++;
                }
            }
        }
    }
    return nmatches;
}

float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}

int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)
{
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));

    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

    int nmatches=0;

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

    while(KFit != KFend && Fit != Fend)
    {
        if(KFit->first == Fit->first)
        {
            const vector<unsigned int> vIndicesKF = KFit->second;
            const vector<unsigned int> vIndicesF = Fit->second;

            for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];

                MapPoint* pMP = vpMapPointsKF[realIdxKF];

                if(!pMP)
                    continue;

                if(pMP->isBad())
                    continue;

                const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF);

                int bestDist1=256;
                int bestIdxF =-1 ;
                int bestDist2=256;

                int bestDist1R=256;
                int bestIdxFR =-1 ;
                int bestDist2R=256;

                for(size_t iF=0; iF<vIndicesF.size(); iF++)
                {
                    if(F.Nleft == -1){
                        const unsigned int realIdxF = vIndicesF[iF];

                        if(vpMapPointMatches[realIdxF])
                            continue;

                        const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                        const int dist =  DescriptorDistance(dKF,dF);

                        if(dist<bestDist1)
                        {
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdxF=realIdxF;
                        }
                        else if(dist<bestDist2)
                        {
                            bestDist2=dist;
                        }
                    }
                    else{
                        const unsigned int realIdxF = vIndicesF[iF];

                        if(vpMapPointMatches[realIdxF])
                            continue;

                        const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                        const int dist =  DescriptorDistance(dKF,dF);

                        if(realIdxF < F.Nleft && dist<bestDist1){
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdxF=realIdxF;
                        }
                        else if(realIdxF < F.Nleft && dist<bestDist2){
                            bestDist2=dist;
                        }

                        if(realIdxF >= F.Nleft && dist<bestDist1R){
                            bestDist2R=bestDist1R;
                            bestDist1R=dist;
                            bestIdxFR=realIdxF;
                        }
                        else if(realIdxF >= F.Nleft && dist<bestDist2R){
                            bestDist2R=dist;
                        }
                    }

                }

                if(bestDist1<=TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        vpMapPointMatches[bestIdxF]=pMP;

                        const cv::KeyPoint &kp =
                                (!pKF->mpCamera2) ? pKF->mvKeysUn[realIdxKF] :
                                (realIdxKF >= pKF -> NLeft) ? pKF -> mvKeysRight[realIdxKF - pKF -> NLeft]
                                                            : pKF -> mvKeys[realIdxKF];

                        if(mbCheckOrientation)
                        {
                            cv::KeyPoint &Fkp =
                                    (!pKF->mpCamera2 || F.Nleft == -1) ? F.mvKeys[bestIdxF] :
                                    (bestIdxF >= F.Nleft) ? F.mvKeysRight[bestIdxF - F.Nleft]
                                                            : F.mvKeys[bestIdxF];

                            float rot = kp.angle-Fkp.angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF);
                        }
                        nmatches++;
                    }

                    if(bestDist1R<=TH_LOW)
                    {
                        if(static_cast<float>(bestDist1R)<mfNNratio*static_cast<float>(bestDist2R) || true)
                        {
                            vpMapPointMatches[bestIdxFR]=pMP;

                            const cv::KeyPoint &kp =
                                    (!pKF->mpCamera2) ? pKF->mvKeysUn[realIdxKF] :
                                    (realIdxKF >= pKF -> NLeft) ? pKF -> mvKeysRight[realIdxKF - pKF -> NLeft]
                                                                : pKF -> mvKeys[realIdxKF];

                            if(mbCheckOrientation)
                            {
                                cv::KeyPoint &Fkp =
                                        (!F.mpCamera2) ? F.mvKeys[bestIdxFR] :
                                        (bestIdxFR >= F.Nleft) ? F.mvKeysRight[bestIdxFR - F.Nleft]
                                                                : F.mvKeys[bestIdxFR];

                                float rot = kp.angle-Fkp.angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdxFR);
                            }
                            nmatches++;
                        }
                    }
                }

            }

            KFit++;
            Fit++;
        }
        else if(KFit->first < Fit->first)
        {
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

int ORBmatcher::SearchByProjection(KeyFrame* pKF, Sophus::Sim3f &Scw, const vector<MapPoint*> &vpPoints,
                                    vector<MapPoint*> &vpMatched, int th, float ratioHamming)
{
    // Get Calibration Parameters for later projection
    [[maybe_unused]] const float &fx = pKF->fx;
    [[maybe_unused]] const float &fy = pKF->fy;
    [[maybe_unused]] const float &cx = pKF->cx;
    [[maybe_unused]] const float &cy = pKF->cy;

    Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
    Eigen::Vector3f Ow = Tcw.inverse().translation();

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        Eigen::Vector3f p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        Eigen::Vector3f p3Dc = Tcw * p3Dw;

        // Depth must be positive
        if(p3Dc(2)<0.0)
            continue;

        // Project into Image
        const Eigen::Vector2f uv = pKF->mpCamera->project(p3Dc);

        // Point must be inside the image
        if(!pKF->IsInImage(uv(0),uv(1)))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3f PO = p3Dw-Ow;
        const float dist = PO.norm();

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        Eigen::Vector3f Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_LOW*ratioHamming)
        {
            vpMatched[bestIdx]=pMP;
            nmatches++;
        }

    }

    return nmatches;
}

int ORBmatcher::SearchByProjection(KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<MapPoint*> &vpPoints, const std::vector<KeyFrame*> &vpPointsKFs,
                                    std::vector<MapPoint*> &vpMatched, std::vector<KeyFrame*> &vpMatchedKF, int th, float ratioHamming)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
    Eigen::Vector3f Ow = Tcw.inverse().translation();

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];
        KeyFrame* pKFi = vpPointsKFs[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        Eigen::Vector3f p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        Eigen::Vector3f p3Dc = Tcw * p3Dw;

        // Depth must be positive
        if(p3Dc(2)<0.0)
            continue;

        // Project into Image
        const float invz = 1/p3Dc(2);
        const float x = p3Dc(0)*invz;
        const float y = p3Dc(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3f PO = p3Dw-Ow;
        const float dist = PO.norm();

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        Eigen::Vector3f Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_LOW*ratioHamming)
        {
            vpMatched[bestIdx] = pMP;
            vpMatchedKF[bestIdx] = pKFi;
            nmatches++;
        }

    }

    return nmatches;
}

int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
    int nmatches=0;
    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
    vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    {
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;
        if(level1>0)
            continue;

        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

        if(vIndices2.empty())
            continue;

        cv::Mat d1 = F1.mDescriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            int dist = DescriptorDistance(d1,d2);

            if(vMatchedDistance[i2]<=dist)
                continue;

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }

        if(bestDist<=TH_LOW)
        {
            if(bestDist<(float)bestDist2*mfNNratio)
            {
                if(vnMatches21[bestIdx2]>=0)
                {
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
                vnMatches12[i1]=bestIdx2;
                vnMatches21[bestIdx2]=i1;
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;

                if(mbCheckOrientation)
                {
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }

    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }

    //Update prev matched
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)
            vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;

    return nmatches;
}

} //namespace ORB_SLAM
