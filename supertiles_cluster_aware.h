#ifndef __SUPERTILES_CLUSTER_AWARE__
#define __SUPERTILES_CLUSTER_AWARE__

// cluster Aware based on supertiles and assignments
#include <vector>
#include <stdint.h>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <unordered_map>
#include "convexity/measureTriples.h"

namespace supertiles
{
    // supertiles std::vector<double> features in qt position
    // assignments std::vector<uint32_t> assignments
    // levels size_t
    // clusterInfo std::vector<int> clusterInfo;  
	// labelnum  cluster number

    static int width = 0;
    static int xy2pos(const int x, const int y) {
        return x * width + y;
    }
    static void pos2xy(const int pos, int & x, int & y) {
        x = pos / width; y = pos % width;
    }

    static void qt2grid(const std::vector<uint32_t> & qt_ass, std::vector<uint32_t> & grid, int start, int cur_x, int cur_y, int cur_width) {
        if(cur_width == 2){
            grid[xy2pos(cur_x, cur_y)] = qt_ass[start];
            grid[xy2pos(cur_x, cur_y+1)] = qt_ass[start+1];
            grid[xy2pos(cur_x+1, cur_y)] = qt_ass[start+2];
            grid[xy2pos(cur_x+1, cur_y+1)] = qt_ass[start+3];
            return;
        }
        int cwidth = cur_width / 2;
        int part = cwidth * cwidth;
        qt2grid(qt_ass, grid, start, cur_x, cur_y, cwidth);
        qt2grid(qt_ass, grid, start + part, cur_x, cur_y + cwidth, cwidth);
        qt2grid(qt_ass, grid, start + 2 * part,  cur_x + cwidth, cur_y, cwidth);
        qt2grid(qt_ass, grid, start + 3 * part, cur_x + cwidth, cur_y + cwidth, cwidth);
    }

    static double distHighDim(const std::vector<double> & feat, const int feat_length, const int delta, const int x, const int y) {
        double res = 0;
        int start_x = feat_length * (delta + x);
        int start_y = feat_length * (delta + y);
        for(auto i = 0; i < feat_length; i++) {
            res += pow(feat[start_x + i] - feat[start_y + i], 2);
        }
        return sqrt(res);
    }

    static double distLowDim(const double x1, const double y1, const double x2, const double y2) {
        double res = pow((x1 - x2) / width, 2) + pow((y1 - y2) / width, 2);
        return sqrt(res);
    }

    class ClusterAwareGridLayout {
        public:
            using feat_t = std::vector<double>;
            using ass_t = std::vector<uint32_t>;
            using cls_t = std::vector<int>;
            using dist_t = std::vector<double>;
            using pos_t = std::vector<std::vector<double>>;
            ClusterAwareGridLayout(const feat_t & feats, const int feat_length, const ass_t & qt_ass, const cls_t & clusters, 
                const size_t level) {
                    ass_t bf_ass(qt_ass);
                    size_t cur_level = 0;
                    int cur_delta = 0;
                    int nxt_length = qt_ass.size() / 4;
                    while(cur_level < level) {
                        ass_t cur_ass;
                        for(auto i = 0; i < nxt_length; i++) {
                            dist_t dist4(4, 0.);
                            for(auto s1 = 0; s1 < 3; s1++) {
                                for(auto s2 = s1 + 1; s2 < 4; s2++) {
                                    auto dist = distHighDim(feats, feat_length, cur_delta, 4*i+s1, 4*i+s2);
                                    dist4[s1] += dist;
                                    dist4[s2] += dist;
                                }
                            }
                            int mins = std::min_element(dist4.begin(), dist4.end()) - dist4.begin();
                            cur_ass.push_back(bf_ass[4*i+mins]);
                        }
                        bf_ass = cur_ass;
                        cur_level += 1;
                        cur_delta += (4 * nxt_length);
                        nxt_length /= 4;
                    }
                    width = int(round(sqrt(bf_ass.size())));
                    assert(width * width == bf_ass.size());
                    ass_t grid(bf_ass.size(), 0.);
                    qt2grid(bf_ass, grid, 0, 0, 0, width);
                    std::unordered_map<int, int> labelmap;
                    int startlabel = 0;
                    for(auto i = 0; i < bf_ass.size(); i++) {
                        grid_asses.push_back(i);
                        auto label = clusters[grid[i]];
                        if(labelmap.find(label) == labelmap.end()) {
                            labelmap[label] = startlabel;
                            startlabel += 1;
                        }
                        labels.push_back(labelmap[label]);
                    }
                    maxlabel = startlabel;
                }
        private:
            using grid_t = std::vector<int>;
            grid_t grid_asses;
            cls_t labels;
            int maxlabel;
        public:
            double compactnessCost(const double alpha = 10) {
                pos_t centers(maxlabel, std::vector<double>(3, 0.));
                width = int(round(sqrt(labels.size())));
                for(auto i = 0; i < labels.size(); i++) {
                    int x, y;
                    pos2xy(i, x, y);
                    centers[labels[i]][0] += x;
                    centers[labels[i]][1] += y;
                    centers[labels[i]][2] += 1;
                }
                for(auto i = 0;i < maxlabel; i++) {
                    if(centers[i][2] > 0) {
                        centers[i][0] /= centers[i][2];
                        centers[i][1] /= centers[i][2];
                    }
                }
                double res = 0;
                for(auto i = 0; i < labels.size(); i++) {
                    int x, y;
                    pos2xy(i, x, y);
                    res += distLowDim(centers[labels[i]][0], centers[labels[i]][1], x, y);
                }
                return alpha * res;
            }
            double convexityCost(const double alpha = 1000) {
                auto convexity = checkConvexForT(grid_asses, labels);
                return alpha * convexity[0] / convexity[1];
            }
    };
}

#endif 