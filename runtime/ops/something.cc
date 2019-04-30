#include <array>
#include <map>
#include <numeric>

#include <chainerx/array.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_xcvm_ops.h>

namespace chainer_compiler {
namespace runtime {

namespace {

constexpr double k_exp_clip = std::log(1000. / 16);

constexpr size_t k_anchor_size = 32;
constexpr std::array<double, 3> k_anchor_ratios{{0.5, 1., 2.}};
constexpr double k_nms_thresh = 0.7;
constexpr size_t k_train_nms_limit_pre = 2000;
constexpr size_t k_train_nms_limit_post = 2000;
constexpr size_t k_test_nms_limit_pre = 1000;
constexpr size_t k_test_nms_limit_post = 1000;

namespace {
template <typename Int, typename Iter, typename Comp>
std::vector<Int> argsort(Iter first, Iter last, Comp comp) {
    std::vector<Int> indices(std::distance(first, last));
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [first, comp](Int i, Int j) { return comp(*(first + i), *(first + j)); });
    return indices;
}

double calc_area(std::array<double, 4> const& tlbr) {
    return (tlbr[2] - tlbr[0]) * (tlbr[3] - tlbr[1]);
}

template <typename T>
const T& at(const std::vector<T>& v, size_t i) {
    // return v.at(i);
    return v[i];
}

}  // namespace

std::vector<size_t> NonMaximumSuppression(std::vector<std::array<double, 4>> const& roi_l_list, double nms_threash, size_t nms_limit) {
    std::vector<double> bbox_area_list(roi_l_list.size());
    std::transform(
            roi_l_list.begin(), roi_l_list.end(), bbox_area_list.begin(), [](const std::array<double, 4>& roi) { return calc_area(roi); });
    std::vector<size_t> selected_indices;
    for (int64_t i = 0; i < roi_l_list.size(); ++i) {
        const std::array<double, 4> b_tlbr = at(roi_l_list, i);
        const bool is_selected = std::all_of(
                selected_indices.begin(), selected_indices.end(), [nms_threash, i, b_tlbr, &bbox_area_list, &roi_l_list](size_t s) {
                    const std::array<double, 4>& selected_roi_tlbr = roi_l_list[s];
                    const std::array<double, 4> tlbr{{std::max(b_tlbr[0], selected_roi_tlbr[0]),
                                                      std::max(b_tlbr[1], selected_roi_tlbr[1]),
                                                      std::min(b_tlbr[2], selected_roi_tlbr[2]),
                                                      std::min(b_tlbr[3], selected_roi_tlbr[3])}};
                    const double area = (tlbr[0] < tlbr[2]) && (tlbr[1] < tlbr[3]) ? calc_area(tlbr) : 0;
                    const double iou = area / (at(bbox_area_list, i) + at(bbox_area_list, s) - area);
                    return nms_threash > iou;
                });
        if (is_selected) {
            selected_indices.push_back(i);
        }
        if (selected_indices.size() >= nms_limit) {
            break;
        }
    }
    return selected_indices;
}

std::vector<chainerx::Array> ChainerCVRPNDecode(
        const std::vector<chainerx::Array>& hs,
        const std::vector<chainerx::Array>& locs,
        const std::vector<chainerx::Array>& confs,
        const chainerx::Array& in_shape,
        const std::vector<double>& scales) {
    for (const chainerx::Array& h : hs) {
        CHECK(IsNativeDevice(&h.device()));
    }
    for (const chainerx::Array& loc : locs) {
        CHECK(IsNativeDevice(&loc.device()));
    }
    for (const chainerx::Array& conf : confs) {
        CHECK(IsNativeDevice(&conf.device()));
    }
    CHECK(IsNativeDevice(&in_shape.device()));

    const size_t k_nms_limit_pre = k_train_nms_limit_pre;
    const size_t k_nms_limit_post = k_train_nms_limit_post;
    const double in_shape_h = static_cast<double>(AsScalar(in_shape.At({2})));
    const double in_shape_w = static_cast<double>(AsScalar(in_shape.At({3})));
    std::vector<std::array<double, 4>> rois_list;
    std::vector<chainerx::Array> roi_indices_list;
    const int64_t batch_size = static_cast<int64_t>(chainerx::AsScalar(in_shape.At({0})));
    for (int64_t b = 0; b < batch_size; ++b) {
        std::vector<std::array<double, 4>> rois_list_per_batch;
        std::vector<double> confs_list_per_batch;
        for (size_t l = 0; l < scales.size(); ++l) {
            const float* loc_l = static_cast<float*>(locs[l].raw_data()) + b * locs[l].shape()[1] * locs[l].shape()[2];
            const float* conf_l = static_cast<float*>(confs[l].raw_data()) + b * confs[l].shape()[1];
            const auto& h_l = hs[l];
            const int64_t k_l = locs[l].shape()[1];
            std::vector<std::array<double, 4>> roi_l_list(h_l.shape()[2] * h_l.shape()[3] * k_anchor_ratios.size());
            size_t ki = 0;
            for (size_t u = 0; u < h_l.shape()[2]; ++u) {
                for (size_t v = 0; v < h_l.shape()[3]; ++v) {
                    for (size_t iar = 0; iar < k_anchor_ratios.size(); ++iar) {
                        double ar = k_anchor_ratios[iar];
                        const double ay = (u + 0.5) / scales[l];
                        const double ax = (v + 0.5) / scales[l];
                        const double w = std::round(1.0 / scales[l] / std::sqrt(ar));
                        const double h = std::round(w * ar);
                        const double ah = h * (k_anchor_size << l) * scales[l];
                        const double aw = w * (k_anchor_size << l) * scales[l];

                        const int64_t k = h_l.shape()[3] * k_anchor_ratios.size() * u + k_anchor_ratios.size() * v + iar;
                        const double loc_l_y = *(loc_l + k * 4 + 0);
                        const double loc_l_x = *(loc_l + k * 4 + 1);
                        const double loc_l_h = *(loc_l + k * 4 + 2);
                        const double loc_l_w = *(loc_l + k * 4 + 3);

                        const double roi_l_y = ay + ah * loc_l_y;
                        const double roi_l_x = ax + aw * loc_l_x;
                        const double roi_l_h = ah * std::exp(std::min(loc_l_h, k_exp_clip));
                        const double roi_l_w = aw * std::exp(std::min(loc_l_w, k_exp_clip));

                        // yxhw -> tlbr (top left, bottom right)
                        roi_l_list[ki][0] = std::max(roi_l_y - roi_l_h / 2.0, 0.0);  // roi_l_tly
                        roi_l_list[ki][1] = std::max(roi_l_x - roi_l_w / 2.0, 0.0);  // roi_l_tlx
                        roi_l_list[ki][2] = std::min(roi_l_y + roi_l_h / 2.0, in_shape_h);  // roi_l_bry
                        roi_l_list[ki][3] = std::min(roi_l_x + roi_l_w / 2.0, in_shape_w);  // roi_l_brx
                        ++ki;
                    }  // end for iar
                }  // end for v
            }  // end for u

            // reduce size of `conf_l` to `nms_limit_pre`
            std::vector<size_t> cut_indices = argsort<size_t>(conf_l, conf_l + k_l, [](float a, float b) { return a > b; });
            std::vector<std::array<double, 4>> roi_l_list_cut(std::min(static_cast<size_t>(k_l), k_nms_limit_pre));
            std::vector<double> conf_l_list_cut(std::min(static_cast<size_t>(k_l), k_nms_limit_pre));
            std::transform(
                    cut_indices.begin(), cut_indices.begin() + roi_l_list_cut.size(), roi_l_list_cut.begin(), [&roi_l_list](size_t i) {
                        return at(roi_l_list, i);
                    });
            std::transform(cut_indices.begin(), cut_indices.begin() + conf_l_list_cut.size(), conf_l_list_cut.begin(), [&conf_l](size_t i) {
                return *(conf_l + i);
            });

            // mask
            std::vector<size_t> mask_indices(roi_l_list_cut.size());
            std::iota(mask_indices.begin(), mask_indices.end(), 0);
            const auto mask_indices_end_iter = std::remove_if(mask_indices.begin(), mask_indices.end(), [&roi_l_list_cut](size_t i) {
                const std::array<double, 4>& roi = at(roi_l_list_cut, i);
                const double roi_l_tly = roi[0];
                const double roi_l_tlx = roi[1];
                const double roi_l_bry = roi[2];
                const double roi_l_brx = roi[3];
                return (roi_l_bry <= roi_l_tly) || (roi_l_brx <= roi_l_tlx);
            });
            const size_t masked_size = std::distance(mask_indices.begin(), mask_indices_end_iter);
            std::vector<std::array<double, 4>> roi_l_list_masked(masked_size);
            std::vector<double> conf_l_list_masked(masked_size);
            std::transform(mask_indices.begin(), mask_indices_end_iter, roi_l_list_masked.begin(), [&roi_l_list_cut](size_t i) {
                return at(roi_l_list_cut, i);
            });
            std::transform(mask_indices.begin(), mask_indices_end_iter, conf_l_list_masked.begin(), [&conf_l_list_cut](size_t i) {
                return at(conf_l_list_cut, i);
            });

            // NMS
            std::vector<size_t> indices = NonMaximumSuppression(roi_l_list_masked, k_nms_thresh, k_nms_limit_post);
            std::vector<std::array<double, 4>> roi_l_list_suppressed(indices.size());
            std::vector<double> conf_l_list_suppressed(indices.size());
            std::transform(indices.begin(), indices.end(), roi_l_list_suppressed.begin(), [&roi_l_list_masked](size_t i) {
                return at(roi_l_list_masked, i);
            });
            std::transform(indices.begin(), indices.end(), conf_l_list_suppressed.begin(), [&conf_l_list_masked](size_t i) {
                return at(conf_l_list_masked, i);
            });

            rois_list_per_batch.insert(rois_list_per_batch.end(), roi_l_list_suppressed.begin(), roi_l_list_suppressed.end());
            confs_list_per_batch.insert(confs_list_per_batch.end(), conf_l_list_suppressed.begin(), conf_l_list_suppressed.end());

        }  // end for l

        // reduce size of `roi_list_per_batch` to `nms_limit_post`
        const auto& conf = confs_list_per_batch;
        std::vector<size_t> nms_post_indices = argsort<size_t>(conf.begin(), conf.end(), [](double a, double b) { return a > b; });
        std::vector<std::array<double, 4>> rois_list_per_batch_sorted(std::min(k_nms_limit_post, nms_post_indices.size()));
        std::transform(
                nms_post_indices.begin(),
                nms_post_indices.begin() + rois_list_per_batch_sorted.size(),
                rois_list_per_batch_sorted.begin(),
                [&rois_list_per_batch](size_t i) { return at(rois_list_per_batch, i); });
        rois_list.insert(rois_list.end(), rois_list_per_batch_sorted.begin(), rois_list_per_batch_sorted.end());
        roi_indices_list.push_back(chainerx::Full({static_cast<int64_t>(rois_list.size())}, b));
    }
    chainerx::Array rois = chainerx::Zeros({static_cast<int64_t>(rois_list.size()) * 4}, hs.front().dtype());
    for (size_t i = 0; i < rois_list.size(); ++i) {
        *(static_cast<float*>(rois.raw_data()) + i * 4 + 0) = rois_list[i][0];
        *(static_cast<float*>(rois.raw_data()) + i * 4 + 1) = rois_list[i][1];
        *(static_cast<float*>(rois.raw_data()) + i * 4 + 2) = rois_list[i][2];
        *(static_cast<float*>(rois.raw_data()) + i * 4 + 3) = rois_list[i][3];
    }
    rois = chainerx::Reshape(rois, {static_cast<int64_t>(rois_list.size()), 4});
    chainerx::Array roi_indices = chainerx::Reshape(chainerx::Concatenate(roi_indices_list), {static_cast<int64_t>(rois_list.size())});
    return std::vector<chainerx::Array>({std::move(rois), std::move(roi_indices)});
}

}  // namespace

std::vector<chainerx::Array> DoSomethingOp::RunImpl(chainer_compiler::runtime::XCVMState* st, const std::vector<chainerx::Array>& inputs) {
    auto found = st->options().custom_op_funcs.find(func_name);
    if (found != st->options().custom_op_funcs.end()) {
        return found->second(inputs);
    }

    if (func_name == "ChainerCVRPNDecode") {
        constexpr int64_t k_n_pyramids = 5;
        const std::vector<double> k_scales({1. / 4, 1. / 8, 1. / 16, 1. / 32, 1. / 64});
        assert(k_scales.size() == k_n_pyramids);
        assert(inputs.size() == (3 * k_n_pyramids + 1));
        const std::vector<chainerx::Array> hs(inputs.begin() + 0 * k_n_pyramids, inputs.begin() + 1 * k_n_pyramids);
        const std::vector<chainerx::Array> locs(inputs.begin() + 1 * k_n_pyramids, inputs.begin() + 2 * k_n_pyramids);
        const std::vector<chainerx::Array> confs(inputs.begin() + 2 * k_n_pyramids, inputs.begin() + 3 * k_n_pyramids);
        const chainerx::Array& in_shape = inputs.back();
        return ChainerCVRPNDecode(hs, locs, confs, in_shape, k_scales);
    }

    CHECK(false) << "Not implemented: " << func_name;
}

}  // namespace runtime
}  // namespace chainer_compiler
