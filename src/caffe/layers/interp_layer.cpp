#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/interp.hpp"
#include "caffe/layers/interp_layer.hpp"

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudawarping.hpp>
#endif  // USE_OPENCV

namespace caffe {

template <typename Dtype>
void InterpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  InterpParameter interp_param = this->layer_param_.interp_param();
  pad_beg_ = interp_param.pad_beg();
  pad_end_ = interp_param.pad_end();
  CHECK_LE(pad_beg_, 0) << "Only supports non-pos padding (cropping) for now";
  CHECK_LE(pad_end_, 0) << "Only supports non-pos padding (cropping) for now";
  use_bilinear_ = ( interp_param.type() == InterpParameter_InterpType_Bilinear );
  CHECK(use_bilinear_ || interp_param.type() == InterpParameter_InterpType_NearestNeighbor );
}

template <typename Dtype>
void InterpLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_in_ = bottom[0]->height();
  width_in_ = bottom[0]->width();
  height_in_eff_ = height_in_ + pad_beg_ + pad_end_;
  width_in_eff_ = width_in_ + pad_beg_ + pad_end_;
  InterpParameter interp_param = this->layer_param_.interp_param();

  if ( bottom.size() > 1 ) {
    if ( interp_param.has_shrink_factor() || interp_param.has_zoom_factor()
        || interp_param.has_height() || interp_param.has_width() ) {
      LOG(FATAL) << "Interp layer cannot have a second bottom layer if any of its shrink, "
          "zoom, height, or width parameters are specified because it is ambiguous";
    }
//    CHECK_LE(num_, 1) << "only a batch size of 1 is currently supported with bottom-specified resizing";
    CHECK_EQ(bottom[1]->num(), num_);
//    CHECK_EQ(bottom[1]->channels(), channels_);
    CHECK_GT(bottom[1]->height(), 0);
    CHECK_GT(bottom[1]->width(), 0);
//    const Dtype * input_dims = bottom[1]->cpu_data() + bottom[1]->offset(num_, 0, 0, 0);
//    height_out_ = input_dims[0];
//    width_out_ = input_dims[1];
    height_out_ = bottom[1]->height();
    width_out_ = bottom[1]->width();
  } else if (interp_param.has_shrink_factor() &&
      !interp_param.has_zoom_factor()) {
    const int shrink_factor = interp_param.shrink_factor();
    CHECK_GE(shrink_factor, 1) << "Shrink factor must be positive";
    height_out_ = (height_in_eff_ - 1) / shrink_factor + 1;
    width_out_ = (width_in_eff_ - 1) / shrink_factor + 1;
  } else if (interp_param.has_zoom_factor() &&
             !interp_param.has_shrink_factor()) {
    const int zoom_factor = interp_param.zoom_factor();
    CHECK_GE(zoom_factor, 1) << "Zoom factor must be positive";
    height_out_ = height_in_eff_ + (height_in_eff_ - 1) * (zoom_factor - 1);
    width_out_ = width_in_eff_ + (width_in_eff_ - 1) * (zoom_factor - 1);
  } else if (interp_param.has_height() && interp_param.has_width()) {
    height_out_  = interp_param.height();
    width_out_  = interp_param.width();
  } else if (interp_param.has_shrink_factor() &&
             interp_param.has_zoom_factor()) {
    const int shrink_factor = interp_param.shrink_factor();
    const int zoom_factor = interp_param.zoom_factor();
    CHECK_GE(shrink_factor, 1) << "Shrink factor must be positive";
    CHECK_GE(zoom_factor, 1) << "Zoom factor must be positive";
    height_out_ = (height_in_eff_ - 1) / shrink_factor + 1;
    width_out_ = (width_in_eff_ - 1) / shrink_factor + 1;
    height_out_ = height_out_ + (height_out_ - 1) * (zoom_factor - 1);
    width_out_ = width_out_ + (width_out_ - 1) * (zoom_factor - 1);
  } else {
    LOG(FATAL) << "Interp layer parameters are unspecified";
  }
  CHECK_GT(height_in_eff_, 0) << "height should be positive";
  CHECK_GT(width_in_eff_, 0) << "width should be positive";
  CHECK_GT(height_out_, 0) << "height should be positive";
  CHECK_GT(width_out_, 0) << "width should be positive";
  top[0]->Reshape(num_, channels_, height_out_, width_out_);
}

template <typename Dtype>
void InterpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if ( use_bilinear_ ) {
    caffe_cpu_interp2< Dtype, false >( num_ * channels_,
            bottom[0]->cpu_data(), -pad_beg_, -pad_beg_, height_in_eff_, width_in_eff_, height_in_,
            width_in_, top[0]->mutable_cpu_data(), 0, 0, height_out_, width_out_, height_out_,
            width_out_ );
  } else {
#ifdef USE_OPENCV
    cv::Size size_in( width_in_, height_in_ );
    cv::Size size_out( width_out_, height_out_ );
    int type = sizeof( Dtype ) == 4 ? CV_32F : CV_64F;
    CHECK(sizeof(Dtype) == 4 || sizeof(Dtype) == 8);
    for ( int i = 0; i < num_; ++i ) {
      for ( int j = 0; j < channels_; ++j ) {
        cv::resize( cv::Mat(size_in, type, (void *) bottom[0]->cpu_data_at( i, j ) ),
                cv::Mat(size_out, type, top[0]->mutable_cpu_data_at( i, j ) ),
                size_out, 0, 0, CV_INTER_NN );
      }
    }
#else
    NOT_IMPLEMENTED;
#endif // USE_OPENCV
  }
}

template <typename Dtype>
void InterpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  if ( use_bilinear_ ) {
    caffe_set( bottom[0]->count(), Dtype( 0 ), bottom[0]->mutable_cpu_diff());
    caffe_cpu_interp2_backward< Dtype, false >( num_ * channels_,
            bottom[0]->mutable_cpu_diff(), -pad_beg_, -pad_beg_, height_in_eff_, width_in_eff_,
            height_in_, width_in_, top[0]->cpu_diff(), 0, 0, height_out_, width_out_, height_out_,
            width_out_ );
  } else {
    NOT_IMPLEMENTED;
  }
}

#ifndef CPU_ONLY
template <typename Dtype>
void InterpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (use_bilinear_) {
    caffe_gpu_interp2<Dtype,false>(num_ * channels_,
      bottom[0]->gpu_data(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
      top[0]->mutable_gpu_data(), 0, 0, height_out_, width_out_, height_out_, width_out_);
  } else {
#ifdef USE_OPENCV
    cv::Size size_in( width_in_, height_in_ );
    cv::Size size_out( width_out_, height_out_ );
    int type = sizeof( Dtype ) == 4 ? CV_32F : CV_64F;
    CHECK(sizeof(Dtype) == 4 || sizeof(Dtype) == 8);
    for ( int i = 0; i < num_; ++i ) {
      for ( int j = 0; j < channels_; ++j ) {
        cv::cuda::resize( cv::cuda::GpuMat(size_in, type, (void *) bottom[0]->gpu_data_at( i, j ) ),
                cv::cuda::GpuMat(size_out, type, top[0]->mutable_gpu_data_at( i, j ) ),
                size_out, 0, 0, CV_INTER_NN );
      }
    }
#else
    NOT_IMPLEMENTED;
#endif // USE_OPENCV
  }
}

template <typename Dtype>
void InterpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  if ( use_bilinear_ ) {
    caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
    caffe_gpu_interp2_backward<Dtype,false>(num_ * channels_,
      bottom[0]->mutable_gpu_diff(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
      top[0]->gpu_diff(), 0, 0, height_out_, width_out_, height_out_, width_out_);
  } else {
    NOT_IMPLEMENTED;
  }
}
#endif

#ifdef CPU_ONLY
STUB_GPU(InterpLayer);
#endif

INSTANTIATE_CLASS(InterpLayer);
REGISTER_LAYER_CLASS(Interp);

}  // namespace caffe
