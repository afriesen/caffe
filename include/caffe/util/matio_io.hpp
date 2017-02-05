#ifndef CAFFE_UTIL_MATIO_IO_H_
#define CAFFE_UTIL_MATIO_IO_H_

#include <unistd.h>
#include <string>

#include "google/protobuf/message.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#include <matio.h>

namespace caffe {

using ::google::protobuf::Message;

template <typename Dtype>
void ReadBlobFromMat(const char *fname, Blob<Dtype>* blob);

template <typename Dtype>
void WriteBlobToMat(const char *fname, bool write_diff,
   Blob<Dtype>* blob);

#ifdef USE_OPENCV
template< typename Dtype >
int read_from_mat(mat_t *matfp, matvar_t *matvar, cv::Mat & m, bool transpose, bool use_channels);

// read the specified field from the provided mat file and (optionally) transpose the data, since
// matlab stores data in column-major ordering
cv::Mat ReadCVMatFromMat(const std::string & filename, const std::string & field_name,
        bool transpose = true, bool use_channels = true);

#endif // USE_OPENCV

}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
