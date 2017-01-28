#include <stdint.h>

#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include <matio.h>

namespace caffe {

template <typename Dtype> enum matio_types matio_type_map();
template <> enum matio_types matio_type_map<float>() { return MAT_T_SINGLE; }
template <> enum matio_types matio_type_map<double>() { return MAT_T_DOUBLE; }
template <> enum matio_types matio_type_map<int>() { return MAT_T_INT32; }
template <> enum matio_types matio_type_map<unsigned int>() { return MAT_T_UINT32; }

template <typename Dtype> enum matio_classes matio_class_map();
template <> enum matio_classes matio_class_map<float>() { return MAT_C_SINGLE; }
template <> enum matio_classes matio_class_map<double>() { return MAT_C_DOUBLE; }
template <> enum matio_classes matio_class_map<int>() { return MAT_C_INT32; }
template <> enum matio_classes matio_class_map<unsigned int>() { return MAT_C_UINT32; }

template <typename Dtype>
void ReadBlobFromMat(const char *fname, Blob<Dtype>* blob) {
  mat_t *matfp;
  matfp = Mat_Open(fname, MAT_ACC_RDONLY);
  CHECK(matfp) << "Error opening MAT file " << fname;
  // Read data
  matvar_t *matvar;
  matvar = Mat_VarReadInfo(matfp,"data");
  CHECK(matvar) << "Field 'data' not present in MAT file " << fname;
  {
    CHECK_EQ(matvar->class_type, matio_class_map<Dtype>())
      << "Field 'data' must be of the right class (single/double) in MAT file " << fname;
    CHECK(matvar->rank < 5) << "Field 'data' cannot have ndims > 4 in MAT file " << fname;
    blob->Reshape((matvar->rank > 3) ? matvar->dims[3] : 1,
	    (matvar->rank > 2) ? matvar->dims[2] : 1,
	    (matvar->rank > 1) ? matvar->dims[1] : 1,
	    (matvar->rank > 0) ? matvar->dims[0] : 0);
    Dtype* data = blob->mutable_cpu_data();
    int ret = Mat_VarReadDataLinear(matfp, matvar, data, 0, 1, blob->count());	 
    CHECK(ret == 0) << "Error reading array 'data' from MAT file " << fname;
    Mat_VarFree(matvar);
  }
  // Read diff, if present
  matvar = Mat_VarReadInfo(matfp,"diff");
  if (matvar && matvar -> data_size > 0) {
    CHECK_EQ(matvar->class_type, matio_class_map<Dtype>())
      << "Field 'diff' must be of the right class (single/double) in MAT file " << fname;
    Dtype* diff = blob->mutable_cpu_diff();
    int ret = Mat_VarReadDataLinear(matfp, matvar, diff, 0, 1, blob->count());	 
    CHECK(ret == 0) << "Error reading array 'diff' from MAT file " << fname;
    Mat_VarFree(matvar);
  }
  Mat_Close(matfp);
}

template <typename Dtype>
void WriteBlobToMat(const char *fname, bool write_diff,
    Blob<Dtype>* blob) {
  mat_t *matfp;
//  matfp = Mat_CreateVer( fname, NULL, MAT_FT_MAT73 );
  matfp = Mat_Create( fname, 0 );
  CHECK(matfp) << "Error creating MAT file " << fname;
  size_t dims[4];
  dims[0] = blob->width();
  dims[1] = blob->height();
  dims[2] = blob->channels();
  dims[3] = blob->num();
  matvar_t *matvar;
  // save data
  {
    matvar = Mat_VarCreate("data", matio_class_map<Dtype>(), matio_type_map<Dtype>(),
			   4, dims, blob->mutable_cpu_data(), 0);
    CHECK(matvar) << "Error creating 'data' variable";
    CHECK_EQ(Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE), 0) 
      << "Error saving array 'data' into MAT file " << fname;
    Mat_VarFree(matvar);
  }
  // save diff
  if (write_diff) {
    matvar = Mat_VarCreate("diff", matio_class_map<Dtype>(), matio_type_map<Dtype>(),
			   4, dims, blob->mutable_cpu_diff(), 0);
    CHECK(matvar) << "Error creating 'diff' variable";
    CHECK_EQ(Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE), 0)
      << "Error saving array 'diff' into MAT file " << fname;
    Mat_VarFree(matvar);
  }
  Mat_Close(matfp);
}


#ifdef USE_OPENCV
template< typename Dtype >
cv::Mat ReadCVMatFromMat(const std::string & filename, const std::string & field_name) {
  cv::Mat m_out;
  mat_t *matfp;
  matfp = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
  CHECK(matfp) << "Error opening MAT file " << filename;
  // Read data
  matvar_t *matvar;
  matvar = Mat_VarReadInfo(matfp, field_name.c_str());
  CHECK(matvar) << "Field '" << field_name << "' not present in MAT file " << filename;
  {
    CHECK_EQ(matvar->class_type, matio_class_map<Dtype>())
      << "Field 'data' must be of the right class (single/double) in MAT file " << filename;
    CHECK(matvar->rank < 3) << "Field '" << field_name << "' cannot have ndims > 2 in MAT file " << filename;

    m_out.create(matvar->rank > 1 ? matvar->dims[1] : 1,
            matvar->rank > 0 ? matvar->dims[0] : 0,
            cv::DataType<float>::type );

    Dtype * data = m_out.ptr< Dtype >( 0 );

    int ret = Mat_VarReadDataLinear(matfp, matvar, data, 0, 1, m_out.total());
    CHECK(ret == 0) << "Error reading array '" << field_name << "' from MAT file " << filename;
    Mat_VarFree(matvar);
  }
  Mat_Close(matfp);
  return m_out;
}

#endif // USE_OPENCV


template void ReadBlobFromMat<float>(const char*, Blob<float>*);
template void ReadBlobFromMat<double>(const char*, Blob<double>*);
template void ReadBlobFromMat<int>(const char*, Blob<int>*);
template void ReadBlobFromMat<unsigned int>(const char*, Blob<unsigned int>*);

template void WriteBlobToMat<float>(const char*, bool, Blob<float>*);
template void WriteBlobToMat<double>(const char*, bool, Blob<double>*);
template void WriteBlobToMat<int>(const char*, bool, Blob<int>*);
template void WriteBlobToMat<unsigned int>(const char*, bool, Blob<unsigned int>*);


#ifdef USE_OPENCV
template cv::Mat ReadCVMatFromMat<float>(const string & filename, const string & field_name);
template cv::Mat ReadCVMatFromMat<double>(const string & filename, const string & field_name);
#endif // USE_OPENCV
}
