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
#include "caffe/util/matio_io.hpp"

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
template <> enum matio_classes matio_class_map<short>() { return MAT_C_INT16; }
template <> enum matio_classes matio_class_map<unsigned short>() { return MAT_C_UINT16; }

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

std::string printvec(const vector<int> & vv) {
    std::stringstream ss;
    for ( int ii = 0; ii < vv.size(); ++ii ) ss << vv[ii] << ",";
    return ss.str();
}

#ifdef USE_OPENCV
template< typename Dtype >
int read_from_mat(mat_t *matfp, matvar_t *matvar, cv::Mat & m, bool transpose, bool use_channels) {
  CHECK_EQ(matvar->class_type, matio_class_map<Dtype>()) << "MAT file field is the wrong class";

  cv::Mat m_in;
  vector< int > sizes;
  for (int ii = matvar->rank-1; ii >= 0; --ii) sizes.push_back((int) matvar->dims[ii]);
  m_in.create(matvar->rank, sizes.data(), cv::DataType<Dtype>::type);

  Dtype * data = m_in.ptr< Dtype >( 0 );
  int ret = Mat_VarReadDataLinear(matfp, matvar, data, 0, 1, m_in.total());

  if ( transpose ) {
    vector< int > sizes_t;
    for (int ii = 0; ii < matvar->rank; ++ii) sizes_t.push_back((int) matvar->dims[ii]);
    m.create(matvar->rank, sizes_t.data(), cv::DataType<Dtype>::type);
    vector< int > idx(sizes.size(), 0), idx_t(sizes_t.size(), 0);

    for (int ii = 0, jj; ii < m_in.total(); ++ii) {
//    std::cout << "copying index " << printvec(idx) << " to index_t " << printvec(idx_t) << std::endl;
      m.at<Dtype>(idx_t.data()) = m_in.at<Dtype>(idx.data());
      for (jj = idx.size()-1; jj >= 0; --jj) {
        if ( ++idx[jj] >= sizes[jj] ) idx[jj] = 0;
        else break;
      }
      for (jj = idx.size()-1; jj >= 0; --jj ) idx_t[idx.size()-jj-1] = idx[jj];
    }
  } else {
    m = m_in;
  }

  if ( use_channels && m.dims > 2 ) {
    CHECK_EQ(m.dims, 3) << "Can only use channels if mat has 3 dimensions";
    cv::Mat m_chan(m.total(), 1, m.depth());
    m_chan = m_chan.reshape(m.size[2], m.size[0]);
    CHECK_EQ(m_chan.total()*m_chan.channels(), m.total());
    for (int ii = 0, idx; ii < m.size[0]; ++ii) {
      for (int jj = 0; jj < m.size[1]; ++jj) {
        for (int kk = 0; kk < m.size[2]; ++kk){
          idx = ii * m.size[1] * m.size[2] + jj * m.size[2] + kk;
          ((Dtype*)m_chan.data)[idx] = m.at<Dtype>(ii, jj, kk);
        }
      }
    }
    m = m_chan;
  }

  return ret;
}

cv::Mat ReadCVMatFromMat(const std::string & filename, const std::string & field_name,
        bool transpose, bool use_channels) {
  cv::Mat m_out;
  mat_t *matfp;
  matfp = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
  CHECK(matfp) << "Error opening MAT file " << filename;
  // Read data
  matvar_t *matvar;
  matvar = Mat_VarReadInfo(matfp, field_name.c_str());
  CHECK(matvar) << "Field '" << field_name << "' not present in MAT file " << filename;
  {
    CHECK(matvar->rank < 4) << "Field '" << field_name << "' cannot have ndims > 3 in MAT file " << filename;
    CHECK(matvar->rank > 0) << "Field '" << field_name << "' must have ndims > 0 in MAT file " << filename;
    int ret = -1;
    switch (matvar->class_type) {
    case MAT_C_SINGLE:
        ret = read_from_mat<float>(matfp, matvar, m_out, transpose, use_channels);
        break;
    case MAT_C_DOUBLE:
        ret = read_from_mat<double>(matfp, matvar, m_out, transpose, use_channels);
        break;
    case MAT_C_INT32:
        ret = read_from_mat<int>(matfp, matvar, m_out, transpose, use_channels);
        break;
    case MAT_C_UINT32:
        ret = read_from_mat<unsigned int>(matfp, matvar, m_out, transpose, use_channels);
        break;
    case MAT_C_INT16:
        ret = read_from_mat<short>(matfp, matvar, m_out, transpose, use_channels);
        break;
    case MAT_C_UINT16:
        ret = read_from_mat<unsigned short>(matfp, matvar, m_out, transpose, use_channels);
        break;
    default:
        LOG(FATAL) << "MAT field " << field_name << " has unsupported class type " << matvar->class_type;
    }

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
template int read_from_mat<float>(mat_t *matfp, matvar_t *matvar, cv::Mat & m, bool transpose, bool use_channels);
template int read_from_mat<double>(mat_t *matfp, matvar_t *matvar, cv::Mat & m, bool transpose, bool use_channels);
template int read_from_mat<int>(mat_t *matfp, matvar_t *matvar, cv::Mat & m, bool transpose, bool use_channels);
template int read_from_mat<unsigned int>(mat_t *matfp, matvar_t *matvar, cv::Mat & m, bool transpose, bool use_channels);
template int read_from_mat<short>(mat_t *matfp, matvar_t *matvar, cv::Mat & m, bool transpose, bool use_channels);
template int read_from_mat<unsigned short>(mat_t *matfp, matvar_t *matvar, cv::Mat & m, bool transpose, bool use_channels);
#endif // USE_OPENCV
}
