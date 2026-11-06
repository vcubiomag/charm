#ifndef GEMS_KVLIMAGE_H
#define GEMS_KVLIMAGE_H

#include <array>
#include <string>
#include <vector>
#include "itkImage.h"
#include "pyKvlNumpy.h"
#include "pyKvlTransform.h"

using ImageType = itk::Image< float, 3 >;
using ImagePointer = ImageType::Pointer;

class KvlImage {
public:
    ImagePointer m_image;
    TransformPointer m_transform;
    std::vector<std::array<int, 2>> m_crop_slices;

    KvlImage(const std::string &imageFileName);
    KvlImage(const std::string &imageFileName, const std::string &boundingFileName);
    KvlImage(const ImageBuffer &buffer);
    KvlTransform GetTransform() const;
    NumpyArray<const float> GetImageBuffer() const;
    nb::tuple GetCropSlices() const;
    void Write(const std::string &fileName, const KvlTransform &transform) const;
    static NumpyArray<float> smoothImageBuffer(const ImageBuffer &imageBuffer, const std::vector<double> &sigmas);

private:
    static ImagePointer numpy_to_image(const ImageBuffer &buffer);
    template <typename T>
    static NumpyArray<T> image_to_numpy(ImagePointer image);
};

#endif //GEMS_KVLIMAGE_H
