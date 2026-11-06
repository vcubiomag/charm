#include "pyKvlImage.h"
#include <algorithm>
#include <limits>
#include "itkChangeInformationImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRegionOfInterestImageFilter.h"

static TransformPointer ImageToWorldTransform(const itk::ImageBase<3> *image) {
    TransformType::MatrixType scale;
    TransformType::OffsetType offset;
    for ( int i = 0; i < 3; i++ ) {
        scale[ i ][ i ] = image->GetSpacing()[ i ];
        offset[ i ] = image->GetOrigin()[ i ];
    }

    TransformPointer transform = TransformType::New();
    transform->SetMatrix( image->GetDirection() * scale );
    transform->SetOffset( offset );
    return transform;
}

template <typename T>
NumpyArray<T> KvlImage::image_to_numpy(ImagePointer image) {
    const auto size = image->GetBufferedRegion().GetSize();
    const std::array<size_t, 3> shape{size[0], size[1], size[2]};
    float *buffer = image->GetBufferPointer();
    nb::capsule owner(new ImagePointer(std::move(image)), [](void *held) noexcept { delete static_cast<ImagePointer *>(held); });
    return NumpyArray<T>(buffer, 3, shape.data(), owner, nullptr, nb::dtype<T>(), nb::device::cpu::value, 0, 'F');
}

ImagePointer KvlImage::numpy_to_image(const ImageBuffer &buffer) {
    if ( buffer.ndim() < 3 ) {
        itkGenericExceptionMacro( "image buffer must have at least three dimensions" );
    }

    ImageType::SizeType  imageSize;
    for ( int i = 0; i < 3; i++ ) {
        imageSize[ i ] = buffer.shape( i );
    }

    ImagePointer image = ImageType::New();
    image->SetRegions( imageSize );
    image->Allocate();
    std::copy_n(buffer.data(), image->GetBufferedRegion().GetNumberOfPixels(), image->GetBufferPointer());
    return image;
}

KvlImage::KvlImage(const std::string &imageFileName) {
    m_image = itk::ReadImage<ImageType>(imageFileName);

    m_transform = ImageToWorldTransform(m_image);
}

KvlImage::KvlImage(const std::string &imageFileName, const std::string &boundingFileName) {
    // Only the pixel data of the cropped region gets read from the file
    auto reader = itk::ImageFileReader<ImageType>::New();
    reader->SetFileName( imageFileName );
    reader->UpdateOutputInformation();
    const ImageType *image = reader->GetOutput();

    auto boundingReader = itk::ImageFileReader<ImageType>::New();
    boundingReader->SetFileName( boundingFileName );
    boundingReader->UpdateOutputInformation();
    const ImageType *bounding = boundingReader->GetOutput();
    const auto boundingSize = bounding->GetLargestPossibleRegion().GetSize();

    // The bounding image's voxel grid mapped into the voxel grid of the image
    TransformPointer  worldToImage = TransformType::New();
    ImageToWorldTransform(image)->GetInverse( worldToImage );
    m_transform = ImageToWorldTransform(bounding);
    m_transform->Compose( worldToImage );

    double  minimum[ 3 ];
    double  maximum[ 3 ];
    std::fill_n( minimum, 3, std::numeric_limits<double>::max() );
    std::fill_n( maximum, 3, std::numeric_limits<double>::lowest() );
    for ( int corner = 0; corner < 8; corner++ ) {
        TransformType::InputPointType  point;
        for ( int i = 0; i < 3; i++ ) {
            point[ i ] = ( corner >> i & 1 ) ? boundingSize[ i ] - 1 : 0;
        }
        const auto mapped = m_transform->TransformPoint( point );
        for ( int i = 0; i < 3; i++ ) {
            minimum[ i ] = std::min( minimum[ i ], mapped[ i ] );
            maximum[ i ] = std::max( maximum[ i ], mapped[ i ] );
        }
    }

    ImageType::RegionType  region;
    TransformType::OutputVectorType  translation;
    for ( int i = 0; i < 3; i++ ) {
        const int  start = std::max( static_cast< int >( minimum[ i ] ), 0 );
        const int  last = std::min( static_cast< long >( maximum[ i ] + 1 ),
                                    static_cast< long >( image->GetLargestPossibleRegion().GetSize()[ i ] ) - 1 );
        region.SetIndex( i, start );
        region.SetSize( i, last - start + 1 );
        translation[ i ] = -start;
        m_crop_slices.push_back( { start, last + 1 } );
    }

    auto cropper = itk::RegionOfInterestImageFilter<ImageType, ImageType>::New();
    cropper->SetInput( reader->GetOutput() );
    cropper->SetRegionOfInterest( region );
    cropper->Update();
    m_image = cropper->GetOutput();
    m_transform->Translate( translation );
}

KvlImage::KvlImage(const ImageBuffer &buffer) {
    m_image = numpy_to_image(buffer);
    m_transform = TransformType::New();
}

nb::tuple KvlImage::GetCropSlices() const {
    nb::list slices;
    for (const auto &range : m_crop_slices) {
        slices.append(nb::slice(nb::int_(range[0]), nb::int_(range[1]), nb::int_(1)));
    }
    return nb::tuple(slices);
}

KvlTransform KvlImage::GetTransform() const {
    return KvlTransform(m_transform);
}

NumpyArray<const float> KvlImage::GetImageBuffer() const {
    // A view of the image rather than a copy, so read-only
    return image_to_numpy<const float>(m_image);
}

void KvlImage::Write(const std::string &fileName, const KvlTransform &transform) const {
    ImageType::PointType   origin;
    ImageType::SpacingType  spacing;
    ImageType::DirectionType  direction;
    const auto matrix = transform.m_transform->GetMatrix();
    for ( int i = 0; i < 3; i++ )
    {
        origin[ i ] = transform.m_transform->GetOffset()[ i ];

        double  normOfColumn = 0.0;
        for ( int j = 0; j < 3; j++ )
        {
            normOfColumn += matrix[ j ][ i ] * matrix[ j ][ i ];
        }
        normOfColumn = sqrt( normOfColumn );
        spacing[ i ] = normOfColumn;
        for ( int j = 0; j < 3; j++ )
        {
            direction[ j ][ i ] = matrix[ j ][ i ] / normOfColumn;
        }
    }

    auto geometry = itk::ChangeInformationImageFilter<ImageType>::New();
    geometry->SetInput( m_image );
    geometry->SetOutputOrigin( origin );
    geometry->SetOutputSpacing( spacing );
    geometry->SetOutputDirection( direction );
    geometry->ChangeOriginOn();
    geometry->ChangeSpacingOn();
    geometry->ChangeDirectionOn();
    itk::WriteImage( geometry->GetOutput(), fileName );
}

NumpyArray<float> KvlImage::smoothImageBuffer(const ImageBuffer &imageBuffer, const std::vector<double> &sigmas)
{
    auto smoother = itk::DiscreteGaussianImageFilter<ImageType, ImageType>::New();
    // The input only gets read, so it can stay in the memory of the numpy array
    ImageType::SizeType  size;
    for ( int i = 0; i < 3; i++ ) {
        size[ i ] = imageBuffer.shape( i );
    }
    ImagePointer input = ImageType::New();
    input->SetRegions( size );
    input->GetPixelContainer()->SetImportPointer( const_cast< float * >( imageBuffer.data() ), size[ 0 ] * size[ 1 ] * size[ 2 ], false );
    smoother->SetInput( input );
    smoother->SetMaximumError( 0.1 );
    smoother->UseImageSpacingOff();
    double  variances[ 3 ];
    for ( int i = 0; i < 3; i++ )
    {
        variances[ i ] = sigmas[ i ] * sigmas[ i ];
    }
    smoother->SetVariance( variances );
    smoother->Update();
    ImagePointer smoothed = smoother->GetOutput();
    return image_to_numpy<float>(std::move(smoothed));
}
