#ifndef GEMS_PYKVLTRANSFORM_H
#define GEMS_PYKVLTRANSFORM_H

#include "itkAffineTransform.h"
#include "pyKvlNumpy.h"

using TransformType = itk::AffineTransform< double, 3 >;
using TransformPointer = TransformType::Pointer;

class KvlTransform {
public:
    KvlTransform(const nb::ndarray<const double, nb::shape<4, 4>, nb::f_contig> &transformMatrix);
    NumpyArray<double> AsNumpyArray() const;

    TransformPointer m_transform;

    KvlTransform(TransformPointer transform) : m_transform(transform) {};
    const TransformPointer GetTransform() const {
        return m_transform;
    }

};
#endif //GEMS_PYKVLTRANSFORM_H
