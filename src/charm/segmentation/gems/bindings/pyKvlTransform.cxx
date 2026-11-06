#include "pyKvlTransform.h"

KvlTransform::KvlTransform(const nb::ndarray<const double, nb::shape<4, 4>, nb::f_contig> &transformMatrix) {
    m_transform = TransformType::New();
    TransformType::ParametersType  parameters( 12 );
    const double* const data = transformMatrix.data();

    for ( unsigned int row = 0; row < 3; row++ )
    {
        for ( unsigned int col = 0; col < 3; col++ )
        {
            parameters[ row * 3 + col ] = data[ col * 4 + row ];
        }
        parameters[ 9 + row ] = data[ 12 + row ];
    }
    m_transform->SetParameters( parameters );
}

NumpyArray<double> KvlTransform::AsNumpyArray() const {
    return transformToNumpy(m_transform->GetMatrix(), m_transform->GetOffset());
}
