#ifndef __kvlContainerPool_h
#define __kvlContainerPool_h

#include <vector>

#include "itkSmartPointer.h"

namespace kvl
{

// Recycles containers of an optimization once nothing refers to them anymore. A mesh evaluation produces one or more, all
// of the same size, with getting fresh pages from the system each time being a substantial part of the serial work.
template< typename TContainer >
class ContainerPool
{
public:
  typename TContainer::Pointer Get()
  {
    for ( const auto& container : m_Containers )
      {
      if ( container->GetReferenceCount() == 1 )
        {
        return container;
        }
      }
    m_Containers.push_back( TContainer::New() );
    return m_Containers.back();
  }

private:
  std::vector< typename TContainer::Pointer >  m_Containers;
};

} // end namespace kvl

#endif
