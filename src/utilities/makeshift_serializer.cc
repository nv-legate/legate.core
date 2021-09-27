#include "utilities/makeshift_serializer.h"

namespace legate{

    void MakeshiftSerializer::packScalar(const Scalar& scalar){
        pack((bool) scalar.is_tuple()); 
        pack((LegateTypeCode) scalar.code_); 
        int32_t size = scalar.size();
        packWithoutType(scalar.data_, size);    
    }

    void MakeshiftSerializer::packBuffer(const Store& buffer)
    {
        pack((bool) buffer.is_future()); //is_future
        pack((int32_t) buffer.dim());
        //int32_t code = buffer.code();
        pack((int32_t)  buffer.code());
        //pack transform:
        //pack trasnform code
        int32_t neg= -1;
        pack((int32_t) neg);
        //skip the rest for now, assume no transform, for now pack -1
        // no need to implement this for benchmarking purposes 
        // TODO: implement transform packing
        // TODO: add "code" to transform object
        //if _isfuture
        if(buffer.is_future_)
        {   
            //pack future_wrapper
        }   
        //elif dim>=0
        else if (buffer.dim()>=0){
            pack((int32_t) buffer.redop_id_);
            //pack reigon field
                //pack dim
                pack((int32_t) buffer.region_field_.dim()); 
                //pack idx (req idx) //need to map regions to idx
                pack((uint32_t) buffer.region_field_.reqIdx_); 
                //pack fid (field id)
                pack((int32_t) buffer.region_field_.fid_); 
        }
        else
        {   
            //pack redop_id
            pack((int32_t) buffer.redop_id_);
            //pack reigon field
                //pack dim; always 1 in an buffer
                pack((int32_t) 1); 
                //pack idx (req idx) //need to map regions to idx
                pack((uint32_t) buffer.region_field_.reqIdx_); 
                //pack fid (field id)
                pack((int32_t) buffer.region_field_.fid_); 
        }   
   }



}
