#include "utilities/makeshift_serializer.h"

namespace legate{

    void MakeshiftSerializer::packScalar(const Scalar& scalar){
        pack((bool) scalar.is_tuple()); 
        pack((LegateTypeCode) scalar.code_); 
        int32_t size = scalar.size();
        packWithoutType(scalar.data_, size);    
    }

    void MakeshiftSerializer::packTransform(const StoreTransform* trans){

        if (trans==nullptr){
            int32_t neg= -1;
            pack((int32_t) neg);
        }
        else{
            int32_t code = trans->getTransformCode();
            switch (code) {
                case -1: {
                }
                case LEGATE_CORE_TRANSFORM_SHIFT: {
                    Shift * shifter = (Shift*) trans;
                    pack((int32_t) shifter->dim_);
                    pack((int64_t) shifter->offset_);
                    packTransform(trans->parent_.get());
                }
                case LEGATE_CORE_TRANSFORM_PROMOTE: {
                    Promote * promoter = (Promote*) trans;
                    pack((int32_t) promoter->extra_dim_);
                    pack((int64_t) promoter->dim_size_);
                    packTransform(trans->parent_.get());
                }
                case LEGATE_CORE_TRANSFORM_PROJECT: {
                    Project * projector = (Project*) trans;
                    pack((int32_t) projector->dim_);
                    pack((int64_t) projector->coord_);
                    packTransform(trans->parent_.get());
                }
                case LEGATE_CORE_TRANSFORM_TRANSPOSE: {
                    Transpose * projector = (Transpose*) trans;
                    packTransform(trans->parent_.get());
                }
                case LEGATE_CORE_TRANSFORM_DELINEARIZE: {
                    Delinearize * projector = (Delinearize*) trans;
                    packTransform(trans->parent_.get());
                }
            }
        }
    }
/*
    case LEGATE_CORE_TRANSFORM_SHIFT: {
      auto dim    = unpack<int32_t>();
      auto offset = unpack<int64_t>();
      auto parent = unpack_transform();
      return std::make_unique<Shift>(dim, offset, std::move(parent));
    }
    case LEGATE_CORE_TRANSFORM_PROMOTE: {
      auto extra_dim = unpack<int32_t>();
      auto dim_size  = unpack<int64_t>();
      auto parent    = unpack_transform();
      return std::make_unique<Promote>(extra_dim, dim_size, std::move(parent));
    }
    case LEGATE_CORE_TRANSFORM_PROJECT: {
      auto dim    = unpack<int32_t>();
      auto coord  = unpack<int64_t>();
      auto parent = unpack_transform();
      return std::make_unique<Project>(dim, coord, std::move(parent));
    }
    case LEGATE_CORE_TRANSFORM_TRANSPOSE: {
      auto axes   = unpack<std::vector<int32_t>>();
      auto parent = unpack_transform();
      return std::make_unique<Transpose>(std::move(axes), std::move(parent));
    }
    case LEGATE_CORE_TRANSFORM_DELINEARIZE: {
      auto dim    = unpack<int32_t>();
      auto sizes  = unpack<std::vector<int64_t>>();
      auto parent = unpack_transform();
      return std::make_unique<Delinearize>(dim, std::move(sizes), std::move(parent));
    }

    def _serialize_transform(self, buf):
        if self._parent is not None:
            self._transform.serialize(buf)
            self._parent._serialize_transform(buf)
        else:
            buf.pack_32bit_int(-1)
*/
    void MakeshiftSerializer::packBuffer(const Store& buffer)
    {
        pack((bool) buffer.is_future()); //is_future
        pack((int32_t) buffer.dim());
        //int32_t code = buffer.code();
        pack((int32_t)  buffer.code());
        //pack transform:
        //pack trasnform code
        packTransform(buffer.transform_.get());

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
