
#pragma once
#include <iostream>
#include <vector>
#include "data/store.h"
#include "data/scalar.h"
#include "data/transform.h"

namespace legate {

class Scalar;
class Store;
class MakeshiftSerializer{
    
    public:
    MakeshiftSerializer(){
        size=512;
        raw.resize(size); 
        write_offset=0;
        read_offset=0;
    }
    void zero(){
        //memset ((void*)raw.data(),0,raw.size());
        write_offset=0;
    }
/*
    template <typename T> void pack(T&& arg) 
    {
        T copy = arg;
        pack(copy); //call l-value version
    }
*/
    template <typename T> void pack(T arg) 
    {
        int8_t * argAddr = (int8_t*) &arg;
        //std::cout<<arg<<std::endl;
        if (size<=write_offset+sizeof(T))
        {
            resize(sizeof(T));
        }
        //for (int i=0; i<sizeof(T); i++)
        //{
        //   raw[write_offset+i] = *reinterpret_cast<const int8_t*>((argAddr)+i);
        //}
        memcpy(raw.data()+write_offset, argAddr, sizeof(T));
        //std::cout<<"reint "<<*reinterpret_cast<T*>(raw.data()+write_offset)<<std::endl;;
        write_offset+=sizeof(T);
        //std::cout<<"    "<<write_offset<<std::endl;
    }
 
    void packWithoutType(const void* arg, int argSize) 
    {
        const int8_t* argByte =(int8_t*) arg;
        //std::cout<<"data of size: "<<argSize<<std::endl;
        if (size<=write_offset+argSize)
        {
            resize(argSize);
        }
        for (int i=0; i<argSize; i++){
            raw[write_offset+i] = *reinterpret_cast<const int8_t*>(argByte+i);
        }
        write_offset+=argSize;
        //std::cout<<"    "<<write_offset<<std::endl;
    }

    void packScalar(const Scalar& scalar);

    void packBuffer(const Store& input);

    void packTransform(const StoreTransform* trans);
    
    template <typename T> T read() 
    {
        if (read_offset<write_offset)
        {
            T datum = *reinterpret_cast<T*>(raw.data()+read_offset);
            read_offset+=sizeof(T);
            return datum;
        }
        else{
            std::cout<<"finished reading buffer"<<std::endl;
            return NULL;
        }
    }

    void resize(size_t argSize){
        while(size<=write_offset+argSize)
        {
            //std::cout<<"resizing from "<<size<<" to "<<2*size<<std::endl; 
            size=2*size;
            raw.resize(size);
        }
    }

    void reset_reader(){
        read_offset=0;
    }

    int8_t* ptr(){
        return raw.data();
    }

    int buffSize(){
        return write_offset;
    }
    private: 
    size_t size;
    int read_offset;
    int write_offset;
    std::vector<int8_t> raw;
};
/*
int main(){
    MakeshiftSerializer ms;
    int a=3; 
    char g='a'; 
    ms.pack<int>(a);
    ms.pack<char>(g);
    ms.pack<int>(a);
    ms.pack<char>(g);
    std::cout<<ms.read<int>()<<std::endl;;
    std::cout<<ms.read<char>()<<std::endl;;
    std::cout<<ms.read<int>()<<std::endl;;
    std::cout<<ms.read<char>()<<std::endl;;
    std::cout<<ms.read<int>()<<std::endl;;
    ms.reset_reader();
    std::cout<<ms.read<int>()<<std::endl;;
     
}*/
}
