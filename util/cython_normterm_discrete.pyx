# distutils: language=c++
# distutils: extra_compile_args = ["-O3"]
# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True

from libcpp cimport bool
#from libcpp.unordered_map cimport unordered_map
from scipy.special.cython_special cimport loggamma
from libc.math cimport exp, log
from libc.stdio cimport printf 

cdef extern from *:
    """
    #include <iostream>
    #include <unordered_map>
    #include <queue>
 
    // https://stackoverflow.com/questions/32685540/why-cant-i-compile-an-unordered-map-with-a-pair-as-key
    struct pair_hash {
        template <class T1, class T2>
        std::size_t operator () (const std::pair<T1,T2> &p) const {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);

            // Mainly for demonstration purposes, i.e. works but is overly simple
            // In the real world, use sth. like boost.hash_combine
            return h1 ^ h2;  
        }
    };

    struct memoization_result{
       double value;
       bool found;
    };

    class memoization{
        private:
           std::unordered_map<std::pair<long long, long long>, double, pair_hash> map;
           std::queue<std::pair<long long, long long>> key_order;
           //size_t max_size;
           long long max_size;

        public:                           
           memoization(): max_size(18446744073709551615){}
           void set_value(long long n, long long k, double val){
                //assumes key isn't yet in map
                //map[std::make_pair(n, k)] = val;
                std::pair<long long, long long> key(n, k);
                map[key] = val;
                //key_order.push(std::make_pair(n, k));
                key_order.push(key);
                if (key_order.size() > max_size) {
                    key_order.pop();
                }
           }
           memoization_result find_value(long long n, long long k) const{
              //auto it = map.find(std::make_pair(n, k));
              //std::cout << "try to find (" << n << ", " << k << ")" << std::endl;
              auto it = map.find(std::make_pair(n, k));
              if (it == map.cend()) {
                  return {0, false};
              }
              else {
                  //std::cout << "map.size: " << map.size() << std::endl;
                  //std::cout << "(" << it->first.first << "," << it->first.second << "): " << it->second << std::endl;

                  return {it->second, true};
              }
           }      
    };
    """
    struct memoization_result:
        double value;
        bool found;

    cppclass memoization:
        memoization()
        void set_value(long long, long long, double)
        memoization_result find_value(long long, long long)


ctypedef double(*f_type)(long long, long long)
cdef double id_fun(long long n, long long k):
    return <double>(n*k)

#cdef class FunWithMemoization:
cdef class FunWithMem:
    cdef memoization mem
    cdef f_type fun
    def __cinit__(self):
        self.fun = id_fun

    cpdef double evaluate(self, long long n, long long k):
        cdef memoization_result look_up = self.mem.find_value(n, k)
        if look_up.found:
            #printf('found: n=%d, k=%d\n\n', n, k)
            return look_up.value
        cdef double val = self.fun(n, k)
        self.mem.set_value(n, k, val)
        return val

cdef memoization mem_each
    
import time
cdef double normterm_discrete_cython(long long n, long long k):
    cdef memoization_result look_up = mem_each.find_value(n, k)
    if look_up.found:
        return look_up.value
    
    cdef double res
    if n == 1:
        res = log(k)
    elif k == 1:
        res = 1.0
    elif k == 2:
        res = 0.0
        for t in range(1, n):
            res += exp(loggamma(<double>(n+1)) - loggamma(<double>(t+1)) - loggamma(<double>(n-t+1)) + <double>t*(log(<double>t) - log(<double>n)) + <double>(n-t)*(log(n-t) - log(n) ) )
        #res = sum([exp(loggamma(double(n+1)) - loggamma(double(t+1)) - loggamma(double(n-t+1)) + double(t)*(log(double(t)) - log(double(n)) + double(n-t)*(log(double(n-t)) - log(double(n)) ) for t in range(1, n)])
    else:
        look_up = mem_each.find_value(n, k-1)
        if look_up.found:
            pre = look_up.value
        else:
            pre = normterm_discrete_cython(n, k-1)
            mem_each.set_value(n, k-1, pre)
        
        look_up = mem_each.find_value(n, k-2)
        if look_up.found:
            pre2 = look_up.value
        else:
            pre2 = normterm_discrete_cython(n, k-2)
            mem_each.set_value(n, k-2, pre2)
            
        #res = normterm_discrete_cython(n, k-1) + <double>n/(k-2) * normterm_discrete_cython(n, k-2)
        res = pre + <double>n/(k-2) * pre2
    
    mem_each.set_value(n, k, res)
    
    return res

def create_fun_with_mem():
    f = FunWithMem()
    f.fun = normterm_discrete_cython
    return f