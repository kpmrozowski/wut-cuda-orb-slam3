#ifndef MSI_CARCASSONNE_TIME_H
#define MSI_CARCASSONNE_TIME_H
#include <chrono>

namespace util {

inline uint64_t unix_time() {
   return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

}

#endif//MSI_CARCASSONNE_TIME_H
