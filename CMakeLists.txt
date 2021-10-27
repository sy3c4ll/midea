cmake_minimum_required(VERSION 3.5.0)

set(PROJNM midea)
set(SRCDIR src)
set(SRC ${SRCDIR}/midea.cpp)
set(LIBDIR lib)
set(LIB eigen json midifile rtmidi)
set(LIB eigen json midifile)

project(${PROJNM})
add_executable(${PROJNM} ${SRC})

foreach(LIBRARY ${LIB})
  add_subdirectory(${LIBDIR}/${LIBRARY})
endforeach(LIBRARY)
target_link_libraries(${PROJNM} ${LIB})
