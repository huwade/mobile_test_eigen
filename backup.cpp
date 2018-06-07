#include <pybind11/pybind11.h>

#include "test.h"

int main()
{
    test();
}

namespace py = pybind11;

PYBIND11_MODULE(example,m)
{
  m.doc() = "pybind11 example plugin";

  m.def("main", &main);

}
