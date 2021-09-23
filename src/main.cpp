#include "eigen/Dense"
#include "midifile/MidiFile.h"
#include "rtmidi/RtMidi.h"
#include "json/json.hpp"
#include <iostream>
int main() {
  Eigen::MatrixXd m(2, 2);
  m(0, 0) = 3, m(1, 0) = 2.5, m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);
  std::cout << m << std::endl;
  return 0;
}
