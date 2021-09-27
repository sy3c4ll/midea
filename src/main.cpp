#include "eigen/Dense"
#include "midifile/MidiFile.h"
#include "rtmidi/RtMidi.h"
#include "json/json.hpp"
#include <cstring>
#include <iostream>

#define DEBUG
#define NULLDEVICE "/dev/null"
#define DATA_DIR "./data/"
#define METADATA_FILE "maestro-v3.0.0.json"

int main(int argc, char **argv) {

#ifndef DEBUG
  freopen(NULLDEVICE, "w", stdout);
  freopen(NULLDEVICE, "w", stderr);
#endif /* DEBUG */

  char path[0xff] = "";
  strcpy(path, DATA_DIR);
  strcat(path, METADATA_FILE);
  nlohmann::json md = nlohmann::json::parse(std::ifstream(path));
  for (const auto &midi_data : md["midi_filename"].items()) {
    strcpy(path, DATA_DIR);
    strcat(path, midi_data.value().get<std::string>().c_str());
    smf::MidiFile midi(path);
    std::cout << midi.status();
  }
  return 0;
}
