#include "eigen/Dense"
#include "midifile/MidiFile.h"
#include "rtmidi/RtMidi.h"
#include "json/json.hpp"
#include <cmath>
#include <cstring>
#include <iostream>

#define MAX_PATH_LEN 0xff
#define LAYER_NUM 3
#define IO_SIZE 4
#define H_SIZE 6
#define T_MIN 0x00
#define T_MAX 0xff
#define A_MIN 0x01
#define A_MAX 0xff
#define DATA_DIR "./data/"
#define METADATA_FILE "maestro-v3.0.0.json"
#define TPQ 960
#define BPM 120

#define min(x, y) ((x) < (y) ? (x) : (y))
#define max(x, y) ((x) > (y) ? (x) : (y))

Eigen::MatrixXd h_swish(Eigen::MatrixXd m);
void train(Eigen::MatrixXd *W, Eigen::VectorXd *b, smf::MidiFile midi);
smf::MidiFile generate(Eigen::MatrixXd *W, Eigen::VectorXd *b);

int main(int argc, char **argv) {

#ifdef LOGFILE
  std::cout << "[*] LOGFILE defined, redirecting output to " << LOGFILE
            << std::endl;
  freopen(LOGFILE, "w", stdout);
  freopen(LOGFILE, "w", stderr);
#endif /* LOGFILE */

  Eigen::MatrixXd W[LAYER_NUM] = {};
  Eigen::VectorXd b[LAYER_NUM] = {};
  size_t batch = 0;
  for (size_t i = 1; i < LAYER_NUM - 1; i++)
    W[i] = Eigen::MatrixXd::Random(H_SIZE, H_SIZE);
  W[0] = Eigen::MatrixXd::Random(H_SIZE, IO_SIZE);
  W[LAYER_NUM - 1] = Eigen::MatrixXd::Random(IO_SIZE, H_SIZE);
  for (size_t i = 0; i < LAYER_NUM - 1; i++)
    b[i] = Eigen::VectorXd::Random(H_SIZE);
  b[LAYER_NUM - 1] = Eigen::VectorXd::Random(IO_SIZE);
  std::cout << "[*] Model weights and biases initialised" << std::endl;
  char path[MAX_PATH_LEN] = "";
  strcpy(path, DATA_DIR);
  strcat(path, METADATA_FILE);
  nlohmann::json md = nlohmann::json::parse(std::ifstream(path));
  std::cout << "[*] JSON parsed from " << path << std::endl;
  for (size_t epoch = 0;; epoch++) {
    for (const auto &midi_data : md["midi_filename"].items()) {
      batch++;
      strcpy(path, DATA_DIR);
      strcat(path, midi_data.value().get<std::string>().c_str());
      std::cout << "[*] Epoch " << epoch << " batch " << batch << ": " << path
                << std::endl;
      smf::MidiFile midi(path);
      if (!midi.status()) {
        std::cerr << "[!] Unable to open MIDI file from path " << path
                  << ", skipping batch " << batch << " file " << path << "..."
                  << std::endl;
        continue;
      }
      train(W, b, midi);
      break;
    }
    if (epoch % 1 == 0) {
      std::cout << "[*] Generating MIDI for epoch " << epoch << "..."
                << std::endl;
      generate(W, b).write("gen.mid");
      std::cout << "[*] MIDI generated and saved to gen.mid" << std::endl;
      break;
    }
  }
  return 0;
}

Eigen::MatrixXd h_swish(Eigen::MatrixXd m) {
  for (size_t i = 0; i < (size_t)m.rows(); i++)
    for (size_t j = 0; j < (size_t)m.cols(); j++)
      m(i, j) = m(i, j) * min(max(m(i, j) + 3, 0), 6) / 6;
  return m;
}

Eigen::VectorXd run(Eigen::MatrixXd *W, Eigen::VectorXd *b, Eigen::VectorXd x) {
  for (size_t j = 0; j < LAYER_NUM; j++)
    x = h_swish(W[j] * x + b[j]);
  x(0) = max(x(0), 0), x(1) = max(x(1), 0),
  x(2) = round(min(max(x(2), T_MIN), T_MAX)),
  x(3) = round(min(max(x(3), A_MIN), A_MAX));
  return x;
}

void train(Eigen::MatrixXd *W, Eigen::VectorXd *b, smf::MidiFile midi) {
  if (!midi.status()) {
    std::cerr << "[!] train: Unable to process MIDI data from arguments"
              << std::endl;
    return;
  }
  midi.absoluteTicks();
  midi.joinTracks();
  midi.doTimeAnalysis();
  midi.linkNotePairs();
  Eigen::VectorXd x(4);
  double psec = 0;
  bool init = false;
  for (size_t i = 0; i < (size_t)midi[0].size(); i++)
    if (midi[0][i].isNoteOn()) {
      if (init) {
        x = run(W, b, x);
        // TODO: calculate loss and backpropagate
      }
      x << midi[0][i].seconds - psec, midi[0][i].getDurationInSeconds(),
          midi[0][i][1], midi[0][i][2];
      psec = midi[0][i].seconds;
      init = true;
    }
  return;
}

smf::MidiFile generate(Eigen::MatrixXd *W, Eigen::VectorXd *b) {
  // TODO: starting and finishing sequences
  smf::MidiFile midi;
  Eigen::VectorXd x(4);
  x << 0, 1, 60, 64;
  std::vector<uint8_t> note{0x90, 0, 0};
  double atime = 0;
  midi.absoluteTicks();
  midi.setTPQ(TPQ);
  midi.addTrack(1);
  for (size_t i = 0; i < 10; i++) {
    note[1] = x(2), note[2] = x(3);
    midi.addEvent(1, (int)((atime + x(0)) * TPQ * BPM / 60), note);
    note[2] = 0;
    midi.addEvent(1, (int)((atime + x(0) + x(1)) * TPQ * BPM / 60), note);
    atime += x(0);
    std::cout << "[*] generate: Recorded note {" << (int)note[1] << ","
              << (int)note[2] << "} ON to "
              << (int)(atime + x(0)) * TPQ * BPM / 60 << " and OFF to "
              << (int)(atime + x(0) + x(1)) * TPQ * BPM / 60 << std::endl;
    x = run(W, b, x);
  }
  midi.sortTracks();
  return midi;
}
