#include "eigen/Dense"
#include "midifile/MidiFile.h"
#include "rtmidi/RtMidi.h"
#include "json/json.hpp"
#include <cstring>
#include <iostream>

#define MAX_PATH_LEN 0xff
#define LAYER_NUM 2
#define IO_SIZE 4
#define H_SIZE 6
#define EPOCH_SAMPLE_PERIOD 1
#define SAMPLE_SIZE 10
#define NOTE_ON 0x90
#define T_MIN 0
#define T_MAX 100000
#define N_MIN 0x00
#define N_MAX 0xff
#define A_MIN 0x01
#define A_MAX 0xff
#define DATA_DIR "./data/"
#define METADATA_FILE "maestro-v3.0.0.json"
#define TPQ 960
#define BPM 120

#define min(x, y) ((x) < (y) ? (x) : (y))
#define max(x, y) ((x) > (y) ? (x) : (y))
#define round(x) ((int)((x) + 0.5))

Eigen::MatrixXd h_swish(Eigen::MatrixXd m);
Eigen::VectorXd run(Eigen::VectorXd *M, Eigen::MatrixXd *W, Eigen::VectorXd *b,
                    Eigen::VectorXd *x);
void train(Eigen::VectorXd *M, Eigen::MatrixXd *W, Eigen::VectorXd *b,
           smf::MidiFile midi);
smf::MidiFile sample(Eigen::VectorXd *M, Eigen::MatrixXd *W, Eigen::VectorXd *b,
                     Eigen::VectorXd *seq, int seqsize);

int main(int argc, char **argv) {

#ifdef LOGFILE
  std::cout << "[*] LOGFILE defined, redirecting output to " << LOGFILE
            << std::endl;
  freopen(LOGFILE, "w", stdout);
  freopen(LOGFILE, "w", stderr);
#endif /* LOGFILE */

  Eigen::VectorXd M[LAYER_NUM] = {};
  Eigen::MatrixXd W[LAYER_NUM + 1] = {};
  Eigen::VectorXd b[LAYER_NUM + 1] = {};
  size_t batch = 0;
  for (size_t i = 0; i < LAYER_NUM; i++)
    M[i] = Eigen::VectorXd::Random(H_SIZE);
  for (size_t i = 1; i < LAYER_NUM; i++)
    W[i] = Eigen::MatrixXd::Random(H_SIZE, H_SIZE);
  W[0] = Eigen::MatrixXd::Random(H_SIZE, IO_SIZE);
  W[LAYER_NUM] = Eigen::MatrixXd::Random(IO_SIZE, H_SIZE);
  for (size_t i = 0; i < LAYER_NUM; i++)
    b[i] = Eigen::VectorXd::Random(H_SIZE);
  b[LAYER_NUM] = Eigen::VectorXd::Random(IO_SIZE);
  std::cout
      << "[*] Model layers, weights and biases initialised with random values"
      << std::endl;
  char path[MAX_PATH_LEN] = "";
  strcpy(path, DATA_DIR);
  strcat(path, METADATA_FILE);
  nlohmann::json md = nlohmann::json::parse(std::ifstream(path));
  std::cout << "[*] Metadata parsed from " << path << std::endl;
  for (size_t epoch = 0;; epoch++) {
    for (const auto &midi_data : md["midi_filename"].items())
      if (strcmp(md["split"][midi_data.key()].get<std::string>().c_str(),
                 "train") == 0) {
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
        train(M, W, b, midi);
        break;
      }
    if (epoch % EPOCH_SAMPLE_PERIOD == 0) {
      std::cout << "[*] Generating MIDI for epoch " << epoch << "..."
                << std::endl;
      strcpy(path, "gen.mid");
      sample(M, W, b, {}, 0).write(path);
      std::cout << "[*] MIDI generated and saved to " << path << std::endl;
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

Eigen::VectorXd run(Eigen::VectorXd *M, Eigen::MatrixXd *W, Eigen::VectorXd *b,
                    Eigen::VectorXd x) {
  M[0] += h_swish(W[0] * x + b[0]);
  for (size_t i = 1; i < LAYER_NUM; i++) {
    M[i] += h_swish(W[i] * M[i - 1] + b[i]);
  }
  x = h_swish(W[LAYER_NUM] * M[LAYER_NUM - 1] + b[LAYER_NUM]);
  return x;
}

void train(Eigen::VectorXd *M, Eigen::MatrixXd *W, Eigen::VectorXd *b,
           smf::MidiFile midi) {
  if (!midi.status()) {
    std::cerr << "[!] train: Unable to process MIDI data from arguments"
              << std::endl;
    return;
  }
  midi.absoluteTicks();
  midi.joinTracks();
  midi.doTimeAnalysis();
  midi.linkNotePairs();
  Eigen::VectorXd x(IO_SIZE);
  double psec = 0;
  bool init = false;
  for (size_t i = 0; i < (size_t)midi[0].size(); i++)
    if (midi[0][i].isNoteOn()) {
      if (init) {
        x = run(M, W, b, x);
        // TODO: calculate loss and backpropagate
      }
      x << midi[0][i].seconds - psec, midi[0][i].getDurationInSeconds(),
          midi[0][i][1], midi[0][i][2];
      psec = midi[0][i].seconds;
      init = true;
    }
  return;
}

smf::MidiFile sample(Eigen::VectorXd *M, Eigen::MatrixXd *W, Eigen::VectorXd *b,
                     Eigen::VectorXd *seq, int seqsize) {
  smf::MidiFile midi;
  std::vector<uint8_t> note{NOTE_ON, 0, 0};
  Eigen::VectorXd x(IO_SIZE);
  double atime = 0;
  midi.absoluteTicks();
  midi.setTPQ(TPQ);
  midi.addTrack(1);
  if (seqsize) {
    for (size_t i = 0; i < (size_t)seqsize; i++) {
      note[1] = seq[i](2), note[2] = seq[i](3);
      midi.addEvent(1, (int)((atime + seq[i](0)) * TPQ * BPM / 60), note);
      note[2] = 0;
      midi.addEvent(1, (int)((atime + seq[i](0) + seq[i](1)) * TPQ * BPM / 60),
                    note);
      std::cout << "[*] generate: Recorded pre-entered note {" << (int)note[1]
                << "," << (int)note[2] << "} ON to "
                << (int)((atime + seq[i](0)) * TPQ * BPM / 60) << " and OFF to "
                << (int)((atime + seq[i](0) + seq[i](1)) * TPQ * BPM / 60)
                << std::endl;
      atime += seq[i](0);
    }
    x = seq[seqsize - 1];
  } else {
    x << 0.5, 0.5, 60, 64;
    note[1] = x(2), note[2] = x(3);
    midi.addEvent(1, (int)((atime + x(0)) * TPQ * BPM / 60), note);
    note[2] = 0;
    midi.addEvent(1, (int)((atime + x(0) + x(1)) * TPQ * BPM / 60), note);
    std::cout << "[*] generate: Recorded pre-entered note {" << (int)note[1]
              << "," << (int)note[2] << "} ON to "
              << (int)((atime + x(0)) * TPQ * BPM / 60) << " and OFF to "
              << (int)((atime + x(0) + x(1)) * TPQ * BPM / 60) << std::endl;
    atime += x(0);
  }
  for (size_t i = 0; i < SAMPLE_SIZE; i++) {
    x = run(M, W, b, x);
    x << min(max(x(0), T_MIN), T_MAX), min(max(x(1), T_MIN), T_MAX),
        round(min(max(x(2), N_MIN), N_MAX)),
        round(min(max(x(3), A_MIN), A_MAX));
    note[1] = x(2), note[2] = x(3);
    midi.addEvent(1, (int)((atime + x(0)) * TPQ * BPM / 60), note);
    note[2] = 0;
    midi.addEvent(1, (int)((atime + x(0) + x(1)) * TPQ * BPM / 60), note);
    std::cout << "[*] generate: Recorded generated note {" << (int)note[1]
              << "," << (int)note[2] << "} ON to "
              << (int)((atime + x(0)) * TPQ * BPM / 60) << " and OFF to "
              << (int)((atime + x(0) + x(1)) * TPQ * BPM / 60) << std::endl;
    atime += x(0);
  }
  midi.sortTracks();
  return midi;
}
