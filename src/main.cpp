#include "eigen/Dense"
#include "midifile/MidiFile.h"
#include "rtmidi/RtMidi.h"
#include "json/json.hpp"
#include <cstring>
#include <iostream>

#ifdef __unix__
#include <unistd.h>
#define NULLDEVICE "/dev/null"
#elif defined _WIN64
#include <windows.h>
#define sleep(x) Sleep((x)*1000)
#define NULLDEVICE "nul"
#endif

#define LAYER_NUM 2
#define IO_SIZE 4
#define H_SIZE 6
#define EPOCH_NUM 2
#define MINI_BATCH_SIZE 100
#define EPOCH_SAMPLE_PERIOD 1
#define SAMPLE_SIZE 10
#define ACT_FUNC h_swish
#define NOTE_ON 0x90
#define T_MIN 0
#define T_MAX 100000
#define N_MIN 0x00
#define N_MAX 0x7f
#define A_MIN 0x01
#define A_MAX 0x7f
#define MAX_PATH_LEN 0xff
#define MAX_FILE_NUM 5000
#define DATA_DIR "./data/"
#define METADATA_FILE "maestro-v3.0.0.json"
#define MIDI_OUT 1
#define TPQ 960
#define BPM 120

typedef struct model {
  Eigen::MatrixXd wxy[LAYER_NUM + 1], waa[LAYER_NUM];
  Eigen::VectorXd b[LAYER_NUM + 1], a[LAYER_NUM];
} model;

inline double min(double x, double y) { return x < y ? x : y; }
inline double max(double x, double y) { return x > y ? x : y; }
inline double round(double x) { return (int)(x + 0.5); }
inline double h_swish(double x) { return x * min(max(x + 3, 0), 6) / 6; }

Eigen::MatrixXd map(Eigen::MatrixXd M, void *func);
Eigen::VectorXd run(model *m, Eigen::VectorXd x);
double train(model *m, smf::MidiFile *midi, size_t batch_size);
smf::MidiFile sample(model *m, Eigen::VectorXd *seq, int seqsize);
void play_midi(RtMidiOut *midiout, smf::MidiFile midi);

int main(int argc, char **argv) {

#ifdef LOGFILE
  std::cout << "[*] LOGFILE defined, redirecting output to " << LOGFILE
            << std::endl;
  freopen(LOGFILE, "w", stdout);
  freopen(LOGFILE, "w", stderr);
#endif /* LOGFILE */

  model *m = new model;

  for (size_t i = 0; i < LAYER_NUM; i++)
    m->a[i] = Eigen::VectorXd::Random(H_SIZE);

  for (size_t i = 1; i < LAYER_NUM; i++)
    m->wxy[i] = Eigen::MatrixXd::Random(H_SIZE, H_SIZE);
  m->wxy[0] = Eigen::MatrixXd::Random(H_SIZE, IO_SIZE);
  m->wxy[LAYER_NUM] = Eigen::MatrixXd::Random(IO_SIZE, H_SIZE);

  for (size_t i = 0; i < LAYER_NUM; i++)
    m->waa[i] = Eigen::MatrixXd::Random(H_SIZE, H_SIZE);

  for (size_t i = 0; i < LAYER_NUM; i++)
    m->b[i] = Eigen::VectorXd::Random(H_SIZE);
  m->b[LAYER_NUM] = Eigen::VectorXd::Random(IO_SIZE);

  std::cout
      << "[*] Model layers, weights and biases initialised with random values"
      << std::endl;

  char path[MAX_PATH_LEN] = "";

  strcpy(path, DATA_DIR);
  strcat(path, METADATA_FILE);
  nlohmann::json md = nlohmann::json::parse(std::ifstream(path));
  std::cout << "[*] Metadata parsed from " << path << std::endl;

  smf::MidiFile midi[MAX_FILE_NUM] = {}, generated;
  size_t batch_num = 0;

  for (nlohmann::json::reference midi_data : md["midi_filename"]) {

    strcpy(path, DATA_DIR);
    strcat(path, midi_data.get<std::string>().c_str());
    midi[batch_num].read(path);

    if (!midi[batch_num].status()) {
      std::cerr << "[!] Unable to open MIDI file from path " << path
                << std::endl;
      continue;
    }

    batch_num++;
  }
  std::cout << "[*] " << batch_num << " MIDI files successfully loaded"
            << std::endl;

  RtMidiOut *midiout = NULL;

  try {
    midiout = new RtMidiOut();
    midiout->openPort(MIDI_OUT);
  } catch (RtMidiError &e) {
    std::cerr << "[!] Unable to open MIDI output ports, writing to file instead"
              << std::endl
              << "    Details:" << std::endl;
    e.printMessage();
    delete midiout;
    midiout = NULL;
  }
  if (!midiout->isPortOpen()) {
    std::cerr << "[!] Unable to open MIDI output ports, writing to file instead"
              << std::endl;
    delete midiout;
    midiout = NULL;
  }
  std::cout << "[*] Connected to MIDI output port " << MIDI_OUT << std::endl;

  double loss = 0;

  for (size_t epoch = 0; epoch < EPOCH_NUM; epoch++) {

    loss = train(m, midi, batch_num);
    std::cout << "[*] Epoch " << epoch + 1 << "/" << EPOCH_NUM << " : Loss "
              << loss << std::endl;

    if (epoch != 0 && epoch % EPOCH_SAMPLE_PERIOD == 0) {
      std::cout << "[*] Generating MIDI sample for epoch " << epoch << "..."
                << std::endl;
      generated = sample(m, {}, 0);
      std::cout << "[*] MIDI sample successfully generated" << std::endl;

      if (midiout != NULL) {
        std::cout << "[*] Now playing: MIDI sample for epoch " << epoch
                  << std::endl;
        play_midi(midiout, generated);
      } else {
        strcpy(path, "sample.mid");
        generated.write(path);
        std::cout << "[*] MIDI sample saved to " << path << std::endl;
      }
    }
  }
  delete m;
  delete midiout;
  return 0;
}

Eigen::MatrixXd map(Eigen::MatrixXd M, double (*func)(double)) {

  for (size_t i = 0; i < (size_t)M.rows(); i++)
    for (size_t j = 0; j < (size_t)M.cols(); j++)
      M(i, j) = (*func)(M(i, j));

  return M;
}

Eigen::VectorXd run(model *m, Eigen::VectorXd x) {

  m->a[0] = map(m->wxy[0] * x + m->waa[0] * m->a[0] + m->b[0], ACT_FUNC);
  for (size_t i = 1; i < LAYER_NUM; i++)
    m->a[i] =
        map(m->wxy[i] * m->a[i - 1] + m->waa[i] * m->a[i] + m->b[i], ACT_FUNC);

  return map(m->wxy[LAYER_NUM] * m->a[LAYER_NUM - 1] + m->b[LAYER_NUM],
             ACT_FUNC);
}

double train(model *m, smf::MidiFile *midi, size_t batch_num) {

  size_t batch_size = 0;
  double loss = 0, batch_loss = 0;
  Eigen::MatrixXd grad_wxy[LAYER_NUM + 1] = {}, grad_waa[LAYER_NUM] = {};
  Eigen::VectorXd grad_b[LAYER_NUM + 1] = {};

  for (size_t i = 1; i < LAYER_NUM; i++)
    grad_wxy[i] = Eigen::MatrixXd::Zero(H_SIZE, H_SIZE);
  grad_wxy[0] = Eigen::MatrixXd::Zero(H_SIZE, IO_SIZE);
  grad_wxy[LAYER_NUM] = Eigen::MatrixXd::Zero(IO_SIZE, H_SIZE);

  for (size_t i = 0; i < LAYER_NUM; i++)
    grad_waa[i] = Eigen::MatrixXd::Zero(H_SIZE, H_SIZE);

  for (size_t i = 0; i < LAYER_NUM; i++)
    grad_b[i] = Eigen::VectorXd::Zero(H_SIZE);
  grad_b[LAYER_NUM] = Eigen::VectorXd::Zero(IO_SIZE);

  for (size_t batch = 0; batch < batch_num; batch++) {

    batch_size = 0;
    batch_loss = 0;

    if (batch != 0 && batch % MINI_BATCH_SIZE == 0) {
      // TODO: finalise loss if needed, calculate gradient functions, and print
      // a pretty message if need be
      loss /= MINI_BATCH_SIZE;
      for (size_t i = 0; i < LAYER_NUM + 1; i++) {
        m->wxy[i] += grad_wxy[i], m->b[i] += grad_b[i];
        grad_wxy[i].setZero(), grad_b[i].setZero();
      }
      for (size_t i = 0; i < LAYER_NUM; i++) {
        m->waa[i] += grad_waa[i];
        grad_waa[i].setZero();
      }
      loss = 0;
    }

    if (!midi[batch].status()) {
      std::cerr << "[!] train: Unable to parse MIDI data from arguments, "
                   "skipping: batch "
                << batch << std::endl;
      continue;
    }

    midi[batch].absoluteTicks();
    midi[batch].joinTracks();
    midi[batch].doTimeAnalysis();
    midi[batch].linkNotePairs();

    Eigen::VectorXd x(IO_SIZE), y(IO_SIZE);
    double ptime = 0;
    bool init = false;

    for (size_t i = 0; i < (size_t)midi[batch][0].size(); i++)
      if (midi[batch][0][i].isNoteOn()) {

        batch_size++;

        x << midi[batch][0][i].seconds - ptime,
            midi[batch][0][i].getDurationInSeconds(), midi[batch][0][i][1],
            midi[batch][0][i][2];

        if (init)
          for (size_t j = 0; j < IO_SIZE; j++)
            batch_loss += (x(j) - y(j)) * (x(j) - y(j));

        y = run(m, x);
        ptime = midi[batch][0][i].seconds;
        init = true;
      }

    batch_loss /= batch_size;
    loss += batch_loss;
  }

  loss /= (batch_num % MINI_BATCH_SIZE);

  for (size_t i = 0; i < LAYER_NUM + 1; i++) {
    m->wxy[i] += grad_wxy[i], m->b[i] += grad_b[i];
    grad_wxy[i].setZero(), grad_b[i].setZero();
  }
  for (size_t i = 0; i < LAYER_NUM; i++) {
    m->waa[i] += grad_waa[i];
    grad_waa[i].setZero();
  }

  return loss;
}

smf::MidiFile sample(model *m, Eigen::VectorXd *seq, int seqsize) {

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

    x = run(m, x);
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

void play_midi(RtMidiOut *midiout, smf::MidiFile midi) {

  if (!midi.status()) {
    std::cerr << "[!] train: Unable to process MIDI data from arguments"
              << std::endl;
    return;
  }

  midi.absoluteTicks();
  midi.joinTracks();
  midi.doTimeAnalysis();

  double psec = 0;

  for (size_t i = 0; i < (size_t)midi[0].size(); i++) {
    sleep(midi[0][i].seconds - psec);
    midiout->sendMessage(&midi[0][i]);
    psec = midi[0][i].seconds;
  }

  return;
}
