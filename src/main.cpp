#include "eigen/Dense"
#include "midifile/MidiFile.h"
#include "rtmidi/RtMidi.h"
#include "json/json.hpp"
#include <cstring>
#include <iostream>
#include <thread>

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
#define NORM_FUNC relu1
#define LEARNING_RATE 0.01
#define NOTE_ON 0x90
#define T_MIN 0
#define T_MAX 10
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
inline double relu1(double x) { return min(max(x, 0), 1); }

Eigen::MatrixXd map(Eigen::MatrixXd M, void *func);
model *mset_zero(model *m);
model *mset_random(model *m);
Eigen::VectorXd predict(model *m, Eigen::VectorXd x);
Eigen::VectorXd output(Eigen::VectorXd x);
void play_midi(RtMidiOut *midiout, smf::MidiFile midi);
double loss(model m, smf::MidiFile *midi);
model *grad_desc(model *m, model *grad, double cost, size_t t);
smf::MidiFile sample(model m, Eigen::VectorXd *seq, int seqsize);
void train(model *m, smf::MidiFile *midi, size_t batch_size,
           RtMidiOut *midiout);

int main(int argc, char **argv) {

#ifdef LOGFILE
  std::cout << "[*] LOGFILE defined, redirecting output to " << LOGFILE
            << std::endl;
  freopen(LOGFILE, "w", stdout);
  freopen(LOGFILE, "w", stderr);
#endif /* LOGFILE */

  model *m = mset_random(new model);

  std::cout
      << "[*] Model layers, weights and biases initialised with random values"
      << std::endl;

  char path[MAX_PATH_LEN] = "";

  strcpy(path, DATA_DIR);
  strcat(path, METADATA_FILE);
  nlohmann::json md = nlohmann::json::parse(std::ifstream(path));
  std::cout << "[*] Metadata parsed from " << path << std::endl;

  smf::MidiFile midi[MAX_FILE_NUM] = {};
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

  train(m, midi, batch_num, midiout);

  std::cout << "[*] Model successfully trained over " << EPOCH_NUM
            << " epochs, quitting..." << std::endl;

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

model *mset_zero(model *m) {

  for (size_t i = 0; i < LAYER_NUM; i++)
    m->a[i] = Eigen::VectorXd::Zero(H_SIZE);

  for (size_t i = 1; i < LAYER_NUM; i++)
    m->wxy[i] = Eigen::MatrixXd::Zero(H_SIZE, H_SIZE);
  m->wxy[0] = Eigen::MatrixXd::Zero(H_SIZE, IO_SIZE);
  m->wxy[LAYER_NUM] = Eigen::MatrixXd::Zero(IO_SIZE, H_SIZE);

  for (size_t i = 0; i < LAYER_NUM; i++)
    m->waa[i] = Eigen::MatrixXd::Zero(H_SIZE, H_SIZE);

  for (size_t i = 0; i < LAYER_NUM; i++)
    m->b[i] = Eigen::VectorXd::Zero(H_SIZE);
  m->b[LAYER_NUM] = Eigen::VectorXd::Zero(IO_SIZE);

  return m;
}

model *mset_random(model *m) {

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

  return m;
}

Eigen::VectorXd predict(model *m, Eigen::VectorXd x) {

  m->a[0] = map(m->wxy[0] * x + m->waa[0] * m->a[0] + m->b[0], ACT_FUNC);
  for (size_t i = 1; i < LAYER_NUM; i++)
    m->a[i] =
        map(m->wxy[i] * m->a[i - 1] + m->waa[i] * m->a[i] + m->b[i], ACT_FUNC);

  return m->wxy[LAYER_NUM] * m->a[LAYER_NUM - 1] + m->b[LAYER_NUM];
}

Eigen::VectorXd output(Eigen::VectorXd x) {

  x = map(x, NORM_FUNC);
  x(0) = round(x(0) * (T_MAX - T_MIN) + T_MIN),
  x(1) = round(x(1) * (T_MAX - T_MIN) + T_MIN),
  x(2) = round(x(2) * (N_MAX - N_MIN) + N_MIN),
  x(3) = round(x(3) * (A_MAX - A_MIN) + A_MIN);
  return x;
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

double loss(model m, smf::MidiFile midi) {

  if (!midi.status()) {
    std::cerr
        << "[!] Unable to parse MIDI data from arguments, skipping batch..."
        << std::endl;
    return 0;
  }

  size_t batch_size = 0;
  double batch_loss = 0;

  midi.absoluteTicks();
  midi.joinTracks();
  midi.doTimeAnalysis();
  midi.linkNotePairs();

  Eigen::VectorXd x(IO_SIZE), y(IO_SIZE);
  double ptime = 0;
  bool init = false;

  for (size_t i = 0; i < (size_t)midi[0].size(); i++)
    if (midi[0][i].isNoteOn()) {

      batch_size++;

      x << ((midi[0][i].seconds - ptime) - T_MIN) / (T_MAX - T_MIN),
          (midi[0][i].getDurationInSeconds() - T_MIN) / (T_MAX - T_MIN),
          ((double)midi[0][i][1] - N_MIN) / (N_MAX - N_MIN),
          ((double)midi[0][i][2] - A_MIN) / (A_MAX - A_MIN);

      if (init)
        for (size_t j = 0; j < IO_SIZE; j++)
          batch_loss += (x(j) - y(j)) * (x(j) - y(j));

      y = predict(&m, x);
      ptime = midi[0][i].seconds;
      init = true;
    }

  return batch_loss / batch_size;
}

model *grad_desc(model *m, model *grad, double cost, size_t t) { return grad; }

smf::MidiFile sample(model m, Eigen::VectorXd *seq, int seqsize) {

  smf::MidiFile midi;
  std::vector<uint8_t> note{NOTE_ON, 0, 0};
  Eigen::VectorXd x(IO_SIZE), n(IO_SIZE);
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

      std::cout << "[*] generate: Recorded pre-entered note " << (int)seq[i](2)
                << " of attack " << (int)seq[i](3) << " ON to "
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

    std::cout << "[*] generate: Recorded pre-entered note " << (int)x(2)
              << " of attack " << (int)x(3) << " ON to "
              << (int)((atime + x(0)) * TPQ * BPM / 60) << " and OFF to "
              << (int)((atime + x(0) + x(1)) * TPQ * BPM / 60) << std::endl;

    atime += x(0);
  }

  for (size_t i = 0; i < SAMPLE_SIZE; i++) {

    x = predict(&m, x);
    n = output(x);

    note[1] = n(2), note[2] = n(3);
    midi.addEvent(1, (int)((atime + n(0)) * TPQ * BPM / 60), note);

    note[2] = 0;
    midi.addEvent(1, (int)((atime + n(0) + n(1)) * TPQ * BPM / 60), note);

    std::cout << "[*] generate: Recorded generated note " << (int)n(2)
              << " of attack " << (int)n(3) << " ON to "
              << (int)((atime + n(0)) * TPQ * BPM / 60) << " and OFF to "
              << (int)((atime + n(0) + n(1)) * TPQ * BPM / 60) << std::endl;

    atime += n(0);
  }

  midi.sortTracks();
  return midi;
}

void train(model *m, smf::MidiFile *midi, size_t batch_num,
           RtMidiOut *midiout) {

  double cost = 0;
  model *grad = mset_zero(new model);
  smf::MidiFile sample_midi;
  char path[MAX_PATH_LEN] = "";

  for (size_t epoch = 0; epoch < EPOCH_NUM; epoch++) {

    if (epoch != 0 && epoch % EPOCH_SAMPLE_PERIOD == 0) {
      std::cout << "[*] Generating MIDI sample for epoch " << epoch + 1 << "..."
                << std::endl;
      sample_midi = sample(*m, {}, 0);
      std::cout << "[*] MIDI sample successfully generated" << std::endl;

      if (midiout != NULL) {
        std::thread(play_midi, midiout, sample_midi).detach();
        std::cout << "[*] Now playing: MIDI sample for epoch " << epoch + 1
                  << std::endl;
      } else {
        strcpy(path, "sample.mid");
        sample_midi.write(path);
        std::cout << "[*] MIDI sample saved to " << path << std::endl;
      }
    }

    for (size_t batch = 0; batch < batch_num; batch++) {

      if (batch != 0 && batch % MINI_BATCH_SIZE == 0) {
        cost /= MINI_BATCH_SIZE;

        grad_desc(m, grad, cost, MINI_BATCH_SIZE);

        for (size_t i = 0; i < LAYER_NUM + 1; i++) {
          m->wxy[i] += grad->wxy[i], m->b[i] += grad->b[i];
          grad->wxy[i].setZero(), grad->b[i].setZero();
        }
        for (size_t i = 0; i < LAYER_NUM; i++) {
          m->waa[i] += grad->waa[i];
          grad->waa[i].setZero();
        }

        cost = 0;

        std::cout << "[*] Model updated over " << MINI_BATCH_SIZE
                  << " mini-batches on batch " << batch << std::endl;
      }

      if (!midi[batch].status()) {
        std::cerr << "[!] train: Unable to parse MIDI data from arguments, "
                     "skipping: batch "
                  << batch + 1 << std::endl;
        continue;
      }
      cost += loss(*m, midi[batch]);
    }

    cost /= ((batch_num % MINI_BATCH_SIZE) ? batch_num % MINI_BATCH_SIZE
                                           : MINI_BATCH_SIZE);

    grad_desc(m, grad, cost,
              (batch_num % MINI_BATCH_SIZE) ? batch_num % MINI_BATCH_SIZE
                                            : MINI_BATCH_SIZE);

    for (size_t i = 0; i < LAYER_NUM + 1; i++) {
      m->wxy[i] += grad->wxy[i], m->b[i] += grad->b[i];
      grad->wxy[i].setZero(), grad->b[i].setZero();
    }
    for (size_t i = 0; i < LAYER_NUM; i++) {
      m->waa[i] += grad->waa[i];
      grad->waa[i].setZero();
    }

    std::cout << "[*] Model updated over "
              << ((batch_num % MINI_BATCH_SIZE) ? batch_num % MINI_BATCH_SIZE
                                                : MINI_BATCH_SIZE)
              << " mini-batches on batch " << batch_num << std::endl;

    std::cout << "[*] Epoch " << epoch + 1 << "/" << EPOCH_NUM << " : Loss "
              << cost << std::endl;

    cost = 0;
  }

  std::cout << "[*] Generating MIDI sample for training results..."
            << std::endl;
  sample_midi = sample(*m, {}, 0);
  std::cout << "[*] MIDI sample successfully generated" << std::endl;

  if (midiout != NULL) {
    std::thread(play_midi, midiout, sample_midi).join();
    std::cout << "[*] Now playing: MIDI sample for training results..."
              << std::endl;
  } else {
    strcpy(path, "sample.mid");
    sample_midi.write(path);
    std::cout << "[*] MIDI sample saved to " << path << std::endl;
  }

  delete grad;
}
