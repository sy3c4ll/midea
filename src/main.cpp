#include "eigen/Dense"
#include "midifile/MidiFile.h"
#include "rtmidi/RtMidi.h"
#include "json/json.hpp"
#include <cstring>
#include <iomanip>
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

#define LAYER_NUM 3
#define IO_SIZE 4
#define H_SIZE 10
#define EPOCH_NUM 500
#define MINI_BATCH_SIZE 100
#define EPOCH_SAMPLE_PERIOD 10
#define SAMPLE_SIZE 200
#define MAX_BATCH_SIZE 100000
#define ACT_FUNC tanh
#define LEARNING_RATE 0.01
#define NOTE_ON 0x90
#define T_MIN 0
#define T_MAX 3
#define N_MIN 0x00
#define N_MAX 0x7f
#define A_MIN 0x01
#define A_MAX 0x7f
#define MAX_PATH_LEN 0xff
#define MAX_FILE_NUM 200000
#define DATA_DIR "./data/"
#define MODEL_DIR "./models/"
#define SAMPLE_DIR "./samples/"
#define METADATA_FILE "METADATA"
#define MIDI_OUT 1
#define TPQ 960
#define BPM 120

typedef struct model {
  Eigen::MatrixXd wxy[LAYER_NUM + 1], waa[LAYER_NUM];
  Eigen::VectorXd b[LAYER_NUM + 1], a[LAYER_NUM];
} model;

inline double min(double x, double y) { return x < y ? x : y; }
inline double max(double x, double y) { return x > y ? x : y; }
inline double cap(double x) { return min(max(x, -1), 1); }

Eigen::MatrixXd map(Eigen::MatrixXd M, void *func);
model *mset_zero(model *x);
model *mset_random(model *x);
model *mop_add(model *x, model y);
model *mop_sub(model *x, model y);
model *mop_mult(model *x, double y);
model *mop_div(model *x, double y);
model *msv_load(model *x, char *path);
model *msv_save(model *x, char *path);
Eigen::VectorXd predict(model *m, Eigen::VectorXd x);
Eigen::VectorXd encode(Eigen::VectorXd x);
Eigen::VectorXd decode(Eigen::VectorXd x);
void play_midi(RtMidiOut *midiout, smf::MidiFile midi);
smf::MidiFile sample(model m, Eigen::VectorXd *seq, int seqsize);
model grad_desc(model m, smf::MidiFile *midi, double *loss);
void train(model *m, smf::MidiFile *midi, size_t batch_size, size_t sepoch,
           RtMidiOut *midiout);

int main(int argc, char **argv) {

#ifdef LOGFILE
  std::cout << "[*] LOGFILE defined, redirecting output to " << LOGFILE
            << std::endl;
  freopen(LOGFILE, "w", stdout);
  freopen(LOGFILE, "w", stderr);
#endif /* LOGFILE */

  char path[MAX_PATH_LEN] = "";

  model *m = mset_random(new model);
  sprintf(path, "%smodel-%s.json", MODEL_DIR, argv[1]);
  // msv_load(m, path);

  std::cout << "[*] Model layers, weights and biases initialised" << std::endl;

  sprintf(path, "%s%s", DATA_DIR, METADATA_FILE);
  FILE *md = fopen(path, "r");
  // nlohmann::json md = nlohmann::json::parse(std::ifstream(path));
  if (md)
    std::cout << "[*] Metadata parsed from " << path << std::endl;
  else {
    std::cerr
        << "[!] Unable to open metadata file, check if it exists. Exiting..."
        << std::endl;
    exit(EXIT_FAILURE);
  }

  smf::MidiFile *midi = new smf::MidiFile[MAX_FILE_NUM];
  char tmp[MAX_PATH_LEN] = "";
  size_t batch_num = 0;

  while (fgets(tmp, MAX_PATH_LEN - 1, md)) {

    sprintf(path, "%s%s", DATA_DIR, tmp);
    path[strlen(path) - 1] = '\0';
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

#ifdef MIDI_OUT
  try {
    midiout = new RtMidiOut();
    midiout->openPort(MIDI_OUT);
  } catch (RtMidiError &e) {
    std::cerr << "[!] Unable to open MIDI output ports, exiting..."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!midiout->isPortOpen()) {
    std::cerr << "[!] Unable to open MIDI output ports, exiting..."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "[*] Connected to MIDI output port " << MIDI_OUT << std::endl;
#endif

  train(m, midi, batch_num, (size_t)atoi(argv[1]), midiout);

  std::cout << "[*] Model successfully trained over " << EPOCH_NUM
            << " epochs, quitting..." << std::endl;

  delete m;
  delete[] midi;
  delete midiout;
  return 0;
}

Eigen::MatrixXd map(Eigen::MatrixXd M, double (*func)(double)) {

  for (size_t i = 0; i < (size_t)M.rows(); i++)
    for (size_t j = 0; j < (size_t)M.cols(); j++)
      M(i, j) = (*func)(M(i, j));

  return M;
}

model *mset_zero(model *x) {

  for (size_t i = 0; i < LAYER_NUM; i++)
    x->a[i] = Eigen::VectorXd::Zero(H_SIZE);

  for (size_t i = 1; i < LAYER_NUM; i++)
    x->wxy[i] = Eigen::MatrixXd::Zero(H_SIZE, H_SIZE);
  x->wxy[0] = Eigen::MatrixXd::Zero(H_SIZE, IO_SIZE);
  x->wxy[LAYER_NUM] = Eigen::MatrixXd::Zero(IO_SIZE, H_SIZE);

  for (size_t i = 0; i < LAYER_NUM; i++)
    x->waa[i] = Eigen::MatrixXd::Zero(H_SIZE, H_SIZE);

  for (size_t i = 0; i < LAYER_NUM; i++)
    x->b[i] = Eigen::VectorXd::Zero(H_SIZE);
  x->b[LAYER_NUM] = Eigen::VectorXd::Zero(IO_SIZE);

  return x;
}

model *mset_random(model *x) {

  for (size_t i = 0; i < LAYER_NUM; i++)
    x->a[i] = Eigen::VectorXd::Random(H_SIZE);

  for (size_t i = 1; i < LAYER_NUM; i++)
    x->wxy[i] = Eigen::MatrixXd::Random(H_SIZE, H_SIZE);
  x->wxy[0] = Eigen::MatrixXd::Random(H_SIZE, IO_SIZE);
  x->wxy[LAYER_NUM] = Eigen::MatrixXd::Random(IO_SIZE, H_SIZE);

  for (size_t i = 0; i < LAYER_NUM; i++)
    x->waa[i] = Eigen::MatrixXd::Random(H_SIZE, H_SIZE);

  for (size_t i = 0; i < LAYER_NUM; i++)
    x->b[i] = Eigen::VectorXd::Random(H_SIZE);
  x->b[LAYER_NUM] = Eigen::VectorXd::Random(IO_SIZE);

  return x;
}

model *mop_add(model *x, model y) {

  for (size_t i = 0; i < LAYER_NUM + 1; i++)
    x->wxy[i] += y.wxy[i], x->b[i] += y.b[i];
  for (size_t i = 0; i < LAYER_NUM; i++)
    x->waa[i] += y.waa[i], x->a[i] += y.a[i];
  return x;
}

model *mop_sub(model *x, model y) {

  for (size_t i = 0; i < LAYER_NUM + 1; i++)
    x->wxy[i] -= y.wxy[i], x->b[i] -= y.b[i];
  for (size_t i = 0; i < LAYER_NUM; i++)
    x->waa[i] -= y.waa[i], x->a[i] -= y.a[i];
  return x;
}

model *mop_mult(model *x, double y) {

  for (size_t i = 0; i < LAYER_NUM + 1; i++)
    x->wxy[i] *= y, x->b[i] *= y;
  for (size_t i = 0; i < LAYER_NUM; i++)
    x->waa[i] *= y, x->a[i] *= y;
  return x;
}

model *mop_div(model *x, double y) {

  for (size_t i = 0; i < LAYER_NUM + 1; i++)
    x->wxy[i] /= y, x->b[i] /= y;
  for (size_t i = 0; i < LAYER_NUM; i++)
    x->waa[i] /= y, x->a[i] /= y;
  return x;
}

model *msv_load(model *x, char *path) {

  nlohmann::json sv = nlohmann::json::parse(std::ifstream(path));
  for (size_t i = 0; i < LAYER_NUM + 1; i++)
    for (size_t j = 0; j < (size_t)x->wxy[i].rows(); j++)
      for (size_t k = 0; k < (size_t)x->wxy[i].cols(); k++)
        x->wxy[i](j, k) = sv["wxy"][i][j][k];
  for (size_t i = 0; i < LAYER_NUM; i++)
    for (size_t j = 0; j < (size_t)x->waa[i].rows(); j++)
      for (size_t k = 0; k < (size_t)x->waa[i].cols(); k++)
        x->waa[i](j, k) = sv["waa"][i][j][k];
  for (size_t i = 0; i < LAYER_NUM + 1; i++)
    for (size_t j = 0; j < (size_t)x->b[i].rows(); j++)
      x->b[i](j) = sv["b"][i][j];
  for (size_t i = 0; i < LAYER_NUM; i++)
    for (size_t j = 0; j < (size_t)x->a[i].rows(); j++)
      x->a[i](j) = sv["a"][i][j];
  return x;
}

model *msv_save(model *x, char *path) {

  nlohmann::json sv;
  for (size_t i = 0; i < LAYER_NUM + 1; i++)
    for (size_t j = 0; j < (size_t)x->wxy[i].rows(); j++)
      for (size_t k = 0; k < (size_t)x->wxy[i].cols(); k++)
        sv["wxy"][i][j][k] = x->wxy[i](j, k);
  for (size_t i = 0; i < LAYER_NUM; i++)
    for (size_t j = 0; j < (size_t)x->waa[i].rows(); j++)
      for (size_t k = 0; k < (size_t)x->waa[i].cols(); k++)
        sv["waa"][i][j][k] = x->waa[i](j, k);
  for (size_t i = 0; i < LAYER_NUM + 1; i++)
    for (size_t j = 0; j < (size_t)x->b[i].rows(); j++)
      sv["b"][i][j] = x->b[i](j);
  for (size_t i = 0; i < LAYER_NUM; i++)
    for (size_t j = 0; j < (size_t)x->a[i].rows(); j++)
      sv["a"][i][j] = x->a[i](j);
  std::ofstream(path) << std::setw(2) << sv << std::endl;
  return x;
}

Eigen::VectorXd predict(model *m, Eigen::VectorXd x) {

  m->a[0] = map(m->wxy[0] * x + m->waa[0] * m->a[0] + m->b[0], ACT_FUNC);
  for (size_t i = 1; i < LAYER_NUM; i++)
    m->a[i] =
        map(m->wxy[i] * m->a[i - 1] + m->waa[i] * m->a[i] + m->b[i], ACT_FUNC);

  return map(m->wxy[LAYER_NUM] * m->a[LAYER_NUM - 1] + m->b[LAYER_NUM],
             ACT_FUNC);
}

Eigen::VectorXd encode(Eigen::VectorXd x) {

  x(0) = (x(0) - T_MIN) / (T_MAX - T_MIN) * 2 - 1;
  x(1) = (x(1) - T_MIN) / (T_MAX - T_MIN) * 2 - 1;
  x(2) = (x(2) - N_MIN) / (N_MAX - N_MIN) * 2 - 1;
  x(3) = (x(3) - A_MIN) / (A_MAX - A_MIN) * 2 - 1;
  return x;
}

Eigen::VectorXd decode(Eigen::VectorXd x) {

  x(0) = (x(0) + 1) * (T_MAX - T_MIN) / 2 + T_MIN;
  x(1) = (x(1) + 1) * (T_MAX - T_MIN) / 2 + T_MIN;
  x(2) = round((x(2) + 1) * (N_MAX - N_MIN) / 2 + N_MIN);
  x(3) = round((x(3) + 1) * (A_MAX - A_MIN) / 2 + A_MIN);
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
                << atime + seq[i](0) << " and OFF to "
                << atime + seq[i](0) + seq[i](1) << std::endl;

      atime += seq[i](0);
    }

    x = seq[seqsize - 1];

  } else {

    x = decode(Eigen::VectorXd::Random(4));

    note[1] = x(2), note[2] = x(3);
    midi.addEvent(1, (int)((atime + x(0)) * TPQ * BPM / 60), note);

    note[2] = 0;
    midi.addEvent(1, (int)((atime + x(0) + x(1)) * TPQ * BPM / 60), note);

    std::cout << "[*] generate: Recorded pre-entered note " << (int)x(2)
              << " of attack " << (int)x(3) << " ON to " << atime + x(0)
              << " and OFF to " << atime + x(0) + x(1) << std::endl;

    atime += x(0);
  }

  for (size_t i = 0; i < SAMPLE_SIZE; i++) {

    x = predict(&m, x);
    n = decode(map(x, cap));

    note[1] = n(2), note[2] = n(3);
    midi.addEvent(1, (int)((atime + n(0)) * TPQ * BPM / 60), note);

    note[2] = 0;
    midi.addEvent(1, (int)((atime + n(0) + n(1)) * TPQ * BPM / 60), note);

    std::cout << "[*] generate: Recorded generated note " << (int)n(2)
              << " of attack " << (int)n(3) << " ON to " << atime + n(0)
              << " and OFF to " << atime + n(0) + n(1) << std::endl;

    atime += n(0);
  }

  midi.sortTracks();
  return midi;
}

model grad_desc(model m, smf::MidiFile midi, double *loss) {

  model grad;
  mset_zero(&grad);

  if (!midi.status()) {
    std::cerr
        << "[!] Unable to parse MIDI data from arguments, skipping batch..."
        << std::endl;
    return grad;
  }

  midi.absoluteTicks();
  midi.joinTracks();
  midi.doTimeAnalysis();
  midi.linkNotePairs();

  size_t batch_size = 0;
  Eigen::VectorXd x[MAX_BATCH_SIZE], y[MAX_BATCH_SIZE],
      a[MAX_BATCH_SIZE][LAYER_NUM], d;
  double ptime = 0;

  *loss = 0;

  for (size_t i = 0; i < (size_t)midi[0].size(); i++)
    if (midi[0][i].isNoteOn()) {

      batch_size++;

      for (size_t j = 0; j < LAYER_NUM; j++)
        a[batch_size - 1][j] = m.a[j];

      x[batch_size - 1].resize(IO_SIZE);
      x[batch_size - 1] << midi[0][i].seconds - ptime,
          midi[0][i].getDurationInSeconds(), midi[0][i][1], midi[0][i][2];
      x[batch_size - 1] = encode(x[batch_size - 1]);

      y[batch_size - 1] = predict(&m, x[batch_size - 1]);

      if (batch_size > 1)
        for (size_t j = 0; j < IO_SIZE; j++)
          *loss += (x[batch_size - 1](j) - y[batch_size - 2](j)) *
                   (x[batch_size - 1](j) - y[batch_size - 2](j)) / IO_SIZE;

      ptime = midi[0][i].seconds;
    }

  *loss /= batch_size;

  for (size_t i = batch_size; i > 0; i--) {

    if (i > 1) {
      d = y[i - 2] - x[i - 1];
      for (size_t k = 0; k < IO_SIZE; k++)
        d(k) *= 1 - y[i - 2](k) * y[i - 2](k);
    }

    for (size_t j = LAYER_NUM; j > 0; j--) {

      if (i < batch_size) {
        for (size_t k = 0; k < H_SIZE; k++)
          grad.a[j - 1](k) *= 1 - a[i][j - 1](k) * a[i][j - 1](k);
        grad.waa[j - 1] = grad.a[j - 1] * a[i - 1][j - 1].transpose();
        grad.a[j - 1] = m.waa[j - 1].transpose() * grad.a[j - 1];
      }

      if (i > 1) {
        grad.wxy[j] += d * a[i - 1][j - 1].transpose();
        grad.b[j] += d;
        grad.a[j - 1] += m.wxy[j].transpose() * d;
      }

      d = grad.a[j - 1];
      for (size_t k = 0; k < H_SIZE; k++)
        d(k) *= 1 - a[i - 1][j - 1](k) * a[i - 1][j - 1](k);
    }

    grad.wxy[0] += d * x[i - 1].transpose();
    grad.b[0] += d;
  }

  return *mop_div(&grad, (double)batch_size);
}

void train(model *m, smf::MidiFile *midi, size_t batch_num, size_t sepoch,
           RtMidiOut *midiout) {

  model grad;
  mset_zero(&grad);
  double loss = 0, batch_loss = 0;
  smf::MidiFile sample_midi;
  char path[MAX_PATH_LEN] = "";

  for (size_t epoch = sepoch; epoch < EPOCH_NUM; epoch++) {

    if (epoch != 0 && epoch % EPOCH_SAMPLE_PERIOD == 0) {

      std::cout << "[*] Saving model for epoch " << epoch << "..." << std::endl;
      sprintf(path, "%smodel-%d.json", MODEL_DIR, (int)epoch);
      msv_save(m, path);
      std::cout << "[*] Model successfully saved" << std::endl;

      std::cout << "[*] Generating MIDI sample for epoch " << epoch << "..."
                << std::endl;
      sample_midi = sample(*m, {}, 0);
      std::cout << "[*] MIDI sample successfully generated" << std::endl;

      sprintf(path, "%ssample-%d.mid", SAMPLE_DIR, (int)epoch);
      sample_midi.write(path);
      std::cout << "[*] MIDI sample saved to " << path << std::endl;

      if (midiout != NULL) {
        std::thread(play_midi, midiout, sample_midi).detach();
        std::cout << "[*] Now playing: MIDI sample for epoch " << epoch
                  << std::endl;
      }
    }

    for (size_t batch = 0; batch < batch_num; batch++) {

      if (batch != 0 && batch % MINI_BATCH_SIZE == 0) {

        mop_add(m, *mop_mult(mop_div(&grad, MINI_BATCH_SIZE), -LEARNING_RATE));

        std::cout << "[*] Model updated over " << MINI_BATCH_SIZE
                  << " mini-batches on batch " << batch << ", with loss "
                  << loss / MINI_BATCH_SIZE << std::endl;

        loss = 0;
      }

      if (!midi[batch].status()) {
        std::cerr << "[!] train: Unable to parse MIDI data from arguments, "
                     "skipping: batch "
                  << batch + 1 << std::endl;
        continue;
      }

      mop_add(&grad, grad_desc(*m, midi[batch], &batch_loss));
      loss += batch_loss;
    }

    mop_add(m, *mop_mult(mop_div(&grad, ((batch_num % MINI_BATCH_SIZE)
                                             ? batch_num % MINI_BATCH_SIZE
                                             : MINI_BATCH_SIZE)),
                         -LEARNING_RATE));
    std::cout << "[*] Model updated over "
              << ((batch_num % MINI_BATCH_SIZE) ? batch_num % MINI_BATCH_SIZE
                                                : MINI_BATCH_SIZE)
              << " mini-batches on batch " << batch_num << ", with loss "
              << loss / ((batch_num % MINI_BATCH_SIZE)
                             ? batch_num % MINI_BATCH_SIZE
                             : MINI_BATCH_SIZE)
              << std::endl;

    loss = 0;

    std::cout << "[*] Epoch " << epoch + 1 << "/" << EPOCH_NUM << std::endl;
  }

  std::cout << "[*] Saving model for epoch " << EPOCH_NUM << "..." << std::endl;
  sprintf(path, "%smodel-%d.json", MODEL_DIR, EPOCH_NUM);
  msv_save(m, path);
  std::cout << "[*] Model successfully saved" << std::endl;

  std::cout << "[*] Generating MIDI sample for epoch " << EPOCH_NUM << "..."
            << std::endl;
  sample_midi = sample(*m, {}, 0);
  std::cout << "[*] MIDI sample successfully generated" << std::endl;

  sprintf(path, "%ssample-%d.mid", SAMPLE_DIR, EPOCH_NUM);
  sample_midi.write(path);
  std::cout << "[*] MIDI sample saved to " << path << std::endl;

  if (midiout != NULL) {
    std::thread(play_midi, midiout, sample_midi).join();
    std::cout << "[*] Now playing: MIDI sample for epoch " << EPOCH_NUM << "..."
              << std::endl;
  }
}
